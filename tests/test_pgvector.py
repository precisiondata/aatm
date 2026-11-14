"""PostgreSQL + pgvector: configuration, naming and filter translation, without a database.

The tests against a real PostgreSQL server are in ``test_pgvector_integration.py``.
"""

import json
from enum import Enum

import numpy as np
import pytest

from aatm.pgvector_store import (
    PG_IDENTIFIER_MAX_LENGTH,
    _name_placeholders,
    pg_table_name,
    to_pg_filter,
    to_pg_metadata,
)
from aatm.registries import retrievers as registry
from aatm.vector_db import (
    VECTOR_DB_MODES,
    ChromaConnection,
    PgVectorConnection,
    VectorDatabaseConfigError,
    connection_from_env,
    redact_dsn,
)
from aatm.vector_stores import UnsupportedFilterError

DSN = "postgresql://aatm:s3cret@db.example:5433/terms"


# --- Configuration ---------------------------------------------------------------


def test_pgvector_is_a_vector_database_mode():
    assert "pgvector" in VECTOR_DB_MODES
    connection = connection_from_env("x", env={"AATM_VECTOR_DB": "pgvector"})
    assert isinstance(connection, PgVectorConnection)
    assert connection.mode == "pgvector"
    assert not connection.is_cloud


def test_pgvector_defaults():
    connection = PgVectorConnection.from_env(env={})

    assert connection.dsn == ""  # libpq's defaults and the PG* variables
    assert connection.table_prefix == "aatm_expressions"
    assert connection.ef_search == 100
    assert connection.iterative_scan == "relaxed_order"
    assert connection.maintenance_work_mem is None


def test_pgvector_settings_come_from_the_environment():
    connection = PgVectorConnection.from_env(
        env={
            "AATM_PG_DSN": f" {DSN} ",
            "AATM_PG_TABLE_PREFIX": "terms",
            "AATM_PG_EF_SEARCH": "400",
            "AATM_PG_ITERATIVE_SCAN": "Strict_Order",
            "AATM_PG_MAINTENANCE_WORK_MEM": "4GB",
        }
    )

    assert connection.dsn == DSN
    assert connection.table_name("qwen3-06B") == "terms_qwen3_06b"
    assert (connection.ef_search, connection.iterative_scan) == (400, "strict_order")
    assert connection.maintenance_work_mem == "4GB"


@pytest.mark.parametrize("value", ["0", "1001", "many"])
def test_ef_search_must_be_between_1_and_1000(value):
    with pytest.raises(VectorDatabaseConfigError, match="AATM_PG_EF_SEARCH"):
        PgVectorConnection.from_env(env={"AATM_PG_EF_SEARCH": value})


def test_iterative_scan_must_be_known():
    with pytest.raises(VectorDatabaseConfigError, match="AATM_PG_ITERATIVE_SCAN"):
        PgVectorConnection.from_env(env={"AATM_PG_ITERATIVE_SCAN": "sometimes"})


def test_password_never_appears_in_repr_or_description():
    connection = PgVectorConnection.from_env(env={"AATM_PG_DSN": DSN})

    assert "s3cret" not in repr(connection)
    assert "s3cret" not in connection.describe()
    assert connection.describe() == (
        "PostgreSQL (db.example:5433/terms, tables 'aatm_expressions_*')"
    )


@pytest.mark.parametrize(
    "dsn, expected",
    [
        (DSN, "postgresql://aatm:***@db.example:5433/terms"),
        ("host=h password=s3cret dbname=d", "host=h password=*** dbname=d"),
        ("host=h PASSWORD = 'a b' dbname=d", "host=h PASSWORD = *** dbname=d"),
        ("postgresql://aatm@h/d", "postgresql://aatm@h/d"),
    ],
)
def test_passwords_are_redacted(dsn, expected):
    assert redact_dsn(dsn) == expected


def test_chroma_connection_refuses_pgvector_mode():
    with pytest.raises(VectorDatabaseConfigError, match="not ChromaDB"):
        ChromaConnection.from_env("x", env={"AATM_VECTOR_DB": "pgvector"})


def test_location_is_one_table_per_model(monkeypatch):
    monkeypatch.setenv("AATM_VECTOR_DB", "pgvector")
    monkeypatch.setenv("AATM_PG_DSN", DSN)
    spec = registry.CHROMADB_RETRIEVER_MODEL_REGISTRY["text-embedding-3-small"]

    connection, name = spec.vector_db_location()

    assert isinstance(connection, PgVectorConnection)
    assert name == "aatm_expressions_text_embedding_3_small"


# --- Table names -----------------------------------------------------------------


@pytest.mark.parametrize(
    "prefix, model, expected",
    [
        ("aatm_expressions", "qwen3-06B", "aatm_expressions_qwen3_06b"),
        (
            "aatm-expressions",
            "text-embedding-3-small",
            "aatm_expressions_text_embedding_3_small",
        ),
        ("", "3-small", "t_3_small"),
        ("Terms", "a..b", "terms_a_b"),
    ],
)
def test_table_names_are_safe_identifiers(prefix, model, expected):
    assert pg_table_name(prefix, model) == expected


def test_long_table_names_are_shortened_without_collisions():
    a = pg_table_name("aatm_expressions", "x" * 80 + "a")
    b = pg_table_name("aatm_expressions", "x" * 80 + "b")

    assert len(a) <= PG_IDENTIFIER_MAX_LENGTH
    assert len(b) <= PG_IDENTIFIER_MAX_LENGTH
    assert a != b


def test_table_name_needs_valid_characters():
    with pytest.raises(ValueError, match="table name"):
        pg_table_name("", "--")


# --- Filters ---------------------------------------------------------------------


def contains(key, value):
    return json.dumps({key: value})


@pytest.mark.parametrize(
    "where, sql, params",
    [
        (
            {"std_domain_id": "Measurement"},
            "metadata @> %s::jsonb",
            [contains("std_domain_id", "Measurement")],
        ),
        (
            {"n": {"$eq": 3}},
            "metadata @> %s::jsonb",
            [contains("n", 3)],
        ),
        (
            {"flag": True},
            "metadata @> %s::jsonb",
            ['{"flag": true}'],  # a JSON boolean, not the number 1
        ),
        (
            {"std_vocabulary_id": {"$in": ["LOINC", "SNOMED"]}},
            "(metadata @> %s::jsonb) OR (metadata @> %s::jsonb)",
            [
                contains("std_vocabulary_id", "LOINC"),
                contains("std_vocabulary_id", "SNOMED"),
            ],
        ),
        (
            {
                "$and": [
                    {"std_domain_id": "Measurement"},
                    {"std_vocabulary_id": {"$in": ["LOINC"]}},
                ]
            },
            "(metadata @> %s::jsonb) AND (metadata @> %s::jsonb)",
            [
                contains("std_domain_id", "Measurement"),
                contains("std_vocabulary_id", "LOINC"),
            ],
        ),
        (
            {"$or": [{"a": 1}, {"b": "x"}]},
            "(metadata @> %s::jsonb) OR (metadata @> %s::jsonb)",
            [contains("a", 1), contains("b", "x")],
        ),
        (
            {"a": 1, "b": 2},
            "(metadata @> %s::jsonb) AND (metadata @> %s::jsonb)",
            [contains("a", 1), contains("b", 2)],
        ),
        (
            {"a": {"$ne": "x"}},
            "jsonb_exists(metadata, %s) AND NOT metadata @> %s::jsonb",
            ["a", contains("a", "x")],
        ),
        (
            {"a": {"$nin": ["x", "y"]}},
            "jsonb_exists(metadata, %s) AND NOT ((metadata @> %s::jsonb) OR (metadata @> %s::jsonb))",
            ["a", contains("a", "x"), contains("a", "y")],
        ),
        ({"a": {"$in": []}}, "FALSE", []),
        ({"a": {"$nin": []}}, "jsonb_exists(metadata, %s)", ["a"]),
        (
            {"n": {"$gte": 2}},
            "jsonb_typeof(metadata -> %s) = 'number' AND (metadata ->> %s)::float8 >= %s",
            ["n", "n", 2.0],
        ),
        (
            {"n": {"$gt": 1, "$lt": 5}},
            "(jsonb_typeof(metadata -> %s) = 'number' AND (metadata ->> %s)::float8 > %s)"
            " AND (jsonb_typeof(metadata -> %s) = 'number' AND (metadata ->> %s)::float8 < %s)",
            ["n", "n", 1.0, "n", "n", 5.0],
        ),
    ],
)
def test_filters_translate_to_sql(where, sql, params):
    assert to_pg_filter(where) == (sql, params)


def test_filter_values_are_never_part_of_the_sql():
    where = {"name": "x'; DROP TABLE expressions; --"}

    sql, params = to_pg_filter(where)

    assert "DROP" not in sql
    assert json.loads(params[0]) == where


@pytest.mark.parametrize(
    "where",
    [
        {},
        {"$not": {"a": 1}},
        {"$and": []},
        {"$and": {"a": 1}},
        {"a": {"$contains": "x"}},
        {"a": {"$gt": "b"}},
        {"a": {"$gt": True}},
        {"a": {"$gt": float("nan")}},
        {"a": {"$in": [{"b": 1}]}},
        {"a": None},
        {"a": {}},
    ],
)
def test_unsupported_filters_are_rejected(where):
    with pytest.raises(UnsupportedFilterError):
        to_pg_filter(where)


def test_positional_placeholders_become_named():
    assert _name_placeholders("a = %s AND b = %s", 2) == "a = %(f0)s AND b = %(f1)s"
    assert _name_placeholders("TRUE", 0) == "TRUE"


# --- Metadata --------------------------------------------------------------------


class Origin(Enum):
    STANDARD = "standard_concept"


def test_metadata_keeps_nulls_and_becomes_json_safe():
    metadata = {
        "expression": "x",
        "std_vocabulary_code": None,
        "score": float("nan"),
        "n": np.int64(3),
        "w": np.float32(0.5),
        "flag": np.bool_(True),
        "origin": Origin.STANDARD,
        "tags": ["a", None, 1],
        "other": object,
    }

    converted = to_pg_metadata(metadata)

    assert converted == {
        "expression": "x",
        "std_vocabulary_code": None,
        "score": None,
        "n": 3,
        "w": 0.5,
        "flag": True,
        "origin": "standard_concept",
        "tags": ["a", None, 1],
        "other": str(object),
    }
    json.dumps(converted, allow_nan=False)


# --- Where a connection goes, and why it failed ---------------------------------


def test_target_comes_from_the_connection_string():
    from aatm.vector_db import pg_target

    assert pg_target(DSN, env={}) == "db.example:5433/terms"
    assert pg_target("host=h dbname=d user=u", env={}) == "h:5432/d"


def test_target_falls_back_to_the_pg_variables_then_libpq_defaults(monkeypatch):
    import getpass

    from aatm.vector_db import pg_target

    monkeypatch.setattr(getpass, "getuser", lambda: "me")
    assert pg_target("", env={}) == "localhost:5432/me"
    env = {"PGHOST": "pg", "PGPORT": "6000", "PGDATABASE": "aatm"}
    assert pg_target("", env=env) == "pg:6000/aatm"
    assert pg_target("postgresql://u@h/d", env=env) == "h:6000/d"


def test_connections_time_out_unless_a_timeout_is_set():
    from aatm.vector_db import PG_DEFAULT_CONNECT_TIMEOUT, pg_connect_timeout

    assert pg_connect_timeout("", env={}) == PG_DEFAULT_CONNECT_TIMEOUT
    assert pg_connect_timeout(f"{DSN}?connect_timeout=3", env={}) is None
    assert pg_connect_timeout("host=h connect_timeout=3", env={}) is None
    assert pg_connect_timeout("", env={"PGCONNECT_TIMEOUT": "30"}) is None


def test_a_failed_connection_says_where_and_how_to_fix_it(monkeypatch):
    from aatm.pgvector_store import connection_failure_message

    monkeypatch.delenv("PGHOST", raising=False)
    monkeypatch.delenv("PGPORT", raising=False)
    message = connection_failure_message("", OSError("connection timeout\n expired"))

    assert message.startswith("Could not connect to PostgreSQL at localhost:5432/")
    assert "connection timeout expired" in message
    assert "AATM_PG_DSN is not set" in message
    assert "AATM_PG_DSN=postgresql://postgres@localhost:5432/aatm" in message
    assert "docker run" in message and "pgvector/pgvector:pg17" in message


def test_a_failed_connection_never_shows_the_password():
    from aatm.pgvector_store import connection_failure_message

    message = connection_failure_message(DSN, OSError("refused"))

    assert "s3cret" not in message
    assert "db.example:5433/terms" in message
    assert "Check AATM_PG_DSN" in message

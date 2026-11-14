"""`aatm init` asks where to build the vector database and shows what is built."""

import pytest

from aatm import main
from aatm.built_stores import BuiltStore
from aatm.vector_db import vector_db_setup_problem


def installed(monkeypatch, *missing):
    """Pretend every optional package is installed except ``missing``."""
    from aatm import vector_db

    monkeypatch.setattr(
        vector_db, "_installed", lambda *modules: not set(modules) & set(missing)
    )


# --- What each backend needs -----------------------------------------------------


def test_local_needs_nothing():
    assert vector_db_setup_problem("local", env={}) is None


def test_cloud_needs_an_api_key():
    assert vector_db_setup_problem("cloud", env={}) == "set CHROMA_API_KEY"
    assert vector_db_setup_problem("cloud", env={"CHROMA_API_KEY": "k"}) is None


def test_pinecone_needs_the_extra_then_a_key(monkeypatch):
    installed(monkeypatch, "pinecone")
    assert vector_db_setup_problem("pinecone", env={}) == "install aatm[pinecone]"

    installed(monkeypatch)
    assert vector_db_setup_problem("pinecone", env={}) == "set PINECONE_API_KEY"
    assert vector_db_setup_problem("pinecone", env={"PINECONE_API_KEY": "k"}) is None


def test_pgvector_needs_the_extra_but_not_a_dsn(monkeypatch):
    installed(monkeypatch, "psycopg")
    assert vector_db_setup_problem("pgvector", env={}) == "install aatm[pgvector]"

    installed(monkeypatch)
    # Without AATM_PG_DSN, libpq's defaults and the PG* variables apply.
    assert vector_db_setup_problem("pgvector", env={}) is None


# --- The vector database prompt --------------------------------------------------


def store(model, mode="local", count=None, built_at=None):
    return BuiltStore(model, mode, f"somewhere/{model}", count, built_at)


@pytest.fixture
def built(monkeypatch):
    """Built databases by mode, returned by `main.built_in`."""
    stores: dict[str, dict] = {}
    monkeypatch.setattr(
        main, "built_in", lambda mode, quick=False: stores.get(mode, {})
    )
    return stores


def test_every_backend_is_offered_with_what_it_is(built, monkeypatch):
    installed(monkeypatch)

    choices, default = main.vector_db_choices("local", env={})

    assert [c.value for c in choices] == ["local", "cloud", "pinecone", "pgvector"]
    assert choices[0].title == "local     ChromaDB on this machine"
    assert choices[3].title.startswith("pgvector  PostgreSQL with pgvector at ")
    assert default == "local"


def test_backends_that_miss_a_setting_are_disabled_with_the_reason(built, monkeypatch):
    installed(monkeypatch, "pinecone")

    choices, _ = main.vector_db_choices("local", env={})

    disabled = {c.value: c.disabled for c in choices}
    assert disabled == {
        "local": None,
        "cloud": "set CHROMA_API_KEY",
        "pinecone": "install aatm[pinecone]",
        "pgvector": None,
    }


def test_choices_list_the_models_built_in_each_backend(built, monkeypatch):
    installed(monkeypatch)
    built["local"] = {"qwen3-06B": store("qwen3-06B"), "qwen3-4B": store("qwen3-4B")}
    built["pgvector"] = {"embeddinggemma-300M": store("embeddinggemma-300M")}

    choices, _ = main.vector_db_choices("local", env={})

    titles = {c.value: c.title for c in choices}
    assert titles["local"].endswith("(built: qwen3-06B, qwen3-4B)")
    assert titles["pgvector"].endswith("(built: embeddinggemma-300M)")
    assert "built" not in titles["cloud"]


def test_the_configured_backend_is_preselected(built, monkeypatch):
    installed(monkeypatch)

    assert main.vector_db_choices("pgvector", env={})[1] == "pgvector"
    # Not usable without its key: fall back to local.
    assert main.vector_db_choices("cloud", env={})[1] == "local"


def test_remote_backends_are_listed_from_the_records_only(monkeypatch):
    """Building the prompt must not connect to every remote backend."""
    from aatm import built_stores

    asked = []

    def built_vector_databases(mode):
        asked.append(mode)
        return []

    monkeypatch.setattr(built_stores, "built_vector_databases", built_vector_databases)
    monkeypatch.setattr(built_stores, "recorded_vector_databases", lambda mode: [])
    installed(monkeypatch)

    main.vector_db_choices(
        "local", env={"CHROMA_API_KEY": "k", "PINECONE_API_KEY": "k"}
    )

    assert asked == ["local"]


def test_built_in_falls_back_to_the_records_when_the_backend_fails(monkeypatch):
    from aatm import built_stores

    def unreachable(mode):
        raise ConnectionError("down")

    recorded = [BuiltStore("qwen3-06B", "pgvector", "aatm_expressions_qwen3_06b")]
    monkeypatch.setattr(built_stores, "built_vector_databases", unreachable)
    monkeypatch.setattr(
        built_stores, "recorded_vector_databases", lambda mode: recorded
    )

    assert list(main.built_in("pgvector")) == ["qwen3-06B"]


# --- The embedding model prompt --------------------------------------------------


def test_model_choices_mark_the_default_and_the_built_models():
    built = {
        main.DEFAULT_EMBEDDING_MODEL: store(
            main.DEFAULT_EMBEDDING_MODEL,
            count=2_458_830,
            built_at="2026-11-12T10:31:05+00:00",
        ),
        "qwen3-4B": store("qwen3-4B"),
    }

    titles = {c.value: c.title for c in main.embedding_model_choices(built)}

    assert titles[main.DEFAULT_EMBEDDING_MODEL] == (
        f"{main.DEFAULT_EMBEDDING_MODEL} (default; built, 2,458,830 expressions, 2026-11-12)"
    )
    assert titles["qwen3-4B"] == "qwen3-4B (built)"
    assert titles["text-embedding-3-small"] == "text-embedding-3-small"
    assert list(titles) == main.SUPPORTED_EMBEDDING_MODELS


# --- aatm init -------------------------------------------------------------------


@pytest.fixture
def init_until_the_build(monkeypatch, tmp_path):
    """Run `aatm init` up to the vector database step, recording what it would build."""
    import sys

    from aatm import local_database_utils as ldu

    monkeypatch.chdir(tmp_path)
    (tmp_path / "vocabularies").mkdir()
    calls = {}
    monkeypatch.setattr(main, "print_logo", lambda: None)
    monkeypatch.setattr(ldu, "build_local_sqlite_vocab_database", lambda path: None)
    monkeypatch.setattr(ldu, "build_mapping_datasets", lambda vocabs: None)
    monkeypatch.setattr(
        ldu,
        "build_local_vector_database",
        lambda model, embedding_dtype=None, vector_db=None: calls.update(
            model=model, vector_db=vector_db
        ),
    )
    monkeypatch.setattr(main, "built_in", lambda mode, quick=False: {})

    def run(tty: bool, args=()):
        monkeypatch.setattr(sys.stdin, "isatty", lambda: tty, raising=False)
        main.init(
            vocab_dir=None,
            embedding_model="qwen3-06B",
            embedding_dtype=None,
            standard_vocabs=["LOINC"],
            vector_db=None,
            vocab_zip=None,
            force=False,
            umls_api_key=None,
            **dict(args),
        )
        return calls

    return run


def test_init_asks_for_the_vector_database_in_a_terminal(
    init_until_the_build, monkeypatch
):
    monkeypatch.setattr(main, "choose_vector_db", lambda: "local")

    assert init_until_the_build(tty=True)["vector_db"] == "local"


def test_init_uses_the_configured_backend_without_a_terminal(
    init_until_the_build, monkeypatch
):
    def ask():
        raise AssertionError("must not prompt without a terminal")

    monkeypatch.setattr(main, "choose_vector_db", ask)

    assert init_until_the_build(tty=False)["vector_db"] == "local"


def test_cancelling_the_prompt_stops_init(init_until_the_build, monkeypatch):
    import typer

    monkeypatch.setattr(main, "choose_vector_db", lambda: None)

    with pytest.raises(typer.Exit):
        init_until_the_build(tty=True)


def test_the_pgvector_choice_names_the_server_it_will_use(built, monkeypatch):
    installed(monkeypatch)

    choices, _ = main.vector_db_choices(
        "local", env={"AATM_PG_DSN": "postgresql://u:pw@pg.example:5433/terms"}
    )
    title = {c.value: c.title for c in choices}["pgvector"]
    assert title == "pgvector  PostgreSQL with pgvector at pg.example:5433/terms"
    assert "pw" not in title

    choices, _ = main.vector_db_choices("local", env={"PGHOST": "h", "PGDATABASE": "d"})
    title = {c.value: c.title for c in choices}["pgvector"]
    assert (
        title == "pgvector  PostgreSQL with pgvector at h:5432/d (AATM_PG_DSN not set)"
    )


def test_records_of_other_tables_are_not_listed_as_built(monkeypatch):
    """A record made with another table prefix is not what init would build into."""
    from aatm import built_stores

    monkeypatch.setenv("AATM_PG_TABLE_PREFIX", "aatm_expressions")
    recorded = [
        store("qwen3-06B", "pgvector", built_at="2"),
        BuiltStore("qwen3-4B", "pgvector", "aatm_expressions_qwen3_4b", None, "1"),
        BuiltStore("qwen3-06B", "pgvector", "aatm_subset_qwen3_06b", None, "3"),
    ]
    monkeypatch.setattr(
        built_stores, "recorded_vector_databases", lambda mode: recorded
    )

    assert list(main.built_in("pgvector", quick=True)) == ["qwen3-4B"]

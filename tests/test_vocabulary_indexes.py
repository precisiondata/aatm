"""Indexes of the vocabulary database: `aatm init`, `aatm index-vocab` and lookups."""

import logging
import sqlite3

import pytest
from typer.testing import CliRunner

from aatm.omop import vocabulary
from aatm.omop.vocabulary import (
    VOCABULARY_INDEXES,
    SqliteConceptLookup,
    index_vocabulary_database,
    missing_vocabulary_indexes,
)

TABLES = {
    "concept": "concept_id TEXT, concept_name TEXT, domain_id TEXT, vocabulary_id TEXT, "
    "concept_class_id TEXT, standard_concept TEXT, concept_code TEXT, "
    "valid_start_date TEXT, valid_end_date TEXT, invalid_reason TEXT",
    "concept_relationship": "concept_id_1 TEXT, concept_id_2 TEXT, relationship_id TEXT, "
    "valid_start_date TEXT, valid_end_date TEXT, invalid_reason TEXT",
    "concept_synonym": "concept_id TEXT, concept_synonym_name TEXT, language_concept_id TEXT",
    "concept_ancestor": "ancestor_concept_id TEXT, descendant_concept_id TEXT, "
    "min_levels_of_separation TEXT, max_levels_of_separation TEXT",
}


def make_db(path, tables=TABLES):
    con = sqlite3.connect(path)
    for name, columns in tables.items():
        con.execute(f"CREATE TABLE {name} ({columns})")
    con.execute(
        "INSERT INTO concept VALUES ('201826', 'Type 2 diabetes mellitus', 'Condition', "
        "'SNOMED', 'x', 'S', '44054006', '', '', NULL)"
    )
    # Enough rows that, after ANALYZE, the planner prefers the indexes to a scan.
    con.executemany(
        "INSERT INTO concept VALUES (?, ?, 'Condition', 'ICD10CM', 'x', NULL, ?, "
        "'', '', NULL)",
        [(str(i), f"concept {i}", f"C{i}") for i in range(1, 500)],
    )
    if "concept_relationship" in tables:
        con.executemany(
            "INSERT INTO concept_relationship VALUES (?, ?, 'Maps to', '', '', NULL)",
            [(str(i), "201826") for i in range(1, 500)],
        )
    con.commit()
    con.close()
    return path


@pytest.fixture(autouse=True)
def _forget_warnings():
    vocabulary._warned_unindexed.clear()
    yield
    vocabulary._warned_unindexed.clear()


def plan(path, sql):
    with sqlite3.connect(path) as con:
        return " ".join(row[-1] for row in con.execute(f"EXPLAIN QUERY PLAN {sql}"))


def test_index_creates_every_missing_index_once(tmp_path):
    db = make_db(tmp_path / "omop.db")
    assert missing_vocabulary_indexes(db) == list(VOCABULARY_INDEXES)
    seen = []
    assert index_vocabulary_database(db, progress=seen.append) == list(
        VOCABULARY_INDEXES
    )
    assert seen == list(VOCABULARY_INDEXES)
    assert missing_vocabulary_indexes(db) == []
    assert index_vocabulary_database(db) == []  # nothing left to do
    # The lookups use them, including the case-insensitive code lookup.
    assert "idx_aatm_relationship_id1" in plan(
        db,
        "SELECT concept_id_2 FROM concept_relationship "
        "WHERE concept_id_1 = '1' AND relationship_id = 'Maps to'",
    )
    assert "idx_aatm_concept_code_lower" in plan(
        db, "SELECT concept_id FROM concept WHERE lower(concept_code) = 'e11'"
    )
    assert "idx_aatm_concept_id" in plan(
        db, "SELECT concept_name FROM concept WHERE concept_id IN ('1', '2')"
    )


def test_indexes_of_missing_tables_are_skipped(tmp_path):
    db = make_db(tmp_path / "omop.db", {"concept": TABLES["concept"]})
    created = index_vocabulary_database(db)
    assert created == [
        name for name, (table, _) in VOCABULARY_INDEXES.items() if table == "concept"
    ]
    assert missing_vocabulary_indexes(db) == []


def test_lookup_warns_once_about_an_unindexed_database(tmp_path, caplog):
    db = make_db(tmp_path / "omop.db")
    with caplog.at_level(logging.WARNING):
        SqliteConceptLookup(db)
        SqliteConceptLookup(db)
    warnings = [r for r in caplog.records if "aatm index-vocab" in r.getMessage()]
    assert len(warnings) == 1
    assert "lacks 7 index(es)" in warnings[0].getMessage()

    caplog.clear()
    index_vocabulary_database(db)
    with caplog.at_level(logging.WARNING):
        lookup = SqliteConceptLookup(db)
    assert not [r for r in caplog.records if "index" in r.getMessage()]
    assert lookup.by_id([201826])[201826].concept_name == "Type 2 diabetes mellitus"


def test_index_vocab_command(tmp_path):
    from aatm.main import app

    db = make_db(tmp_path / "omop.db")
    result = CliRunner().invoke(app, ["index-vocab", "--path", str(db)])
    assert result.exit_code == 0, result.output
    assert "Added 7 index(es)" in result.output
    result = CliRunner().invoke(app, ["index-vocab", "--path", str(db)])
    assert result.exit_code == 0
    assert "already has every index" in result.output

    result = CliRunner().invoke(
        app, ["index-vocab", "--path", str(tmp_path / "none.db")]
    )
    assert result.exit_code == 1
    assert "aatm init" in result.output


def test_init_indexes_the_database_it_builds(tmp_path, monkeypatch):
    from aatm import local_database_utils as ldu

    vocab_dir = tmp_path / "vocab"
    vocab_dir.mkdir()
    (vocab_dir / "CONCEPT.csv").write_text(
        "concept_id\tconcept_name\tdomain_id\tvocabulary_id\tconcept_class_id\t"
        "standard_concept\tconcept_code\tvalid_start_date\tvalid_end_date\t"
        "invalid_reason\n"
        "201826\tType 2 diabetes mellitus\tCondition\tSNOMED\tx\tS\t44054006\t\t\t\n",
        encoding="utf-8",
    )
    (vocab_dir / "CONCEPT_RELATIONSHIP.csv").write_text(
        "concept_id_1\tconcept_id_2\trelationship_id\tvalid_start_date\t"
        "valid_end_date\tinvalid_reason\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".aatm").mkdir()
    ldu.build_local_sqlite_vocab_database(vocab_dir)
    db = tmp_path / ".aatm" / "omop.db"
    assert missing_vocabulary_indexes(db) == []
    with sqlite3.connect(db) as con:
        names = {r[0] for r in con.execute("SELECT name FROM sqlite_master")}
    assert "idx_aatm_relationship_id1" in names


def test_init_indexes_an_existing_database_it_keeps(tmp_path, monkeypatch):
    from aatm import local_database_utils as ldu

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".aatm").mkdir()
    db = make_db(tmp_path / ".aatm" / "omop.db")

    class Skip:
        def ask(self):
            return "Skip"

    monkeypatch.setattr(ldu.questionary, "select", lambda *a, **k: Skip())
    ldu.build_local_sqlite_vocab_database(tmp_path / "unused")
    assert missing_vocabulary_indexes(db) == []

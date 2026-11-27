"""Importing the Athena vocabulary files into the SQLite database built by `aatm init`."""

import sqlite3

import pytest

from aatm.local_database_utils import (
    build_local_sqlite_vocab_database,
    import_vocab_file,
)

CONCEPT_COLUMNS = [
    "concept_id",
    "concept_name",
    "domain_id",
    "vocabulary_id",
    "concept_class_id",
    "standard_concept",
    "concept_code",
    "valid_start_date",
    "valid_end_date",
    "invalid_reason",
]
CONCEPT_HEADER = "\t".join(CONCEPT_COLUMNS) + "\n"


def concept_line(concept_id, name="x", standard_concept="S"):
    return (
        f"{concept_id}\t{name}\tCondition\tSNOMED\tClinical Finding\t{standard_concept}\t"
        f"C{concept_id}\t19700101\t20991231\t\n"
    )


def write_concepts(path, n):
    rows = "".join(concept_line(i, f"concept {i}") for i in range(n))
    path.write_text(CONCEPT_HEADER + rows, encoding="utf-8")
    return path


def rows(con, sql):
    return con.execute(sql).fetchall()


@pytest.mark.parametrize("chunk_rows", [1, 3, 7, 100])
def test_every_row_is_imported_once_whatever_the_chunk_size(tmp_path, chunk_rows):
    con = sqlite3.connect(tmp_path / "omop.db")
    import_vocab_file(con, write_concepts(tmp_path / "CONCEPT.csv", 7), chunk_rows)

    assert rows(con, "SELECT concept_id, concept_name FROM concept ORDER BY rowid") == [
        (str(i), f"concept {i}") for i in range(7)
    ]


def test_columns_are_text_and_empty_values_null(tmp_path):
    path = tmp_path / "CONCEPT.csv"
    path.write_text(
        CONCEPT_HEADER + concept_line("0042", standard_concept=""), encoding="utf-8"
    )
    con = sqlite3.connect(tmp_path / "omop.db")
    import_vocab_file(con, path, chunk_rows=1)

    assert rows(con, "SELECT concept_id, standard_concept FROM concept") == [
        ("0042", None)
    ]
    assert {t for (_, _, t, *_) in rows(con, "PRAGMA table_info(concept)")} == {"TEXT"}


def test_a_file_with_only_a_header_creates_an_empty_table(tmp_path):
    con = sqlite3.connect(tmp_path / "omop.db")
    import_vocab_file(con, write_concepts(tmp_path / "CONCEPT.csv", 0))

    assert rows(con, "SELECT count(*) FROM concept") == [(0,)]
    assert [
        c for (_, c, *_) in rows(con, "PRAGMA table_info(concept)")
    ] == CONCEPT_COLUMNS


def test_an_existing_table_is_replaced_not_appended_to(tmp_path):
    con = sqlite3.connect(tmp_path / "omop.db")
    import_vocab_file(con, write_concepts(tmp_path / "CONCEPT.csv", 5), chunk_rows=2)
    import_vocab_file(con, write_concepts(tmp_path / "CONCEPT.csv", 3), chunk_rows=2)

    assert rows(con, "SELECT count(*) FROM concept") == [(3,)]


def test_source_to_concept_map_is_comma_separated(tmp_path):
    path = tmp_path / "SOURCE_TO_CONCEPT_MAP.csv"
    path.write_text("source_code,target_concept_id\nA01,201826\n", encoding="utf-8")
    con = sqlite3.connect(tmp_path / "omop.db")
    import_vocab_file(con, path)

    assert rows(
        con, "SELECT source_code, target_concept_id FROM source_to_concept_map"
    ) == [("A01", "201826")]


def test_build_imports_every_file(tmp_path, monkeypatch):
    vocab_dir = tmp_path / "vocab"
    vocab_dir.mkdir()
    write_concepts(vocab_dir / "CONCEPT.csv", 5)
    (vocab_dir / "VOCABULARY.csv").write_text(
        "vocabulary_id\tvocabulary_name\nSNOMED\tSNOMED CT\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".aatm").mkdir()

    build_local_sqlite_vocab_database(vocab_dir)

    con = sqlite3.connect(tmp_path / ".aatm" / "omop.db")
    assert rows(con, "SELECT count(*) FROM concept") == [(5,)]
    assert rows(con, "SELECT vocabulary_id FROM vocabulary") == [("SNOMED",)]

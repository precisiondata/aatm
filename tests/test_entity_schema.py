"""The JSON Schema of entity definition files (aatm.data_models.entity_schema)."""

import json
import logging
from pathlib import Path

import jsonschema
import pytest
import yaml
from typer.testing import CliRunner

from aatm import main
from aatm.data_models.entity_schema import (
    SCHEMA_FILENAME,
    SCHEMA_URL,
    entity_definitions_schema,
    schema_json,
)
from aatm.data_models.extractors import EntityDefinitions
from aatm.logs import LOG_LEVEL_ENV_VAR
from aatm.omop.tables import OMOP_TABLES

ROOT = Path(__file__).resolve().parents[1]
COMMITTED_SCHEMA = ROOT / "docs" / "schemas" / SCHEMA_FILENAME
DEFINITION_FILES = sorted(
    [
        *(ROOT / "aatm" / "omop" / "entity_definitions").glob("*.yaml"),
        *(ROOT / "examples" / "definitions").glob("*.yaml"),
    ]
)
MODELINE = f"# yaml-language-server: $schema={SCHEMA_URL}"


@pytest.fixture(scope="module")
def validator() -> jsonschema.Draft202012Validator:
    schema = entity_definitions_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def errors(validator, entity: dict) -> list[str]:
    return [e.message for e in validator.iter_errors([entity])]


def entity(**omop) -> dict:
    return {"name": "Body weight", "definition": "A weight.", "omop": omop}


def test_committed_schema_is_up_to_date():
    committed = COMMITTED_SCHEMA.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert committed == schema_json(), (
        "docs/schemas/entity_definitions.schema.json is out of date. "
        "Regenerate it with: uv run aatm schema docs/schemas"
    )


@pytest.mark.parametrize("path", DEFINITION_FILES, ids=lambda p: p.name)
def test_shipped_definitions_match_the_schema(validator, path):
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == MODELINE
    assert list(validator.iter_errors(yaml.safe_load(text))) == []
    EntityDefinitions.from_yaml(path)


def test_tables_are_enumerated(validator):
    schema = entity_definitions_schema()
    assert schema["$defs"]["OmopMapping"]["properties"]["table"]["enum"] == list(
        OMOP_TABLES
    )
    assert errors(validator, entity(table="measurment", fields={})) == [
        f"'measurment' is not one of {list(OMOP_TABLES)}"
    ]


def test_misspelt_column_is_reported_on_the_column(validator):
    found = errors(
        validator,
        entity(table="measurement", fields={"value_as_numbr": {"from": "text"}}),
    )
    assert found == [
        "Additional properties are not allowed ('value_as_numbr' was unexpected)"
    ]


def test_each_table_lists_its_columns_without_the_primary_key():
    rules = entity_definitions_schema()["$defs"]["OmopMapping"]["allOf"]
    for rule, (table, model) in zip(rules, OMOP_TABLES.items()):
        assert rule["if"]["properties"]["table"]["const"] == table
        columns = rule["then"]["properties"]["fields"]["properties"]
        assert set(columns) == set(model.model_fields) - {model.primary_key}


@pytest.mark.parametrize(
    "fields",
    [
        {"person_id": {"from": "context.person_id"}},
        {"measurement_source_value": {"from": "text"}},
        {"value_as_number": {"from": "child.weight_value", "as": "float"}},
        {"value_as_concept_id": {"from": "attribute.result", "concepts": {"a": "X:1"}}},
        {"measurement_concept_id": {"concept": "LOINC:29463-7", "maps_to": True}},
        {"measurement_concept_id": {"from": "text", "map": {"min_score": 0.8}}},
        {"unit_concept_id": {"from": "child.unit", "lookup": "UCUM"}},
        {"measurement_type_concept_id": {"value": 32858}},
        {
            "measurement_date": [
                {"from": "child.date", "as": "date", "when": {"attribute.k": ["x"]}},
                {"from": "context.note_date"},
            ]
        },
    ],
)
def test_valid_sources(validator, fields):
    assert errors(validator, entity(table="measurement", fields=fields)) == []


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ({"from": "contxt.person_id"}, "is not valid under any of the given schemas"),
        ({"from": "context.patient"}, "is not valid under any of the given schemas"),
        ({"from": "attribute"}, "is not valid under any of the given schemas"),
        ({"concept": "29463-7"}, "is not valid under any of the given schemas"),
        ({"from": "text", "value": 1}, "is valid under each of"),
        ({"maps_to": True}, "is not valid under any of the given schemas"),
        ({"from": "text", "as": "integer"}, "is not one of"),
        ({"from": "text", "when": {"child.x": ["a"]}}, "does not match"),
        ({"from": "text", "when": {"attribute.k": []}}, "should be non-empty"),
        ({"from": "text", "map": {"min_score": 2}}, "is greater than the maximum"),
        ({"form": "text"}, "Additional properties are not allowed"),
    ],
)
def test_invalid_sources(validator, source, message):
    found = errors(validator, entity(table="measurement", fields={"person_id": source}))
    assert found and message in found[0], found


def test_entity_rules(validator):
    base = {"name": "Dose", "definition": "A dose."}
    assert errors(validator, {**base, "has_parent": True}) == [
        "'parent_key' is a required property"
    ]
    assert errors(validator, {**base, "key": "Dose-Key"}) == [
        "'Dose-Key' does not match '^[a-z0-9]+(?:_[a-z0-9]+)*$'"
    ]
    assert errors(validator, {**base, "atributes": []}) == [
        "Additional properties are not allowed ('atributes' was unexpected)"
    ]


def test_descriptions_come_from_the_docstrings():
    defs = entity_definitions_schema()["$defs"]
    from_ = defs["FieldSource"]["properties"]["from"]
    assert from_["description"].startswith("Where in the note the value comes from")
    assert "``" not in from_["description"]
    assert from_["markdownDescription"] == from_["description"]
    labels = defs["AttributeDefinition"]["properties"]["labels"]
    assert labels["description"] == "The allowed labels."
    assert "Attributes:" not in defs["EntityDefinition"]["description"]


def test_schema_command_writes_the_schema(tmp_path, monkeypatch):
    logger = logging.getLogger("aatm")
    handlers, level = list(logger.handlers), logger.level
    monkeypatch.setenv(LOG_LEVEL_ENV_VAR, "INFO")
    try:
        result = CliRunner().invoke(main.app, ["schema", str(tmp_path / "out")])
    finally:
        logger.handlers[:] = handlers
        logger.setLevel(level)

    assert result.exit_code == 0, result.output
    written = tmp_path / "out" / SCHEMA_FILENAME
    assert json.loads(written.read_text(encoding="utf-8")) == json.loads(schema_json())
    assert f"$schema={SCHEMA_FILENAME}" in result.output

"""JSON Schema for entity definition YAML files, for editor support.

Editors that understand JSON Schema (VS Code with the Red Hat YAML extension,
JetBrains IDEs and others) use it to autocomplete keys and OMOP column names, show
descriptions on hover and underline mistakes while a definition file is written, before
any Python code runs.

The schema is generated from the Pydantic models (`EntityDefinition`,
`AttributeDefinition`, `AttributeLabel`, `OmopMapping`, `FieldSource`, `MapSpec`), so
it cannot drift from the code, and adds what the models alone do not say: the
supported OMOP tables, the columns of each one, the forms of ``from:`` and
``concept:``, and the ``context.`` fields of `ClinicalNote`. The descriptions come from
the ``Attributes:`` sections of the models' docstrings.

Loading a file with `EntityDefinitions.from_yaml` stays the full check: the schema
cannot see, for example, whether ``child.<key>`` names a real child entity.

A copy is committed as ``docs/schemas/entity_definitions.schema.json`` and published
with the documentation at `SCHEMA_URL`. ``aatm schema`` writes it next to your files,
for use offline.
"""

import copy
import inspect
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, TypeAdapter

from aatm.data_models.extractors import (
    AttributeDefinition,
    AttributeLabel,
    EntityDefinition,
    _KEY_PATTERN,
)
from aatm.omop.mapping import CONTEXT_FIELDS, FieldSource, MapSpec, OmopMapping
from aatm.omop.tables import OMOP_TABLES

SCHEMA_FILENAME = "entity_definitions.schema.json"
SCHEMA_URL = f"https://precisiondata.github.io/aatm/latest/schemas/{SCHEMA_FILENAME}"

_DOCUMENTED_MODELS: tuple[type[BaseModel], ...] = (
    EntityDefinition,
    AttributeDefinition,
    AttributeLabel,
    OmopMapping,
    FieldSource,
    MapSpec,
)

# ConceptRef.parse: "<VOCABULARY>:<code>", both parts non-blank.
_CONCEPT_PATTERN = r"^[^:]*[^:\s][^:]*:.*\S.*$"
_KEYED_SOURCE_PATTERN = r"^(attribute|child)\.\S+$"


def _split_docstring(model: type[BaseModel]) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into its summary and its ``Attributes:`` entries.

    Entries are ``name (type): text`` or ``name: text``; indented lines continue them.
    """
    doc = inspect.cleandoc(model.__doc__ or "")
    summary, _, section = doc.partition("\nAttributes:\n")
    fields: dict[str, str] = {}
    current = None
    for line in section.splitlines():
        entry = re.match(r"^\s{4}(\w+)(?:\s*\([^)]*\))?:\s*(.*)$", line)
        if entry:
            current = entry.group(1)
            fields[current] = entry.group(2)
        elif current and line.strip():
            fields[current] += " " + line.strip()
        elif not line.strip():
            current = None
    return summary.strip(), fields


def _drop_titles(node: Any) -> None:
    """Remove the ``title`` Pydantic adds to every model and field: editors show it
    instead of the description, and it only repeats the name."""
    if isinstance(node, dict):
        node.pop("title", None)
        for value in node.values():
            _drop_titles(value)
    elif isinstance(node, list):
        for value in node:
            _drop_titles(value)


def _drop_nulls(node: Any) -> None:
    """Unwrap optional fields: ``anyOf: [X, {type: null}]`` becomes ``X``.

    Leaving a field out is how a definition says "none", and the wrapper would make
    editors report a mistake inside, e.g. a misspelt column, as "not valid under any
    of the given schemas" on the whole ``omop`` block instead of on the column.
    """
    if isinstance(node, dict):
        options = node.get("anyOf")
        if isinstance(options, list) and {"type": "null"} in options:
            options.remove({"type": "null"})
            if len(options) == 1:
                del node["anyOf"]
                node.update(options[0])
        if node.get("default", 0) is None:
            del node["default"]
        for value in node.values():
            _drop_nulls(value)
    elif isinstance(node, list):
        for value in node:
            _drop_nulls(value)


def _add_markdown(node: Any) -> None:
    """Write descriptions with single backticks, and repeat them as
    ``markdownDescription``, which VS Code renders on hover."""
    if isinstance(node, dict):
        if isinstance(node.get("description"), str):
            node["description"] = node["description"].replace("``", "`")
            node["markdownDescription"] = node["description"]
        for value in node.values():
            _add_markdown(value)
    elif isinstance(node, list):
        for value in node:
            _add_markdown(value)


def _column_description(table: str, column: str, prop: dict[str, Any]) -> str:
    model = OMOP_TABLES[table]
    kinds = [
        option.get("format", option.get("type"))
        for option in prop.get("anyOf", [prop])
        if option.get("type") != "null"
    ]
    parts = [f"Column `{column}` of `{table}` ({' or '.join(map(str, kinds))})."]
    if column in model.required_fields:
        parts.append("Required by the CDM.")
    if column in model.concept_domains:
        parts.append(
            f"Holds a concept of the {model.concept_domains[column]!r} domain."
        )
    return " ".join(parts)


def _table_rules() -> list[dict[str, Any]]:
    """One ``if``/``then`` per table: the columns ``fields`` may map."""
    rules = []
    for table, model in OMOP_TABLES.items():
        properties = model.model_json_schema()["properties"]
        columns = {
            column: {
                "$ref": "#/$defs/Sources",
                "description": _column_description(table, column, prop),
            }
            for column, prop in properties.items()
            if column != model.primary_key
        }
        rules.append(
            {
                "if": {
                    "properties": {"table": {"const": table}},
                    "required": ["table"],
                },
                "then": {
                    "properties": {
                        "fields": {
                            "properties": columns,
                            "additionalProperties": False,
                        }
                    }
                },
            }
        )
    return rules


def entity_definitions_schema() -> dict[str, Any]:
    """Build the JSON Schema of an entity definition file.

    Returns:
        A JSON Schema (draft 2020-12) for a YAML list of entity definitions.
    """
    schema = TypeAdapter(list[EntityDefinition]).json_schema(mode="validation")
    defs = schema["$defs"]
    _drop_titles(schema)
    _drop_nulls(schema)

    for model in _DOCUMENTED_MODELS:
        node = defs[model.__name__]
        summary, field_docs = _split_docstring(model)
        node["description"] = summary
        for name, prop in node["properties"].items():
            if name in field_docs:
                prop["description"] = field_docs[name]

    # Entities, attributes and labels: Python ignores unknown keys, but in an editor a
    # misspelt key such as `atributes` is almost always a mistake.
    for name in ("EntityDefinition", "AttributeDefinition", "AttributeLabel"):
        defs[name]["additionalProperties"] = False

    entity = defs["EntityDefinition"]
    entity["properties"]["key"]["pattern"] = _KEY_PATTERN
    entity["if"] = {
        "properties": {"has_parent": {"const": True}},
        "required": ["has_parent"],
    }
    entity["then"] = {"required": ["parent_key"]}

    concept = {
        "type": "string",
        "pattern": _CONCEPT_PATTERN,
        "description": "A concept written as `<VOCABULARY>:<code>`, e.g. `SNOMED:423740007`.",
    }
    concept_ref = {
        "anyOf": [{"$ref": "#/$defs/Concept"}, {"$ref": "#/$defs/ConceptRef"}]
    }
    defs["Concept"] = concept

    source = defs["FieldSource"]
    props = source["properties"]
    del props["from"]["type"]
    props["from"]["anyOf"] = [
        {"enum": ["text", *(f"context.{name}" for name in CONTEXT_FIELDS)]},
        {"type": "string", "pattern": _KEYED_SOURCE_PATTERN},
    ]
    del props["concept"]["$ref"]
    props["concept"].update(copy.deepcopy(concept_ref))
    props["concepts"]["additionalProperties"] = copy.deepcopy(concept_ref)
    props["when"].update(
        propertyNames={"pattern": r"^attribute\.\S+$"},
        additionalProperties={
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
    )
    source["oneOf"] = [{"required": [name]} for name in ("from", "value", "concept")]

    # One source, or a list tried in order. if/then/else rather than anyOf, so a
    # mistake is reported inside the source, not as "matches none of the schemas".
    defs["Sources"] = {
        "if": {"type": "array"},
        "then": {"items": {"$ref": "#/$defs/FieldSource"}, "minItems": 1},
        "else": {"$ref": "#/$defs/FieldSource"},
    }
    mapping = defs["OmopMapping"]
    mapping["properties"]["table"]["enum"] = list(OMOP_TABLES)
    # The columns, and their sources, are given per table by these rules.
    del mapping["properties"]["fields"]["additionalProperties"]
    mapping["allOf"] = _table_rules()

    _add_markdown(defs)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_URL,
        "title": "AATM entity definitions",
        "description": (
            "A list of entities to extract from clinical notes, and optionally how "
            "each one becomes a row of an OMOP CDM table. See "
            "https://precisiondata.github.io/aatm/latest/getting-started/entity_definitions/"
        ),
        **schema,
        "$defs": dict(sorted(defs.items())),
    }


def schema_json() -> str:
    """The schema as the JSON text that is committed and published."""
    return json.dumps(entity_definitions_schema(), indent=2, ensure_ascii=False) + "\n"


def write_schema(directory: str | Path = ".") -> Path:
    """Write the schema to ``<directory>/entity_definitions.schema.json``.

    Args:
        directory: Where to write it, created if missing.

    Returns:
        The path of the written file.
    """
    path = Path(directory) / SCHEMA_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(schema_json(), encoding="utf-8", newline="\n")
    return path

# Information Extraction Quick Start

This page shows how to extract structured information, such as conditions, drugs and
their dates, from clinical documents with the **AATM information extraction pipeline**.

---

## What the pipeline does

Given a clinical note and a set of **entity definitions** (what to extract), the
pipeline asks an LLM to find every relevant mention and then **validates** the answer
against the note. Only spans that really exist in the note survive, each with its exact
character position, linked to the entity it belongs to.

```mermaid
flowchart LR
    A[Clinical note] --> B[Numbered lines]
    D[Entity definitions] --> C
    B --> C[LLM extraction]
    C --> E[Validation]
    E --> F[Validated entities<br/>+ rejected candidates]
    F --> G[Records view]
```

The main pieces are:

| Piece | Where | Role |
|---|---|---|
| `EntityDefinitions` | `aatm.data_models.extractors` | Which entities to extract, loaded from YAML |
| `SimpleLLMExtractor` | `aatm.extractors` | Sends each note to the LLM and validates the output |
| `validate_extracted_entity` | `aatm.data_models.extractors` | Grounds and checks the model output (also usable on its own) |
| `EntityValidationResult` | `aatm.data_models.extractors` | Validated entities, rejected candidates and the records view |

The extracted text (for example `"Hypertension"` or `"Furosemide 40 mg"`) can then be
mapped to standard concepts with the terminology mapping pipeline.

---

## 1. Requirements

- A model that supports structured outputs, from one of the providers below. Model ids
  are prefixed with the provider: `<provider>/<model name>`.
- The provider's API key in its environment variable (see [Environment variables](env.md)),
  or passed with `api_key=...`.

| Provider | `model_id` example | Key | Notes |
|---|---|---|---|
| OpenAI | `openai/gpt-5` | `OPENAI_API_KEY` | |
| Google Gemini | `gemini/gemini-3-flash-preview` | `GOOGLE_API_KEY` | |
| Anthropic | `anthropic/claude-opus-5-5` | `ANTHROPIC_API_KEY` | Install with `pip install "aatm[anthropic]"` |
| Ollama | `ollama/llama3.3` | none | Uses `http://localhost:11434/v1` unless `base_url` is given |
| OpenRouter | `openrouter/qwen/qwen3-235b-a22b` | `OPENROUTER_API_KEY` | |
| Any OpenAI-compatible server (vLLM, LM Studio, Together, ...) | `openai-compatible/<model>` | `api_key=...` if needed | Requires `base_url=...` |

```python
extractor = SimpleLLMExtractor(
    model_id="openai-compatible/Qwen/Qwen3-32B",
    base_url="http://localhost:8000/v1",  # e.g. a local vLLM server
    entity_definitions=entity_definitions,
)
```

Other providers can be added with `aatm.llm.register_provider`, or by passing any object
with an async `generate(system=..., user=..., output_type=...)` method as `llm=...`.

---

## 2. A first extraction

```python
from aatm.data_models.extractors import EntityDefinitions
from aatm.extractors import SimpleLLMExtractor

entity_definitions = EntityDefinitions.from_predefined("condition_occurrence")

extractor = SimpleLLMExtractor(
    model_id="openai/gpt-5",  # or "gemini/...", "anthropic/...", "ollama/...", ...
    entity_definitions=entity_definitions,
)

note = """## Problem list
* Hypertension, diagnosed 2015
* Type 2 diabetes mellitus and CKD stage 3, both diagnosed 2019

## Assessment
Hypertension worsened since March 2026."""

results = await extractor([note])  # one result per note, in the same order
result = results[0]

for entity in result.validated_entities:
    print(entity.id, entity.key, repr(entity.text), entity.line, entity.start, entity.end, entity.parents)
```

The extractor is **asynchronous**. In a notebook you can `await` it directly; in a
script, wrap the call:

```python
import asyncio

results = asyncio.run(extractor([note]))
```

Several notes can be passed at once. Use `max_concurrency` to control how many model
calls run in parallel (default: 1):

```python
extractor = SimpleLLMExtractor(
    model_id="openai/gpt-5",
    entity_definitions=entity_definitions,
    max_concurrency=4,
)
results = await extractor(list_of_notes)
```

---

## 3. Entity definitions

Entity definitions tell the model **what** to extract. Each definition has a name, a
key, a definition text (this is what the model reads, so write it carefully) and,
optionally, a parent.

### Predefined definitions

AATM ships definitions aligned with OMOP CDM tables:

| Name | Keys |
|---|---|
| `condition_occurrence` | `condition_name`; children of a condition: `verbatim_start_date`, `verbatim_end_date` |
| `drug_exposure` | `drug_name`; children of a drug: `verbatim_start_date`, `verbatim_end_date`, `route_source_value`, `dose_unit_source_value`, `sig` |

```python
EntityDefinitions.from_predefined("drug_exposure")
```

### Writing your own

Definitions are a YAML list:

```yaml title="my_definitions.yaml"
- name: Allergen
  definition: A substance the patient is allergic to, e.g. "penicillin" or "peanuts".

- name: Reaction
  definition: The reaction caused by an allergen, e.g. "rash" or "anaphylaxis".
  has_parent: true
  parent_key: allergen
```

```python
entity_definitions = EntityDefinitions.from_yaml("my_definitions.yaml")
```

Or build them in Python:

```python
from aatm.data_models.extractors import EntityDefinition, EntityDefinitions

entity_definitions = EntityDefinitions(
    entities=[
        EntityDefinition(name="Allergen", definition="A substance the patient is allergic to."),
        EntityDefinition(
            name="Reaction",
            definition="The reaction caused by an allergen.",
            has_parent=True,
            parent_key="allergen",
        ),
    ]
)
```

Rules:

- **`key`** is optional. If omitted, it is generated from the name: lowercase, accents
  removed, other characters replaced by `_` (`"Condição — Início"` becomes
  `condicao_inicio`).
- **Parents**: an entity with `has_parent: true` must set `parent_key` to the key of
  another definition, and every extracted mention of it must be linked to a mention of
  that parent. Hierarchies can have any depth (drug → dose → unit).
- **Invalid files fail when loaded**, before any model call, with a message that names
  the file: duplicate keys, a `parent_key` that does not exist, an entity that is its
  own parent, `has_parent` without `parent_key` (or the reverse), and loops such as
  `a -> b -> a`. For example:

```text
ValueError: Invalid entity definitions in my_definitions.yaml:
1 validation error for EntityDefinitions
  Value error, Cycle in entity parent keys: drug_name -> unit -> dose -> drug_name
```

---

## 4. What the model sees and returns

The note is split into **numbered lines** before it is sent to the model. Blank lines
are skipped and every line keeps its position in the original note:

```text
[L1] ## Problem list
[L2] * Hypertension, diagnosed 2015
[L3] * Type 2 diabetes mellitus and CKD stage 3, both diagnosed 2019
[L4] ## Assessment
[L5] Hypertension worsened since March 2026.
```

For every mention, the model returns:

| Field | Meaning |
|---|---|
| `id` | A local id, unique in the response: `e1`, `e2`, ... |
| `text` | The exact text, copied from the line |
| `key` | The entity key from the definitions |
| `line` | The line the mention is on, e.g. `L2` |
| `occurrence` | Only when the same text appears more than once on that line: which one (1-based) |
| `parents` | The ids of the parent mentions, e.g. `["e1"]`; several parents are allowed |

Because each mention has a line and each link points to an id, the result is
unambiguous: the `2015` above belongs to the `Hypertension` on line 2, not the one on
line 5.

Lines are kept whole by default. For notes with very long paragraphs (or text that lost
its line breaks), you can split long lines at sentence boundaries:

```python
extractor = SimpleLLMExtractor(..., max_segment_chars=400)
```

The sentence splitter is simple and also splits after abbreviations such as `p.o.`, so
only enable it when your notes need it.

---

## 5. Validation

The model output is never trusted as is. Each candidate is checked, and the ones that
fail are kept in `result.unmatched_entities` with the reasons:

| Reason | Meaning |
|---|---|
| `invalid_line` | The line id does not exist |
| `invalid_text` | The text does not appear anywhere in the note |
| `text_not_in_line` | The text is in the note, but not on the given line |
| `invalid_occurrence` | The line has fewer copies of the text than `occurrence` |
| `invalid_key` | The key is not in the entity definitions |
| `required_parent_missing` | The entity needs a parent but lists none |
| `invalid_parent` | A parent id does not exist or has the wrong key |
| `parent_rejected` | A parent was itself rejected |
| `unexpected_parent` | An entity without a parent definition lists parents |
| `duplicate_id` | An id was reused; the first entity keeps it |
| `duplicate` | The same mention (same span and key) was extracted twice |

Matching is exact: case, spacing and punctuation must be identical to the note. Small
formatting slips are tolerated: `line` may be written `L4`, `[L4]` or `4`, and a copied
`[L4]` prefix is removed from `text`.

When the same mention is extracted twice, the first one is kept, the parents of both
are combined, and any child pointing at the dropped copy is relinked to the kept one.

For the note above, a response with three deliberate mistakes gives:

```text
Validated:
  e1  condition_name       'Hypertension'              L2  [18:30]   parents=[]
  e2  verbatim_start_date  '2015'                      L2  [42:46]   parents=['e1']
  e3  condition_name       'Type 2 diabetes mellitus'  L3  [49:73]   parents=[]
  e4  condition_name       'CKD stage 3'               L3  [78:89]   parents=[]
  e5  verbatim_start_date  '2019'                      L3  [106:110] parents=['e3', 'e4']
  e6  condition_name       'Hypertension'              L5  [126:138] parents=[]
Rejected:
  e7  'March 2026'    L4  ['text_not_in_line']   # the text is on L5
  e8  '2026'          L5  ['invalid_parent']     # its parent id does not exist
  e9  'Hypertension'  L2  ['duplicate']          # same mention as e1
```

`result.unmatched_keys` lists the definition keys with no validated entity (here
`['verbatim_end_date']`), which helps spot what the model never found.

---

## 6. Working with the results

`extractor(...)` returns one `EntityValidationResult` per note:

| Attribute / method | Content |
|---|---|
| `validated_entities` | Validated entities in document order, each with `id`, `key`, `text`, `line`, `start`, `end` and `parents` (ids) |
| `unmatched_entities` | Rejected candidates with their `reasons` |
| `unmatched_keys` | Definition keys with no validated entity |
| `get_entity(id)` | One validated entity by id |
| `to_records()` | Entities nested under their parents |

### Follow links between entities

```python
date = result.get_entity("e5")
[result.get_entity(p).text for p in date.parents]
# ['Type 2 diabetes mellitus', 'CKD stage 3']
```

`start` and `end` are character offsets in the original note, so
`note[entity.start:entity.end] == entity.text` always holds.

### Records view

`to_records()` returns one record per entity without parents, with its children grouped
by key. A child with several parents appears under each of them:

```python
for record in result.to_records():
    print(record.text, record.line, {k: [c.text for c in v] for k, v in record.children.items()})
```

```text
Hypertension L2 {'verbatim_start_date': ['2015']}
Type 2 diabetes mellitus L3 {'verbatim_start_date': ['2019']}
CKD stage 3 L3 {'verbatim_start_date': ['2019']}
Hypertension L5 {}
```

### Building table rows

Records map naturally to OMOP-like rows:

```python
rows = []
for record in result.to_records():
    starts = record.children.get("verbatim_start_date", [])
    ends = record.children.get("verbatim_end_date", [])
    rows.append(
        {
            "condition_source_value": record.text,
            "verbatim_start_date": starts[0].text if starts else None,
            "verbatim_end_date": ends[0].text if ends else None,
            "evidence": f"{record.line} [{record.start}:{record.end}]",
        }
    )
```

Dates are returned exactly as written in the note (`"2015"`, `"March 2026"`) and still
need to be normalized, and `condition_source_value` still needs to be mapped to a
standard concept.

---

## 7. Validating without calling a model

`validate_extracted_entity` works on any model output that follows the
`ExtractedEntityCandidates` schema. This is useful in tests, or to validate the output of
a model called outside AATM:

```python
from aatm.data_models.extractors import (
    ExtractedEntityCandidates,
    validate_extracted_entity,
)

candidates = ExtractedEntityCandidates.model_validate(
    {
        "entities": [
            {"id": "e1", "text": "Hypertension", "key": "condition_name", "line": "L2", "occurrence": None, "parents": []},
            {"id": "e2", "text": "2015", "key": "verbatim_start_date", "line": "L2", "occurrence": None, "parents": ["e1"]},
        ]
    }
)
result = validate_extracted_entity(note, entity_definitions, candidates)
```

The line ids are computed with `segment_text(note)`. If you used `max_segment_chars`,
pass the same segments: `validate_extracted_entity(note, defs, candidates,
segments=segment_text(note, max_segment_chars=400))`.

---

## 8. Customizing the extractor

| Argument | Default | Purpose |
|---|---|---|
| `model_id` | required unless `llm` is given | `<provider>/<model>`, see [Requirements](#1-requirements) |
| `entity_definitions` | required | What to extract |
| `api_key` | `None` | Provider key; the provider's environment variable is used when omitted |
| `base_url` | `None` | Server URL for OpenAI-compatible providers |
| `settings` | `None` | `aatm.llm.LLMSettings` or a dict, e.g. `{"temperature": 0, "max_tokens": 8000}` |
| `llm` | `None` | A ready-made `aatm.llm.StructuredLLM`, used instead of `model_id` |
| `max_concurrency` | `1` | Parallel model calls |
| `max_segment_chars` | `None` | Split lines longer than this at sentence boundaries |
| `instructions` | built-in prompt | Custom system prompt; `{output_format}` and `{entity_definitions}` are filled in |

`settings` accepts `temperature`, `max_tokens`, `reasoning_effort` (`"minimal"` to
`"max"`), `timeout` (seconds) and `provider_kwargs` (extra arguments passed as-is to the
provider's API call). Settings left unset use the provider's defaults. Current Anthropic
models do not accept `temperature`, so it is ignored there with a warning.

When writing custom instructions, keep the explanation of line ids, ids and parents from
the built-in prompt (`aatm.extractors.llm_extractors.DEFAULT_INSTRUCTION`); validation
expects them.

---

## 9. Debugging

AATM is silent by default. To see what validation does:

```python
import logging
import aatm

aatm.configure_logging()  # shows AATM messages on stderr
logging.getLogger("aatm.data_models.extractors").setLevel(logging.DEBUG)
```

The `unmatched_entities` reasons are usually the fastest way to understand a poor
extraction: many `text_not_in_line` rejections point to line-id mistakes, and many
`invalid_text` rejections to paraphrased spans.

---

## 10. Current limitations

- Matching is exact; spans with different case or spacing than the note are rejected.
- Negation, uncertainty and family history are not captured as attributes. Mentions
  such as "denies chest pain" are only excluded if the entity definitions say so.
- `GeminiExtractor` (also in `aatm.extractors`) is an older interface based on
  `ExtractionTask` and does not use entity definitions or this validation.

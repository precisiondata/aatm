# Writing Entity Definitions

Entity definitions tell the extractor **what to find** in a clinical note. They live in
a YAML file that you write once and reuse for every note. This page explains every
field, how to choose between them, and how to load and check a file before spending
any model calls.

If you have not run an extraction yet, start with the
[Information extraction quick start](quick_start_extraction.md).

---

## A complete example

Suppose you want every BRCA test mentioned in oncology notes, when it was done, and
what the result was:

```yaml title="brca.yaml"
# BRCA testing in oncology notes.

- name: BRCA gene test
  key: brca_test
  definition: >
    A mention of BRCA1 and/or BRCA2 genetic testing performed for the patient,
    e.g. "BRCA1/2 panel" or "germline BRCA testing". Does not include tests of
    relatives or tests that are only planned.
  attributes:
    - key: result
      description: The result of the test as stated in the note.
      labels:
        - value: positive
          description: A pathogenic or likely pathogenic variant was reported.
        - value: negative
          description: No pathogenic or likely pathogenic variant was found.
        - value: uncertain
          description: Only a variant of uncertain significance (VUS) was reported.
    - key: origin
      description: Whether germline or tumour (somatic) DNA was tested.
      labels:
        - value: germline
          description: Germline testing, e.g. blood or saliva.
        - value: somatic
          description: Testing of tumour tissue.

- name: Test date
  definition: The date a BRCA test was performed or reported, e.g. "2024" or "March 2025".
  has_parent: true
  parent_key: brca_test
```

This file uses the three building blocks:

| Building block | In the example | Answers |
|---|---|---|
| **Entity** | `brca_test` | *Where* in the note is this mentioned? (an exact span) |
| **Parent link** | `test_date` → `brca_test` | *Which* mention does this detail belong to? (another span) |
| **Attribute** | `result`, `origin` | *What* does the note say about this mention? (a label from a fixed list) |

Use it like any other definition file:

```python
from aatm.data_models.extractors import EntityDefinitions
from aatm.extractors import SimpleLLMExtractor

entity_definitions = EntityDefinitions.from_yaml("brca.yaml")

extractor = SimpleLLMExtractor(model_id="openai/gpt-5", entity_definitions=entity_definitions)

note = "Germline BRCA1/2 panel 2024: no pathogenic variants detected."
(result,) = await extractor([note])

for record in result.to_records():
    dates = [d.text for d in record.children.get("test_date", [])]
    print(record.text, record.attributes, dates)
```

A typical answer:

```text
Germline BRCA1/2 panel {'result': 'negative', 'origin': 'germline'} ['2024']
```

---

## Field reference

A definition file is a YAML **list**. Each item is one entity.

### Entity fields

| Field | Required | Default | Meaning |
|---|---|---|---|
| `name` | yes | | Human-readable name, shown to the model |
| `key` | no | generated from `name` | Identifier used in the model output and in your results. Lowercase letters, digits and `_` only |
| `definition` | yes | | What counts as a mention. **The model reads this**, so it decides quality |
| `has_parent` | no | `false` | Every mention must be linked to a mention of `parent_key` |
| `parent_key` | when `has_parent` is true | | Key of the parent entity |
| `attributes` | no | none | Classifications the model assigns to each mention (see below) |
| `omop` | no | none | How each mention becomes a row of an OMOP CDM table. Never sent to the model. See [From notes to OMOP tables](omop_tables.md) |

### Attribute fields

| Field | Required | Default | Meaning |
|---|---|---|---|
| `key` | yes | | Identifier, unique within the entity. Lowercase letters, digits and `_` only |
| `description` | yes | | What the attribute describes. The model reads it |
| `required` | no | `false` | Reject mentions that have no valid label for this attribute |
| `ordered` | no | `false` | The labels form a scale, listed lowest first (see [Ordered labels](#ordered-labels)) |
| `labels` | yes | | The allowed values, at least one |

### Label fields

| Field | Required | Meaning |
|---|---|---|
| `value` | yes | The label the model returns and you get back, e.g. `negative`. Unique within the attribute, ignoring case |
| `description` | yes | When this label applies. The model reads it |

---

## Keys

Keys are how entities refer to each other (`parent_key`) and how you find them in the
results (`entity.key`, `record.children["test_date"]`).

- **Entity keys** are optional. When omitted, the key is generated from the name:
  lowercase, accents removed, any other character replaced by `_`. `"Test date"`
  becomes `test_date`, `"Condição — Início"` becomes `condicao_inicio`.
- When you set a key yourself, it must use **lowercase letters, digits and single
  underscores between them** (`brca_test`), the same form the generated keys have,
  because the model must return keys in this form. Anything else (`BRCA-Test`,
  `brca__test`, `_brca`) fails when the file is loaded, with a suggested key:

    ```text
    Value error, Entity key 'BRCA-Test' must use only lowercase letters, digits and single underscores between them. Use 'brca_test' instead.
    ```

- **Attribute keys** are always written by you and follow the same rule.
- Keys must be unique: entity keys within the file, attribute keys within their entity.

Set the key explicitly when other entities refer to it as `parent_key`. Renaming the
entity then cannot break the links.

---

## Parents: details that belong to a mention

Use a parent link when a piece of information **only makes sense attached to another
mention**: a start date of a condition, the dose of a drug, the unit of a dose.

```yaml
- name: Drug name
  key: drug_name
  definition: The name of a drug used by the patient, e.g. "Aspirin".

- name: Dose
  key: dose
  definition: The amount of one administration of a drug, e.g. "40".
  has_parent: true
  parent_key: drug_name

- name: Dose unit
  key: dose_unit
  definition: The unit of a dose, e.g. "mg".
  has_parent: true
  parent_key: dose
```

- The child is a **span** too: it must appear on a line of the note, and you get its
  exact position.
- Every mention of a child must name at least one parent mention. A child without a
  parent is rejected (`required_parent_missing`), and so is a child whose parent was
  rejected (`parent_rejected`).
- One child mention can have **several parents**, e.g. one date that applies to two
  conditions.
- Hierarchies can be **any depth** (drug → dose → unit), but not circular.

---

## Attributes: what the note says about a mention

Use an attribute when you need a **judgement from a fixed list**, not a piece of text:
a test result, whether a condition is present or denied, whether it is the patient's
or a relative's, whether a drug is current or past.

The label does not have to appear in the note. The model reads "no pathogenic variants
detected" and answers `negative`.

### Attribute or child entity?

| You want... | Use | Example |
|---|---|---|
| The text itself, with its position | A child entity | The date `"2024"`, the dose `"40 mg"` |
| One of a fixed set of values | An attribute | `result: negative` |
| Both: a fixed value **and** the text that justifies it | An attribute on a child entity | A `brca_result` child with span `"no pathogenic variants detected"` and attribute `result: negative` |

The last pattern is useful when someone will audit the data: the label points at the
exact words it was based on.

### Optional or required?

**Optional (the default)** is the right choice almost always. When the note does not
state the attribute, the model leaves it out and the mention is kept without it. The
model is told never to guess, so "the note does not say" and a real answer stay
distinct. For BRCA, this keeps a VUS (`uncertain`, a real result) apart from a note
that never gives the result (no `result` at all).

You do not need a `not_stated` or `unknown` label: a missing attribute already means
that.

**`required: true`** means a mention is useless to you without the attribute. Mentions
without a valid label are **rejected, together with their children**. Use it only when
you would discard those mentions anyway, for example when you only want tests with a
known result.

### Writing labels

- **Describe every label**, including the borderline cases. The descriptions are where
  you decide whether a VUS counts as `uncertain` or `negative`, or whether "possible
  pneumonia" is `present` or `possible`.
- Make labels **mutually exclusive**: the model picks one per attribute.
- Cover the cases you expect. A value that is not in the list is dropped and reported
  as `invalid_label`, so many such issues on one attribute suggest a missing label or
  an unclear description.
- Keep values short and stable (`positive`, not `Positive result reported`). You will
  filter and count by them.

### Ordered labels

Mark an attribute `ordered: true` when its labels are a scale, listed from lowest to
highest, like the ECOG grades 0 to 5. The extractor does not use it. A
[decision model labeller](../reference/decision_models.md#labelling-attributes-decisionattributelabeller),
if you add one, then asks a score question, which knows that grade 2 lies between 1
and 3, instead of a choice between unrelated options. Leave it out for labels with no
order, such as `never` / `former` / `current`.

### How the model's labels are checked

Labels are checked after the model answers, against the file:

- **Case and surrounding spaces are ignored.** If the model answers `" Negative "`, it
  is accepted as `negative` and stored exactly as you defined it, without any issue.
- **A wrong label on an optional attribute never drops the mention.** The attribute is
  removed and the problem is reported, with the attribute key, so you can see which
  attribute causes trouble.
- **Attributes are checked only for mentions that pass every other check.** A mention
  rejected for its text, line or parent adds no attribute issues, so the issue counts
  cover only mentions that were really found.
- **Duplicates:** when the same mention is extracted twice, the first one's labels are
  kept. A label the first mention lacks is copied from the duplicate, a different
  label is reported as `conflicting_duplicate`, and a duplicate that is missing a
  required attribute is not reported, since the kept mention already has it.

See [Attribute issues](quick_start_extraction.md#attribute-issues) for the full list of
issue reasons and how to count them.

---

## Writing good definitions

The `definition` and `description` fields are the prompt. Most extraction problems are
fixed here.

- **Say what counts, with examples.** `e.g. "BRCA1/2 panel" or "germline BRCA testing"`.
- **Say what does not count.** Tests of relatives, planned tests, negated findings,
  family history. The model includes them unless told otherwise.
- **Say how much text to take.** "The name and dose form of a drug, e.g. "Aspirin 81 mg
  oral tablet"" gives longer spans than "The name of a drug".
- **One idea per entity.** Split "condition and its severity" into an entity and an
  attribute (or a child entity).
- **Iterate on real notes.** Look at `unmatched_entities`, `unmatched_keys` and the
  attribute issues after a few notes, and adjust the wording.

---

## YAML pitfalls

YAML converts some unquoted words, which then fail validation when the file is loaded:

| Written | YAML reads it as | Write instead |
|---|---|---|
| `value: yes` / `no` / `on` / `off` | `true` / `false` | `value: "yes"` |
| `value: true` / `false` | a boolean | `value: "true"` |
| `value: null` / `~` | nothing | `value: "null"` |
| `value: 1` / `2.5` | a number | `value: "1"` |
| `definition: Result: negative` | invalid YAML (second `: `) | quote the text, or use `>` |

For long text, use a folded block (`>`): lines are joined with spaces, and the trailing
newline is removed when loaded.

```yaml
  definition: >
    A mention of BRCA1 and/or BRCA2 genetic testing performed for the patient.
    Does not include tests of relatives.
```

Comments start with `#` and are ignored, which makes them a good place to explain
choices to the next developer.

---

## Checking a file before using it

### Load errors

Invalid files fail when loaded, **before any model call**, with a message that names
the file:

- an entity or attribute key that is not lowercase `snake_case`
- duplicate entity keys, or duplicate attribute keys in one entity
- `has_parent: true` without `parent_key`, or the reverse
- a `parent_key` that does not exist, an entity that is its own parent, or a loop such
  as `a -> b -> a`
- an attribute without labels, or two labels that differ only in case
- a missing `name`, `definition`, `description` or label `value`, or a value of the
  wrong type (see [YAML pitfalls](#yaml-pitfalls))

```text
ValueError: Invalid entity definitions in brca.yaml:
1 validation error for EntityDefinitions
entities.0.attributes.0.labels.0.value
  Input should be a valid string [type=string_type, input_value=True, input_type=bool]
```

The location (`entities.0.attributes.0.labels.0.value`) counts from 0: the first
entity, its first attribute, its first label.

### See what the model will read

```python
print(entity_definitions.to_prompt_str())
```

```text
BRCA gene test (key: `brca_test`, has_parent: False, parent_key: `None`): A mention of BRCA1 and/or BRCA2 genetic testing performed for the patient, e.g. "BRCA1/2 panel" or "germline BRCA testing". Does not include tests of relatives or tests that are only planned.
  Attributes:
  - `result` (optional): The result of the test as stated in the note.
    - `positive`: A pathogenic or likely pathogenic variant was reported.
    - `negative`: No pathogenic or likely pathogenic variant was found.
    - `uncertain`: Only a variant of uncertain significance (VUS) was reported.
  - `origin` (optional): Whether germline or tumour (somatic) DNA was tested.
    - `germline`: Germline testing, e.g. blood or saliva.
    - `somatic`: Testing of tumour tissue.
Test date (key: `test_date`, has_parent: True, parent_key: `brca_test`): The date a BRCA test was performed or reported, e.g. "2024" or "March 2025".
```

If this text would confuse a colleague, it will confuse the model too.

### Test it without a model

`validate_extracted_entity` checks a hand-written answer against your definitions, so
you can try edge cases in a unit test:

```python
from aatm.data_models.extractors import ExtractedEntityCandidates, validate_extracted_entity

note = "BRCA1/2 panel 2024: no pathogenic variants detected."
candidates = ExtractedEntityCandidates.model_validate(
    {
        "entities": [
            {
                "id": "e1", "text": "BRCA1/2 panel", "key": "brca_test", "line": "L1",
                "attributes": [{"key": "result", "value": "negative"}],
            },
            {"id": "e2", "text": "2024", "key": "test_date", "line": "L1", "parents": ["e1"]},
        ]
    }
)
result = validate_extracted_entity(note, entity_definitions, candidates)
assert result.validated_entities[0].get_attribute("result") == "negative"
```

---

## Other ways to get definitions

**Predefined files** ship with AATM, aligned with OMOP CDM tables and common clinical
scales. Their source in
`aatm/omop/entity_definitions/` is a good starting point to copy:

```python
EntityDefinitions.from_predefined("condition_occurrence")
EntityDefinitions.from_predefined("drug_exposure")
EntityDefinitions.from_predefined("ecog_performance_status")
EntityDefinitions.from_predefined("lab_results")
EntityDefinitions.from_predefined("smoking_status")
EntityDefinitions.from_predefined("family_history")
EntityDefinitions.from_predefined("medical_history")
EntityDefinitions.from_predefined("device_exposure")
```

**In Python**, the same fields are available as models, which is handy for tests or
definitions built at run time:

```python
from aatm.data_models.extractors import (
    AttributeDefinition,
    AttributeLabel,
    EntityDefinition,
    EntityDefinitions,
)

entity_definitions = EntityDefinitions(
    entities=[
        EntityDefinition(
            name="BRCA gene test",
            key="brca_test",
            definition="A mention of BRCA1 and/or BRCA2 genetic testing.",
            attributes=[
                AttributeDefinition(
                    key="result",
                    description="The result of the test as stated in the note.",
                    labels=[
                        AttributeLabel(value="positive", description="A pathogenic variant was reported."),
                        AttributeLabel(value="negative", description="No pathogenic variant was found."),
                        AttributeLabel(value="uncertain", description="Only a VUS was reported."),
                    ],
                )
            ],
        ),
        EntityDefinition(
            name="Test date",
            definition="The date a BRCA test was performed.",
            has_parent=True,
            parent_key="brca_test",
        ),
    ]
)
```

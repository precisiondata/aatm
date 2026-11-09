# From notes to OMOP tables

An entity definition can say how each extracted mention becomes a row of an
[OMOP CDM](https://ohdsi.github.io/CommonDataModel/cdm55.html) table. AATM then runs the
whole chain in one call: extraction with an LLM, coding with the OMOP vocabularies, and
table rows that are filled as far as the notes allow.

Two tables are supported: `measurement` and `observation`. This page uses the
predefined ECOG performance status definition, which builds `measurement` rows, then
shows [lab results](#coding-text-with-the-terminology-mapper) coded with the
terminology mapper and [smoking and family history](#the-observation-table) in the
`observation` table.

!!! note "Prerequisites"
    Concept codes are looked up in the local vocabulary database built by
    [`aatm init`](quick_start_cli.md) (`.aatm/omop.db`), so run it once first. Extraction
    needs an API key for the model provider, e.g. `OPENAI_API_KEY`.

---

## Quick start

```python
from aatm.omop import ClinicalNote, OmopExtractionPipeline

pipeline = OmopExtractionPipeline("ecog_performance_status", model_id="openai/gpt-5-mini")

notes = [
    ClinicalNote(
        text="Assessment 05/12/2025\nOncology follow-up. Bedbound, ECOG 4.",
        note_id=107,
        person_id=3,
        note_date="2025-06-01",
        visit_occurrence_id=507,
    ),
]
tables = await pipeline(notes)

tables["measurement"]
```

| measurement_id | person_id | measurement_concept_id | measurement_date | measurement_type_concept_id | value_as_number | value_as_concept_id | visit_occurrence_id | measurement_source_value | measurement_source_concept_id | value_source_value |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 36305384 | 2025-05-12 | 32858 | 4.0 | 4174251 | 507 | ECOG 4 | 4308014 | 4 |

The other columns of the table are present and empty (`NULL`).

`ClinicalNote` carries the note text and the values that come **with** the note rather
than from its text: `person_id`, `visit_occurrence_id`, `visit_detail_id`,
`provider_id`, `note_id` and `note_date`. Plain strings are accepted too, but then
those columns stay empty.

To run the steps separately, for example to reuse extraction results:

```python
from aatm.data_models.extractors import EntityDefinitions
from aatm.extractors import SimpleLLMExtractor
from aatm.omop import OmopTableBuilder

definitions = EntityDefinitions.from_predefined("ecog_performance_status")
extractor = SimpleLLMExtractor(model_id="openai/gpt-5-mini", entity_definitions=definitions)
builder = OmopTableBuilder(definitions)

results = await extractor([note.text for note in notes])
tables = builder.build(notes, results)
```

---

## What you get back

`OmopTables` holds the rows of every mapped table, where each row came from, and what
could not be filled.

```python
tables["measurement"]                               # DataFrame, CDM columns in CDM order
tables.to_dataframes(provenance=True)["measurement"]  # plus _note_id, _entity_id, _text, _start, _end
tables.issues_dataframe()                           # one row per problem
tables.mappings_dataframe()                         # how each text was coded (map, lookup)
tables.extraction_results                           # the extractor's raw output
```

With `provenance=True`, every row points at the exact span of the note it was built
from, so it can be audited.

### Issues

| Reason | Meaning |
|---|---|
| `missing_required` | A column the CDM requires is empty, usually because the note's context did not give it, e.g. `person_id` |
| `unparsable_date` | A date in the note is not a full date, e.g. "on admission". The next source is used if there is one, and the detail says which |
| `imputed_date_parts` | A date has no day or month, e.g. "March 2024", and was read as the first day (2024-03-01) |
| `unmapped_label` | An attribute label has no concept in `concepts`. `0` is stored, as the CDM requires for unmapped values |
| `invalid_value` | A value does not fit its column, e.g. a text that is not a number |
| `truncated` | A text was longer than its column (50 characters for `*_source_value`) and was cut |
| `multiple_children` | A mention had several children of the same key, e.g. two dates. The first was used |
| `unmapped` | A `map` or `lookup` found no concept for the text. `0` is stored |
| `low_confidence` | A `map` result scored below `min_score`. `0` is stored; the candidate is in the mappings |
| `domain_mismatch` | A coded concept is not in the domain the column requires |

---

## Writing a mapping

The mapping is an `omop` block on a top-level entity. This is the one in
`ecog_performance_status.yaml`:

```yaml
- name: ECOG performance status
  key: ecog_performance_status
  definition: >
    An explicit statement of the patient's own ECOG performance status. ...
  attributes:
    - key: score
      ...
  omop:
    table: measurement
    fields:
      person_id: {from: context.person_id}
      measurement_concept_id: {concept: "SNOMED:423740007", maps_to: true}
      measurement_source_concept_id: {concept: "SNOMED:423740007"}
      measurement_source_value: {from: text}
      measurement_date:
        - {from: child.assessment_date, as: date}
        - {from: context.note_date}
      measurement_type_concept_id: {value: 32858}  # NLP
      value_as_number: {from: attribute.score, as: float}
      value_as_concept_id:
        from: attribute.score
        concepts:
          "0": "SNOMED:425389002"  # ECOG performance status - grade 0
          "1": "SNOMED:422512005"  # ECOG performance status - grade 1
          # ...
      value_source_value: {from: attribute.score}
      visit_occurrence_id: {from: context.visit_occurrence_id}
```

!!! tip "The model never sees the mapping"
    `to_prompt_str()` renders only the name, key, definition, parent and attributes.
    Concept codes, table names and label-to-concept maps stay out of the prompt, so
    they cost no tokens and cannot confuse the model.

### Column sources

Each column lists where its value comes from:

| Source | Value |
|---|---|
| `{from: text}` | The text of the mention, e.g. `"ECOG 4"` |
| `{from: attribute.<key>}` | The label the model chose for an attribute, e.g. `"4"` |
| `{from: child.<key>}` | The text of the mention's child with that key, e.g. the date `"05/12/2025"` |
| `{from: context.<name>}` | A value of the `ClinicalNote`, e.g. `person_id` |
| `{value: <constant>}` | A fixed value, e.g. a type concept id |
| `{concept: "<VOCABULARY>:<code>"}` | The concept id of a code, e.g. `"SNOMED:423740007"` |

Options:

| Option | Effect |
|---|---|
| `as: int`, `float`, `str` or `date` | Convert the value. Dates are read from text such as `"05/12/2025"` or `"March 2024"` |
| `concepts: {<label>: "<VOCABULARY>:<code>"}` | With `from: attribute.<key>`: store the concept of the chosen label instead of the label |
| `maps_to: true` | Store the standard concept the code maps to (`Maps to` relationship) |

A **list** of sources is tried in order, and the first that gives a value wins. ECOG
uses this for the date: the assessment date if the note states a full one, else the
note date.

The primary key (`measurement_id`) is generated, numbered from `id_start` (default 1).

### Codes, standard concepts and domains

Write codes as `VOCABULARY:code`, the way they appear in the vocabulary, not as
`concept_id`s, so the YAML stays readable. They are looked up when the builder is
created, so mistakes fail before any model call:

- a code that is not in the vocabulary;
- a non-standard concept in a `*_concept_id` column, e.g. SNOMED 423740007 in
  `measurement_concept_id`. Add `maps_to: true` to use its standard concept;
- a concept of the wrong domain, e.g. a Condition concept in `measurement_concept_id`;
- a `maps_to: true` code without exactly one standard mapping.

`*_source_concept_id` columns keep the code as written, standard or not, as the CDM
asks.

Check what a mapping resolves to before a run:

```python
builder.describe_concepts()
```

| entity | column | label | code | concept_id | concept_name | vocabulary_id | domain_id |
|---|---|---|---|---|---|---|---|
| ecog_performance_status | measurement_concept_id | | SNOMED:423740007 | 36305384 | ECOG Performance Status score | LOINC | Measurement |
| ecog_performance_status | measurement_source_concept_id | | SNOMED:423740007 | 4308014 | ECOG performance status | SNOMED | Observation |
| ecog_performance_status | value_as_concept_id | 0 | SNOMED:425389002 | 4175026 | ECOG performance status - grade 0 | SNOMED | Observation |
| ... | | | | | | | |

The first row shows why `maps_to` matters: the SNOMED concept for ECOG performance
status is not standard, and the vocabulary maps it to LOINC 89247-1, so that is the
concept the CDM expects in `measurement_concept_id`.

### Load-time checks

The rest of the block is checked when the YAML is loaded: the table and column names
must exist, `attribute.<key>` must be one of the entity's attributes, the labels in
`concepts` must be its labels, `child.<key>` must be one of its children and
`context.<name>` one of the `ClinicalNote` fields. Only top-level entities can have a
mapping; map a child's value from its parent with `from: child.<key>`.

---

## Coding text with the terminology mapper

Some concepts are fixed (ECOG is always the same measurement), but most must be coded
from what the note says: the name of a lab test, a condition, a drug. A `map` source
sends the extracted text through the same `TerminologyMapper` used for
[terminology mapping](quick_start_python_api.md): translation, retrieval, reranking
and selection. The predefined `lab_results` definition does this for the test name:

```yaml
- name: Laboratory test
  key: lab_test
  definition: ...
  omop:
    table: measurement
    fields:
      measurement_concept_id:
        from: text
        map: {vocabularies: [LOINC]}       # domain defaults to Measurement
      unit_concept_id: {from: child.lab_unit, lookup: UCUM}
      value_as_number: {from: child.lab_value, as: float}
      # ...
```

The YAML says **what** to code and where to look; your code supplies **how**, since
the mapper involves embedding models, LLMs and API keys:

```python
from aatm.omop import ClinicalNote, OmopExtractionPipeline
from aatm.selectors import OpenAILLMSelector
from aatm.terminology_mapper import TerminologyMapper

mapper = TerminologyMapper(
    retriever="text-embedding-3-small",
    selector=OpenAILLMSelector("gpt-5-mini"),
)
mapper.retriever.top_k = 30

pipeline = OmopExtractionPipeline("lab_results", model_id="openai/gpt-5-mini", mapper=mapper)
tables = await pipeline([
    ClinicalNote(text="Labs 03/15/2025: Hb 11.2 g/dL, creatinine 1.4 mg/dl.", person_id=10),
])
tables.mappings_dataframe()
```

| row_id | column | text | method | concept_id | concept_name | concept_code | score | stored |
|---|---|---|---|---|---|---|---|---|
| 1 | measurement_concept_id | Hb | map | 3000963 | Hemoglobin [Mass/volume] in Blood | 718-7 | 0.30 | 3000963 |
| 1 | unit_concept_id | g/dL | lookup | 8713 | gram per deciliter | g/dL | | 8713 |
| 2 | measurement_concept_id | creatinine | map | ... | ... | ... | ... | ... |
| 2 | unit_concept_id | mg/dl | lookup | 8840 | milligram per deciliter | mg/dL | | 8840 |

How it works:

- **Retrieval is restricted** to the domain and vocabularies you give. `domain`
  defaults to the domain the column requires (`Measurement` for
  `measurement_concept_id`), and a conflicting one fails when the builder is created.
  The filter is passed to the retriever with `TerminologyMapper.map(..., where=...)`.
- **Each distinct text is coded once per batch**: a hundred notes mentioning "Hb" make
  one mapper query.
- **The result is already standard**: the index holds standard concepts and
  non-standard expressions linked to them.
- **Review the mappings** with `mappings_dataframe()`, one row per coded text, with the
  concept chosen, its score and what was stored.
- **`min_score`** (`map: {min_score: 0.8}`) stores `0` and reports `low_confidence`
  below that score. The score is `1 - distance` of the selected candidate, so its scale
  depends on the retriever: check it on your data before setting a threshold.

!!! warning "Mapping quality depends on the mapper configuration"
    Test names are short and ambiguous, and LOINC has many variants of each test. With
    the default selector, which takes the nearest candidate, "Hb" became *Hemoglobin
    H/Hemoglobin.total in Blood* and "creatinine" a urine epinephrine ratio. With
    `OpenAILLMSelector("gpt-5-mini")` choosing among 30 candidates, 7 of 9 test names in
    a small sample were coded correctly, including Portuguese ones. Both misses were
    creatinine, whose serum concept was not among the retrieved candidates.

### Code lookups

`lookup: <VOCABULARY>` finds the extracted text as a code of that vocabulary, which fits
units: `"g/dL"` is the UCUM code of *gram per deciliter*. An exact match wins;
otherwise a single standard concept matching while ignoring case is used (`"mg/dl"` ->
`mg/dL`). Texts that are not codes, such as `"x10^3/uL"`, are reported as `unmapped`.

---

## The observation table

The `observation` table holds clinical facts that are not measurements, conditions,
procedures, drugs or devices: social and lifestyle facts, medical and family history.
Like measurements, they are attribute-value pairs: `observation_concept_id` is the
attribute, and the value is a concept, a number, a string or a date.

Two predefined definitions build observation rows, and the pipeline can run them
together, so each note is extracted once:

```python
mapper = TerminologyMapper(retriever="text-embedding-3-small", selector=OpenAILLMSelector("gpt-5-mini"))
pipeline = OmopExtractionPipeline(
    ["smoking_status", "family_history"], model_id="openai/gpt-5-mini", mapper=mapper
)
tables = await pipeline([
    ClinicalNote(
        text="Former smoker, 30 pack-years. Mother had breast cancer at 45.",
        person_id=20,
        note_date="2025-06-01",
    ),
])
tables["observation"]
```

A typical result (concept names added in italics):

| observation_id | observation_concept_id | observation_source_value | value_as_number | value_as_concept_id | value_source_value | qualifier_concept_id | unit_concept_id |
|---|---|---|---|---|---|---|---|
| 1 | 43054909 *Tobacco smoking status* | Former smoker | | 45883458 *Former smoker* | former | | |
| 2 | 3003421 *Cigarettes smoked total (pack per year)* | 30 pack-years | 30.0 | | 30 | | 0 |
| 3 | 4167217 *Family history of clinical finding* | breast cancer | | 4112853 *Malignant tumor of breast* | breast cancer | 4248584 *Mother* | |

**`smoking_status`** has two entities, so one note can give two rows:

- *Smoking status*: LOINC 72166-2 *Tobacco smoking status*, with its standard LOINC
  answer in `value_as_concept_id` (never, former, current every day, current some day,
  current with unknown frequency). "Non-smoker" or "denies smoking" don't say whether
  the patient ever smoked; they get the label `not_current`, which has no LOINC answer,
  so `0` is stored, as the CDM asks for an unmapped value.
- *Pack-years*: LOINC 8664-5, with the number in `value_as_number`. There is no UCUM
  concept for pack-years, so `unit_concept_id` is `0` and `unit_source_value` is
  `"pack-years"`.

**`family_history`** follows the CDM convention: `observation_concept_id` is SNOMED
416471007 *Family history of clinical finding*, and `value_as_concept_id` is the
condition, coded from the mention with the terminology mapper, restricted to SNOMED
concepts of the Condition domain:

```yaml
      value_as_concept_id:
        from: text
        map: {domain: Condition, vocabularies: [SNOMED]}
```

The relative goes to `qualifier_concept_id` as a standard SNOMED relationship concept
(*Mother*, *Sister*...). The CDM has no column for the relative, so this is a
convention of the definition file; `qualifier_source_value` keeps the label.

**Domain rules.** The CDM sends Condition, Procedure, Drug, Measurement, Device and
Specimen concepts to their own tables, so the builder rejects them in
`observation_concept_id`, whether they come from a code, a constant or a `map`
domain. `value_as_concept_id` accepts any domain, since for family history it holds
a condition.

---

## Current limits

- Only the `measurement` and `observation` tables are supported so far.
- `map` sends the text alone: specimen hints from the unit (mg/dL suggests serum) are
  not used.
- Mapper scores are not calibrated across retrievers and selectors.
- Units written outside UCUM (`x10^3/uL`, `mEq/L`) are not normalized before the lookup.
- Relative dates ("on admission", "after cycle 2") are not resolved: they fall back to
  the next source and are reported.

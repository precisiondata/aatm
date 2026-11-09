# From notes to OMOP tables

This page shows how to turn clinical notes into rows of
[OMOP CDM](https://ohdsi.github.io/CommonDataModel/cdm55.html) tables. You describe
what to find in the notes and where each piece goes in the table; AATM extracts it with
an LLM, codes it with the OMOP vocabularies and fills the rows as far as the notes
allow.

Three tables are supported: `measurement` (scores, lab results), `observation`
(smoking, family and medical history, and other facts) and `condition_occurrence` (diagnoses, signs
and symptoms).

If you have not extracted entities from a note before, read the
[Information extraction quick start](quick_start_extraction.md) first: this page builds
on its *entity definitions*.

---

## Before you start

1. **Install AATM** ([Installation](installation.md)).
2. **Download the OMOP vocabularies** from [Athena](https://athena.ohdsi.org/vocabulary/list).
   The predefined definitions on this page use **SNOMED**, **LOINC** and **UCUM**
   (units), so include at least those three.
3. **Run `aatm init`** ([CLI quick start](quick_start_cli.md)). It builds the local
   vocabulary database (`.aatm/omop.db`), which every example on this page reads, and
   the search index the terminology mapper uses. Note the embedding model you choose
   (for example `text-embedding-3-small`): you pass the same name to the mapper later.
4. **Set an API key** for the model provider, e.g. `OPENAI_API_KEY`
   ([Environment variables](env.md)).

!!! tip "Which definitions need what"
    | Definition | Table | Needs the terminology mapper? |
    |---|---|---|
    | `ecog_performance_status` | `measurement` | no |
    | `smoking_status` | `observation` | no |
    | `lab_results` | `measurement` | yes, to code test names |
    | `family_history` | `observation` | yes, to code conditions |
    | `medical_history` | `observation` | yes, to code conditions |
    | `condition_occurrence` | `condition_occurrence` | yes, to code conditions |

    Definitions without the mapper only need the vocabulary database and an LLM, so
    they are the easiest way to start.

---

## Key terms

You do not need to know OMOP well to follow this page, but these words come up often.

| Term | Meaning |
|---|---|
| **CDM table** | A table of the OMOP Common Data Model, e.g. `measurement`. Each row is one fact about a patient. |
| **Concept** | An entry of the OMOP vocabularies, e.g. *Body weight*. Columns ending in `_concept_id` hold concept ids, numbers such as `3025315`. |
| **Vocabulary and code** | Where a concept comes from and its code there, e.g. LOINC `29463-7`. On this page codes are written `LOINC:29463-7`. |
| **Standard concept** | The concept OMOP uses for analysis when several mean the same thing. `*_concept_id` columns must hold standard concepts, from any vocabulary. A non-standard concept usually **maps to** a standard one. |
| **Domain** | The kind of concept: Measurement, Observation, Condition, Unit... Each column expects certain domains, e.g. `measurement_concept_id` takes Measurement concepts. |
| **Source value** | The text as written in the note, kept next to the code, e.g. `measurement_source_value = "ECOG 4"`. |
| **`0` and `NULL`** | `0` means "there is a value, but it has no concept". Empty (`NULL`) means "nothing there". |

---

## Your first run

This uses the predefined ECOG performance status definition, which needs no mapper.

```python title="ecog_to_omop.py"
import asyncio

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

tables = asyncio.run(pipeline(notes))
measurement = tables["measurement"]
print(measurement.dropna(axis=1, how="all").to_string())  # only the filled columns
```

!!! note "Scripts and notebooks"
    The pipeline is asynchronous. In a script, wrap the call in `asyncio.run(...)` as
    above. In a Jupyter notebook, write `tables = await pipeline(notes)` instead.

The table has every column of the CDM `measurement` table; `dropna` leaves out the
empty ones when printing:

| measurement_id | person_id | measurement_concept_id | measurement_date | measurement_type_concept_id | value_as_number | value_as_concept_id | visit_occurrence_id | measurement_source_value | measurement_source_concept_id | value_source_value |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 36305384 | 2025-05-12 | 32858 | 4.0 | 4174251 | 507 | ECOG 4 | 4308014 | 4 |

### Reading the result

- `measurement_id` is generated, numbered from 1.
- `person_id` and `visit_occurrence_id` come from the `ClinicalNote`, not from the
  text. They identify the patient and visit in your database, which the note cannot
  know.
- `measurement_concept_id` **36305384** is the standard concept *ECOG Performance
  Status score* (LOINC 89247-1).
- `measurement_date` is **2025-05-12**, the assessment date written in the note. When a
  note gives no date, the note date (`2025-06-01`) is used instead.
- `measurement_type_concept_id` **32858** means *NLP*: the row was derived from text.
- `value_as_number` **4.0** and `value_as_concept_id` **4174251** (*ECOG performance
  status - grade 4*) are the grade, as a number and as a concept.
- `measurement_source_value` and `value_source_value` keep what the note said.

`ClinicalNote` takes the note `text` and the values that come **with** the note:
`person_id`, `note_id`, `note_date`, `visit_occurrence_id`, `visit_detail_id` and
`provider_id`. All but `text` are optional; columns that need a missing value stay
empty and are reported (see [Issues](#issues)). You can also pass plain strings instead
of `ClinicalNote` objects.

---

## What you get back

`pipeline(notes)` returns an `OmopTables` object:

```python
tables["measurement"]                                 # the rows, as a pandas DataFrame
tables.to_dataframes(provenance=True)["measurement"]  # plus where each row came from
tables.issues_dataframe()                             # what could not be filled, and why
tables.mappings_dataframe()                           # how extracted text was coded
tables.extraction_results                             # the extractor's raw output
```

With `provenance=True`, each row gets extra columns (`_note_id`, `_entity_id`, `_text`,
`_start`, `_end`) pointing at the exact words of the note it was built from, so every
row can be checked by a person.

### Issues

Rows are always returned, even when some columns could not be filled. Each gap is listed
in `tables.issues_dataframe()`, with the row, the column, a reason and a detail.

| Reason | Meaning | What to do |
|---|---|---|
| `missing_required` | A column the CDM requires is empty, e.g. `person_id` | Pass it in the `ClinicalNote` |
| `unparsable_date` | A date is not a full date, e.g. "on admission". The next source (often the note date) was used | Usually nothing; check the detail |
| `imputed_date_parts` | A date had no day or month, e.g. "March 2024", and was read as 2024-03-01 | Decide if day-level precision matters |
| `unmapped_label` | The model chose a label that has no concept; `0` was stored | Add the label to `concepts`, or accept `0` |
| `unmapped` | No standard concept was found for a text, e.g. the unit "lb"; `0` was stored | Review in `mappings_dataframe()` |
| `low_confidence` | The mapper's score was below your `min_score`; `0` was stored | Review the candidate in `mappings_dataframe()` |
| `domain_mismatch` | A coded concept is not of the domain the column expects | Review the mapping |
| `invalid_value` | A value does not fit its column, e.g. text that is not a number | Check the extracted text |
| `truncated` | A text was longer than its column (50 characters for `*_source_value`) | Usually nothing |
| `multiple_children` | A mention had two children of the same kind, e.g. two dates; the first was used | Check the note |

---

## Write your own: body weight, step by step

This walkthrough builds a definition from scratch: body weight, stored in the
`measurement` table with its unit. It needs no mapper.

### 1. Describe what to extract

An entity definition says what the model should find. Body weight has a value and a
unit, so they are children of the weight mention (see
[Writing entity definitions](entity_definitions.md)):

```yaml title="body_weight.yaml"
- name: Body weight
  key: body_weight
  definition: >
    The patient's measured body weight: the word for weight, the value and the unit,
    copied as written, e.g. "weight 72 kg" or "Wt: 158 lb". Stop at the unit: no
    words or punctuation after it. Does not include target or ideal weights, weight
    changes such as "lost 5 kg", or the weight of anyone other than the patient.

- name: Weight value
  key: weight_value
  definition: The number of a body weight, without the unit, e.g. "72" from "weight 72 kg".
  has_parent: true
  parent_key: body_weight

- name: Weight unit
  key: weight_unit
  definition: The unit of a body weight, copied as written without punctuation, e.g. "kg" or "lb".
  has_parent: true
  parent_key: body_weight
```

### 2. Say where each value goes

Add an `omop` block to the top-level entity. Each line names a column of the
`measurement` table and where its value comes from:

```yaml title="body_weight.yaml (first entity)"
- name: Body weight
  key: body_weight
  definition: >
    ...
  omop:
    table: measurement
    fields:
      person_id: {from: context.person_id}               # from the ClinicalNote
      measurement_concept_id: {concept: "LOINC:29463-7"}  # Body weight, always the same
      measurement_date: {from: context.note_date}
      measurement_type_concept_id: {value: 32858}         # NLP
      measurement_source_value: {from: text}              # the words of the mention
      value_as_number: {from: child.weight_value, as: float}
      unit_concept_id: {from: child.weight_unit, lookup: UCUM}  # "kg" -> kilogram
      unit_source_value: {from: child.weight_unit}
```

The `omop` block is **never sent to the model**: it only tells AATM how to fill the
table. You can check what the model will read with
`definitions.to_prompt_str()`.

### 3. Check the codes before spending model calls

```python
from aatm.data_models.extractors import EntityDefinitions
from aatm.omop import OmopTableBuilder

definitions = EntityDefinitions.from_yaml("body_weight.yaml")
OmopTableBuilder(definitions).describe_concepts()
```

| entity | column | label | code | concept_id | concept_name | vocabulary_id | domain_id |
|---|---|---|---|---|---|---|---|
| body_weight | measurement_concept_id | | LOINC:29463-7 | 3025315 | Body weight | LOINC | Measurement |

Loading the file and creating the builder already check a lot: unknown columns,
typos in `child.` or `attribute.` names, codes missing from the vocabulary, and
non-standard concepts all fail here, with a message saying what is wrong
(see [Troubleshooting](#troubleshooting)).

### 4. Run it

```python
import asyncio

from aatm.omop import ClinicalNote, OmopExtractionPipeline

notes = [
    ClinicalNote(text="Weight 72 kg today. Goal weight 65 kg.", person_id=1, note_date="2025-06-01"),
    ClinicalNote(text="Wt: 158 lb. Lost 5 lb since March.", person_id=2, note_date="2025-06-02"),
]

pipeline = OmopExtractionPipeline(definitions, model_id="openai/gpt-5-mini")
tables = asyncio.run(pipeline(notes))
tables["measurement"]
```

| measurement_id | person_id | measurement_concept_id | measurement_date | measurement_type_concept_id | value_as_number | unit_concept_id | measurement_source_value | unit_source_value |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 3025315 | 2025-06-01 | 32858 | 72.0 | 9529 | Weight 72 kg | kg |
| 2 | 2 | 3025315 | 2025-06-02 | 32858 | 158.0 | 0 | Wt: 158 lb | lb |

The goal weight and the weight loss were left out, as the definition asks.

### 5. Read the issues

```python
tables.issues_dataframe()
```

| row_id | column | reason | detail |
|---|---|---|---|
| 2 | unit_concept_id | unmapped | No concept for 'lb' with a UCUM code lookup; stored 0 |

"kg" is a UCUM code, so it was found (*kilogram*, 9529). "lb" is not: UCUM writes
pounds as `[lb_av]`. The row is kept with `unit_concept_id = 0`, which the CDM uses for
"there is a unit, but it is not mapped", and `unit_source_value` keeps `"lb"`.

!!! tip "Iterate on real notes"
    Most fixes are in the wording of the definitions. The first version of this example
    said only "copied as written", and the model returned "Weight 72 kg today." and
    "lb.". Adding "Stop at the unit: no words or punctuation after it" fixed both.

---

## Reference: the `omop` block

### Column sources

Each column has one source:

| Source | Value | Example |
|---|---|---|
| `{from: text}` | The words of the mention | `"ECOG 4"` |
| `{from: attribute.<key>}` | The label the model chose for an attribute | `"4"` |
| `{from: child.<key>}` | The text of the mention's child with that key | `"05/12/2025"` |
| `{from: context.<name>}` | A field of the `ClinicalNote` | `person_id` |
| `{value: <constant>}` | A fixed value | `32858` |
| `{concept: "<VOCABULARY>:<code>"}` | The concept id of a code | `"LOINC:29463-7"` |

A **list** of sources is tried in order, and the first that gives a value wins. The ECOG
definition uses this for the date: the assessment date if the note states a full one,
else the note date:

```yaml
      measurement_date:
        - {from: child.assessment_date, as: date}
        - {from: context.note_date}
```

### Options

| Option | Use with | Effect |
|---|---|---|
| `as: int`, `float`, `str` or `date` | `from:` | Convert the text. Dates are read from text such as `"05/12/2025"` or `"March 2024"`; decimal commas (`"11,2"`) are accepted |
| `concepts: {<label>: "<VOCABULARY>:<code>"}` | `from: attribute.<key>` | Store the concept of the chosen label instead of the label |
| `maps_to: true` | `concept:` or `concepts:` | Store the standard concept the code maps to |
| `lookup: <VOCABULARY>` | `from:` | Find the text as a code of that vocabulary, e.g. units in UCUM |
| `map: {...}` | `from:` | Code the text with the terminology mapper (see [below](#coding-text-with-the-terminology-mapper)) |

The table's id column (`measurement_id`, `observation_id`) is generated, numbered from
`id_start` (default 1). Only top-level entities can have an `omop` block; a child's text
is used from its parent with `from: child.<key>`.

### Fixed codes, standard concepts and domains

Write codes as `VOCABULARY:code`, as they appear in the vocabulary, rather than as
concept ids, so the file stays readable. They are looked up when the builder is
created, and these fail before any model call:

- a code that is not in the vocabulary;
- a non-standard concept in a `*_concept_id` column. Add `maps_to: true` to use the
  standard concept it maps to;
- a concept of a domain the column does not accept, e.g. a Condition concept in
  `measurement_concept_id`;
- `maps_to: true` on a code without exactly one standard mapping.

`*_source_concept_id` columns keep the code as written, standard or not, as the CDM
asks. The ECOG definition shows why both exist:

```yaml
      measurement_concept_id: {concept: "SNOMED:423740007", maps_to: true}
      measurement_source_concept_id: {concept: "SNOMED:423740007"}
```

SNOMED 423740007 *ECOG performance status* is not standard; it maps to LOINC 89247-1
*ECOG Performance Status score*, which goes to `measurement_concept_id`, while the
SNOMED concept is kept in `measurement_source_concept_id`.

!!! note "Standard, from any vocabulary"
    A `*_concept_id` column needs a **standard** concept of the right domain. Its
    vocabulary does not matter: SNOMED, LOINC, RxNorm or any other. The rule applies to
    every source: fixed codes and label maps are checked when the builder is created,
    and a `map` or `lookup` that finds only a non-standard concept stores `0` and
    reports `unmapped`.

---

## Coding text with the terminology mapper

Some concepts never change (body weight is always *Body weight*), but many depend on
what the note says: the name of a lab test, a condition. A `map` source sends the
extracted text to the same `TerminologyMapper` used for
[terminology mapping](quick_start_python_api.md), which searches the vocabularies and
picks the best standard concept.

The predefined `lab_results` definition codes the test name this way:

```yaml
      measurement_concept_id:
        from: text
        map: {}          # any standard concept of the column's domain (Measurement)
```

The YAML says **what** to code. Your code chooses **how**, by configuring the mapper,
because that involves search models, LLMs and API keys:

```python
import asyncio

from aatm.omop import ClinicalNote, OmopExtractionPipeline
from aatm.selectors import OpenAILLMSelector
from aatm.terminology_mapper import TerminologyMapper

mapper = TerminologyMapper(
    retriever="text-embedding-3-small",          # the embedding model chosen in `aatm init`
    selector=OpenAILLMSelector("gpt-5-mini"),    # an LLM picks among the candidates
)
mapper.retriever.top_k = 30                      # how many candidates it sees

pipeline = OmopExtractionPipeline("lab_results", model_id="openai/gpt-5-mini", mapper=mapper)
notes = [ClinicalNote(text="Labs 03/15/2025: Hb 11.2 g/dL, creatinine 1.4 mg/dl.", person_id=10)]
tables = asyncio.run(pipeline(notes))
tables.mappings_dataframe()
```

`mappings_dataframe()` has one row per coded text, so you can review every choice:

| row_id | column | text | method | concept_id | concept_name | score | stored |
|---|---|---|---|---|---|---|---|
| 1 | measurement_concept_id | Hb | map | 3000963 | Hemoglobin [Mass/volume] in Blood | 0.30 | 3000963 |
| 1 | unit_concept_id | g/dL | lookup | 8713 | gram per deciliter | | 8713 |

How it works:

- **The search is limited to a domain, not a vocabulary.** By default it is the domain
  the column requires (Measurement here). `map: {domain: Condition}` sets it
  explicitly, and `map: {vocabularies: [LOINC]}` narrows the search further if your
  data must use one vocabulary.
- **Each distinct text is coded once per run**: a hundred notes mentioning "Hb" make
  one mapper query.
- **`min_score`** (`map: {min_score: 0.8}`) stores `0` and reports `low_confidence`
  below that score. Scores depend on the retriever, so look at real scores in
  `mappings_dataframe()` before choosing a threshold.

!!! warning "Review the mappings"
    Short names are ambiguous. In a small test with nine lab results, the default
    selector, which takes the closest match, coded only 2 test names correctly; with
    `OpenAILLMSelector` choosing among 30 candidates, 7 of 9 were correct. Without a
    vocabulary filter, some tests map to general SNOMED procedures such as *Potassium
    measurement* rather than the more specific LOINC *Potassium [Moles/volume] in Serum
    or Plasma*; both are valid standard concepts.

### Code lookups

`lookup: <VOCABULARY>` finds the extracted text as a code of a vocabulary. It suits
units: `"g/dL"` is the UCUM code of *gram per deciliter*. An exact match wins; otherwise
a single standard concept matching while ignoring case is used (`"mg/dl"` → `mg/dL`).
Texts that are not codes, such as `"lb"` or `"x10^3/uL"`, are reported as `unmapped`.

---

## The observation table

The `observation` table holds facts that are not measurements, conditions, procedures,
drugs or devices: smoking, family history, social and lifestyle facts. Each row pairs
an `observation_concept_id` (what was observed) with a value: a concept, a number, a
text or a date.

Several definitions can run together, so each note is sent to the model only once:

```python
import asyncio

from aatm.omop import ClinicalNote, OmopExtractionPipeline
from aatm.selectors import OpenAILLMSelector
from aatm.terminology_mapper import TerminologyMapper

mapper = TerminologyMapper(retriever="text-embedding-3-small", selector=OpenAILLMSelector("gpt-5-mini"))
mapper.retriever.top_k = 30

pipeline = OmopExtractionPipeline(
    ["smoking_status", "family_history", "medical_history"],  # names, YAML paths, or both
    model_id="openai/gpt-5-mini",
    mapper=mapper,  # family_history and medical_history code conditions with it
)
notes = [
    ClinicalNote(
        text=(
            "Former smoker, 30 pack-years. Mother had breast cancer at 45. "
            "History of appendicitis."
        ),
        person_id=20,
        note_date="2025-06-01",
    ),
]
tables = asyncio.run(pipeline(notes))
tables["observation"]
```

A typical result (concept names added in italics):

| observation_id | observation_concept_id | observation_source_value | value_as_number | value_as_concept_id | value_source_value | qualifier_concept_id | unit_concept_id |
|---|---|---|---|---|---|---|---|
| 1 | 43054909 *Tobacco smoking status* | Former smoker | | 45883458 *Former smoker* | former | | |
| 2 | 3003421 *Cigarettes smoked total (pack per year)* | 30 pack-years | 30.0 | | 30 | | 0 |
| 3 | 4167217 *Family history of clinical finding* | breast cancer | | 4112853 *Malignant tumor of breast* | breast cancer | 4248584 *Mother* | |
| 4 | 3002660 *History of Past illness Narrative* | appendicitis | | 440448 *Appendicitis* | appendicitis | | |

**`smoking_status`** gives up to two rows per mention:

- *Smoking status*, with the status as a concept in `value_as_concept_id`: never,
  former, current every day, current some day, or current without a frequency.
  "Non-smoker" or "denies smoking" do not say whether the patient ever smoked, so they
  get the label `not_current`, stored as `0` and reported as `unmapped_label`.
- *Pack-years*, with the number in `value_as_number`. UCUM has no code for pack-years,
  so `unit_concept_id` is `0` and `unit_source_value` is `"pack-years"`.

**`family_history`** follows the CDM convention for family history:
`observation_concept_id` is *Family history of clinical finding*, `value_as_concept_id`
is the condition, coded by the mapper, and `qualifier_concept_id` is the relative
(*Mother*, *Sister*...). The CDM has no column for the relative, so that last choice is
a convention of this definition; `qualifier_source_value` keeps the label.

**`medical_history`** takes the patient's past conditions, which the CDM keeps out of
`condition_occurrence`: "history of appendicitis", "h/o stroke, no sequelae",
"hepatitis C, treated and cured". `observation_concept_id` is LOINC 11348-0 *History of
Past illness Narrative* and `value_as_concept_id` is the condition, coded by the mapper,
as for family history. Conditions the patient still has, even when listed under past
medical history ("PMH: hypertension"), go to `condition_occurrence`, and past surgeries
("s/p appendectomy") are not extracted. When the past condition happened is not
captured: `observation_date` is the note date.

!!! note "Why LOINC 11348-0"
    SNOMED *History of clinical finding in subject* is not a standard concept. The
    vocabulary maps precoordinated SNOMED concepts such as *History of myocardial
    infarction* to the condition with *Maps to value*. Their *Maps to* target is in
    the OMOP Extension vocabulary, which the other predefined definitions do not need.
    To use a different concept, copy `medical_history.yaml` and change
    `observation_concept_id`.

!!! note "Which concepts the observation table accepts"
    The CDM sends Condition, Procedure, Drug, Measurement, Device and Specimen concepts
    to their own tables, so `observation_concept_id` rejects them. `value_as_concept_id`
    accepts any domain: for family history it holds a condition.

When you combine definitions, every entity key must be unique. A clash fails with a
message naming the files:

```text
Entity keys must be unique across the combined definitions, since the model's output is matched to definitions by key:
  - key 'smoking_status' is defined in predefined 'smoking_status' and definitions/alcohol_use.yaml
List each file once, and rename keys that clash in your own files.
```

---

## The condition_occurrence table

The `condition_occurrence` table holds the patient's diagnoses, problems, signs and
symptoms. Each row is one mention of a condition, with its standard concept, its dates
and a status.

```python
import asyncio

from aatm.omop import ClinicalNote, OmopExtractionPipeline
from aatm.selectors import OpenAILLMSelector
from aatm.terminology_mapper import TerminologyMapper

mapper = TerminologyMapper(retriever="text-embedding-3-small", selector=OpenAILLMSelector("gpt-5-mini"))
mapper.retriever.top_k = 30

pipeline = OmopExtractionPipeline(
    "condition_occurrence",
    model_id="openai/gpt-5-mini",
    mapper=mapper,  # codes condition names
)
notes = [
    ClinicalNote(
        text=(
            "Type 2 diabetes since 2019. Probable community-acquired pneumonia. "
            "History of appendicitis. Rule out PE. Denies chest pain. "
            "Mother had breast cancer."
        ),
        person_id=20,
        note_date="2025-06-01",
    ),
]
tables = asyncio.run(pipeline(notes))
tables["condition_occurrence"]
```

A run with `gpt-5-mini` gave two rows (concept names added in italics):

| condition_occurrence_id | condition_concept_id | condition_source_value | condition_start_date | condition_status_concept_id | condition_status_source_value | condition_type_concept_id |
|---|---|---|---|---|---|---|
| 1 | 201826 *Type 2 diabetes mellitus* | Type 2 diabetes | 2019-01-01 | 32893 *Confirmed diagnosis* | confirmed | 32858 *NLP* |
| 2 | 4293463 *Community acquired pneumonia* | community-acquired pneumonia | 2025-06-01 | 32899 *Preliminary diagnosis* | preliminary | 32858 *NLP* |

The only issue is `imputed_date_parts` on row 1: "2019" has no month or day and is
read as 2019-01-01.

**What is not a row.** The definition follows the
[CDM conventions for this table](https://ohdsi.github.io/CommonDataModel/cdm55.html#condition_occurrence),
so the other four conditions of the note are left out:

| In the note | Why it is not here |
|---|---|
| "History of appendicitis" | Past diagnoses belong in `observation`, says the CDM: add `medical_history` to the list of definitions to keep it. Long-standing conditions the patient still has, e.g. "PMH: hypertension", are kept. |
| "Rule out PE" | Rule-out and "question of" diagnoses are not recorded in this table. |
| "Denies chest pain" | Negated. |
| "Mother had breast cancer" | Family history belongs in `observation`: add `family_history` to the list of definitions to keep it. |

**Status.** Each condition gets a required `status` label, stored in
`condition_status_concept_id`:

| Label | Means | Concept |
|---|---|---|
| `confirmed` | Stated as a fact, e.g. "hypertension", "diagnosed with asthma in 2010" | 32893 *Confirmed diagnosis* |
| `preliminary` | A likely working diagnosis, e.g. "probable pneumonia", "presumed sepsis" | 32899 *Preliminary diagnosis* |

The CDM's own examples of a status are the point in the visit when the diagnosis was
given (admitting, principal, secondary diagnosis). Notes rarely say that, so it is not
extracted.

**Dates.** `condition_start_date` is the first that applies:

1. the start date the note gives for that condition ("since 2019" → 2019-01-01,
   reported as `imputed_date_parts`);
2. the end date, when the note gives only that one ("pneumonia until 2024-04-10"): the
   CDM allows a single date for both;
3. the note date. The condition is known to be present then, but may have started much
   earlier, so read these start dates as "recorded on" rather than "onset".

The CDM notes that most sources have no condition start date, and records most
conditions as snapshots without an end date. `condition_end_date` is the end date the
note gives, e.g. for an episode in a discharge summary ("pneumonia from 03/01/2025 to
03/10/2025"). An end date before the start date is left empty and reported as
`invalid_value`.

!!! note "Which concepts the condition_occurrence table accepts"
    `condition_concept_id` takes standard concepts of the Condition domain only. The
    mapper searches that domain, and a concept of another domain, e.g. a procedure, is
    stored but reported as `domain_mismatch`. Check the coded conditions with
    `tables.mappings_dataframe()`.

---

## Troubleshooting

| You see | Why | Fix |
|---|---|---|
| `OMOP vocabulary database not found: .aatm/omop.db. Run aatm init to build it.` | The vocabulary database is missing, or the script runs from another folder | Run `aatm init`, or pass `vocabulary="path/to/omop.db"` |
| `Concepts not found in the vocabulary: ['LOINC:29463-8']` | A typo in a code, or that vocabulary was not downloaded from Athena | Check the code on Athena and that its vocabulary was included in `aatm init` |
| `... is not a standard concept. Add 'maps_to: true' to use the standard concept it maps to` | A `*_concept_id` column got a non-standard code | Add `maps_to: true`, or use the standard concept's code |
| `Table 'measurement' has no column 'value_as_numbr'` | A typo in a column name | Use the column names of the [CDM table](https://ohdsi.github.io/CommonDataModel/cdm55.html) |
| `'weight_val' is not a child entity of 'body_weight'` | `from: child.<key>` names an entity that is not a child of this one | Use the child's `key`, and check its `parent_key` |
| `... uses 'map', so the builder needs a terminology mapper` | The definition codes text, but no mapper was given | Pass `mapper=TerminologyMapper(...)` |
| `SyntaxError: 'await' outside function` | `await` used in a script | Use `asyncio.run(pipeline(notes))`, or run it in a notebook |
| Every `map` is `unmapped` | The mapper found no candidates, e.g. the search index for its retriever was not built | Use the embedding model you chose in `aatm init` as the `retriever` |

---

## Current limits

- Only the `measurement`, `observation` and `condition_occurrence` tables are supported
  so far.
- `condition_start_date` falls back to the note date when the note gives no date, which
  is later than the real onset of long-standing conditions.
- `map` searches with the extracted text alone: a unit such as mg/dL, which hints at a
  serum test, is not used.
- Mapper scores are not comparable across retrievers and selectors.
- Units written outside UCUM ("lb", "x10^3/uL", "mEq/L") are not converted before the
  lookup.
- Relative dates ("on admission", "after cycle 2") are not resolved: the next source,
  usually the note date, is used and the issue is reported.

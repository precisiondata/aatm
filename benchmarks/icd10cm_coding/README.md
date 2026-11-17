# ICD-10-CM coding benchmark (ACI-bench)

How well do LLMs code a clinical encounter with ICD-10-CM, on their own and through
AATM?

## Data

[ACI-bench](https://github.com/wyim/aci-bench) (CC BY 4.0): 207 doctor-patient
dialogues with the clinical notes written from them. Every encounter was coded with
ICD-10-CM by two experts for Yuan et al.,
[*Toward Reliable Clinical Coding with Language Models: Verification and Lightweight
Adaptation*](https://github.com/amazon-science/toward-clinical-coding-verification-adaptation)
(EMNLP 2025, Industry Track; annotations CC BY-NC 4.0).

| Split | Encounters | Codes | Distinct codes | Codes per encounter |
|---|---|---|---|---|
| train | 67 | 153 | 112 | 1–5 |
| valid | 20 | 38 | 31 | 1–4 |
| test | 120 | 247 | 132 | 1–6 |

The most frequent codes are I10, E11.9 and F32.A. Neither dataset is committed here.
`prepare_data.py` downloads both, pinned to a commit, and merges them into
`data/{train,valid,test}.jsonl` (git-ignored):

```bash
uv run python benchmarks/icd10cm_coding/prepare_data.py
```

## Methods

- **`llm`**: the model reads the note (or, with `--source dialogue`, the dialogue)
  and answers with ICD-10-CM codes and the words that support each. This is the
  paper's setting. AATM is not involved.
- **`aatm`**: `OmopExtractionPipeline` with the predefined `condition_occurrence`
  definitions. The model extracts the patient's conditions, and `AgenticCoder` (the
  concept search agent, with the same model) codes each one to a standard concept of
  the Condition domain. These are the rows AATM writes to an OMOP CDM.

## Scores

Micro precision, recall and F1 over encounters:

- **code**: exact ICD-10-CM codes (`llm` only);
- **category**: the first 3 characters, e.g. E11 for E11.9 (`llm` only);
- **standard**: gold and predicted ICD-10-CM codes are replaced by the standard
  concepts they map to ("Maps to"), and compared with AATM's concepts. This is the
  only score that compares the two methods: AATM codes to standard concepts, not to
  ICD-10-CM;
- **`llm (verified)`**: the `llm` answer without the codes that AATM's vocabulary
  database doesn't know as valid ICD-10-CM codes. It also reports invalid and
  non-billable codes.

**Standard concepts and verification need ICD10CM in `.aatm/omop.db`.** The
predefined definitions don't need it, so a bundle with only SNOMED, LOINC, UCUM and
RxNorm lacks it. Add ICD10CM to the bundle on
[Athena](https://athena.ohdsi.org/vocabulary/list) and run
`aatm init --vocab-zip <link>` again. Without it, only the code and category scores
are printed.

## Run

```bash
uv run python benchmarks/icd10cm_coding/run_benchmark.py --model openai/gpt-5-mini --split valid
```

- `--methods llm aatm`: the methods to run (both by default).
- `--split`: `train`, `valid` or `test` (default). Use `--limit N` for a quick check.
- `--source`: `note` (default) or `dialogue`.
- `--retriever`, `--vector-db`, `--max-tool-calls`: the agent's semantic search
  (default `text-embedding-3-small`, local ChromaDB) and tool budget. The retriever is
  part of the result's name, e.g. `aatm-text-embedding-3-small`.
- `--temperature`: for models that accept it.

Results go to `results/<split>__<source>__<method>__<model>.jsonl`, one line per
encounter: predictions, the agent's mappings and issues, LLM calls, tokens and
seconds. A rerun skips encounters already done and retries those that failed.
`--summary-only` scores every result file of a split and source.

## Notes from a smoke test (2026-11-17, `openai/gpt-5-mini`, valid)

- `llm`, 3 encounters: about 30 s and 3,400 tokens per encounter. Its errors were
  near misses, e.g. I50.22 for I50.21 and I10 for I11.0 (hypertensive heart
  disease with heart failure).
- `aatm`, 2 encounters: a note with 5 conditions took 65 s and 12 LLM calls. A
  longer heart-failure note made 74 calls in 600 s without finishing, which is why
  the default `--timeout` is now 1,800 s. Most of that time is local ChromaDB
  searches; the pgvector index is faster.

## Not covered yet

- `aatm` answers with standard concepts only. An ICD-10-CM answer from AATM, for the
  code and category scores, needs the agent to select non-standard ICD10CM concepts.
  Its tools return standard concepts only today.
- The gold codes include encounter codes such as Z09 (follow-up exam), which map to
  the Observation domain. `condition_occurrence` doesn't extract them, so they count
  as misses for `aatm`.

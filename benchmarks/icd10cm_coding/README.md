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
- **`aatm`**: two steps, with the same model.
  1. `SimpleLLMExtractor` with [`diagnoses.yaml`](diagnoses.yaml) extracts the
     diagnoses a coder would report for the encounter, following the outpatient
     coding guidelines: no symptoms that are part of a stated diagnosis, no uncertain
     diagnoses, chronic conditions only when addressed. Each diagnosis comes with the
     details written elsewhere in the note that change its code (laterality, acuity,
     type, a complication, the encounter for an injury).
  2. The concept search agent codes each diagnosis, with its line and details, to an
     ICD-10-CM code: `ConceptSearchAgent.search(..., vocabularies=["ICD10CM"],
     standard=False)`. In this mode its tools search ICD10CM concepts, which are not
     standard. Semantic search returns the ICD-10-CM names in the index and the codes
     that map to the standard concepts it finds, and hierarchy follows ICD-10-CM's
     own parents and children. The agent is asked for the most specific code the
     context supports.

## Scores

Micro precision, recall and F1 over encounters:

- **code**: exact ICD-10-CM codes;
- **category**: their first 3 characters, e.g. E11 for E11.9;
- **standard**: gold and predicted codes replaced by the standard concepts they map
  to ("Maps to"), the concepts an OMOP CDM would hold;
- **`llm (verified)`**: the `llm` answer without the codes that the vocabulary
  database doesn't know as valid ICD-10-CM. It also reports invalid and non-billable
  codes. `aatm` only answers with codes of the database.

**`aatm`, standard scores and verification need ICD10CM in `.aatm/omop.db`.** The
predefined definitions don't need it, so a bundle with only SNOMED, LOINC, UCUM and
RxNorm lacks it. Add ICD10CM to the bundle on
[Athena](https://athena.ohdsi.org/vocabulary/list) and run
`aatm init --vocab-zip <link>` again; rebuild the index too, so it holds the
ICD-10-CM names.

## Run

```bash
uv run python benchmarks/icd10cm_coding/run_benchmark.py --model openai/gpt-5-mini --split valid --vector-db pgvector --retriever embeddinggemma-300M
```

- `--methods llm aatm`: the methods to run (both by default).
- `--split`: `train`, `valid` or `test` (default). Use `--limit N` for a quick check.
- `--source`: `note` (default) or `dialogue`.
- `--retriever`, `--vector-db`, `--max-tool-calls`: the agent's semantic search
  (default `text-embedding-3-small`, local ChromaDB) and tool budget. The retriever is
  part of the result's name, e.g. `aatm-embeddinggemma-300M-pgvector`.
- `--temperature`: for models that accept it.

Results go to `results/<split>__<source>__<method>__<model>.jsonl`, one line per
encounter: codes, and for `aatm` each diagnosis with its details, code, confidence
and rationale; LLM calls, tokens and seconds. A rerun skips encounters already done
and retries those that failed. `--summary-only` scores every result file of a split
and source.

## Smoke test (2026-11-17, `openai/gpt-5-mini`, first 2 valid encounters)

Too few encounters to compare methods; it checks that everything runs.

| Method | Code F1 | Category F1 | Standard F1 | Codes / encounter | Tokens / encounter | Seconds |
|---|---|---|---|---|---|---|
| `llm` | 0.25 | 0.50 | 0.18 | 2.5 | 2,500 | 17 |
| `aatm` (embeddinggemma-300M, pgvector) | 0.33 | 0.67 | 0.25 | 1.5 | 14,500 | 55 |

- `aatm` coded the knee note exactly: *right knee acute medial meniscus sprain* →
  S83.241A *Other tear of medial meniscus, current injury, right knee, initial
  encounter*. `llm` added a code for the effusion.
- Both missed the same two codes of the heart-failure note: I50.22 (chronic
  systolic) instead of I50.21 (acute systolic), and I10 instead of I11.0.
- An earlier version of `aatm`, with the predefined `condition_occurrence`
  definitions and standard concepts, extracted 26 mentions from the heart-failure
  note (every symptom) and needed 57 LLM calls; the targeted definitions extract 2.

## Known limitations

- **Combination codes.** ICD-10-CM presumes hypertension and heart failure are linked
  and codes them together (I11.0). `aatm` codes each diagnosis on its own, so it
  can't produce a code that combines two of them; `llm` missed it as well.
- **Encounter codes.** The gold codes include codes such as Z09 (follow-up exam)
  that describe the visit rather than a diagnosis. `diagnoses.yaml` doesn't ask for
  them.

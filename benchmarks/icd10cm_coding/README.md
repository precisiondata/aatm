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
     coding guidelines: taken from the assessment and plan, no symptoms that are part
     of a stated diagnosis, chronic conditions only when addressed. Each diagnosis
     has a required certainty; uncertain ones ("suspected", "rule out") are not
     coded. Each comes with the details written elsewhere in the note that change
     its code (laterality, acuity, type, a complication, the encounter for an
     injury).
  2. The concept search agent codes each diagnosis to an ICD-10-CM code:
     `ConceptSearchAgent.search(..., vocabularies=["ICD10CM"], standard=False)`. In
     this mode its tools search ICD10CM concepts, which are not standard: semantic
     search returns the ICD-10-CM names in the index and the codes that map to the
     standard concepts it finds, hierarchy follows ICD-10-CM's own parents and
     children, and a code with children (a category such as K21) is sent back once
     with them listed. The agent gets the diagnosis's line, its details, the other
     diagnoses of the encounter, and four ICD-10-CM conventions (`TARGET` in
     `run_benchmark.py`): code what is stated, the combination codes ICD-10-CM
     presumes (hypertension with heart failure, diabetes with its complications),
     the 7th character of injuries, and no external-cause codes in place of the
     injury.

## Scores

Micro precision, recall and F1, pooled over every encounter of the split:

- **exact**: exact ICD-10-CM codes;
- **prefix-1, prefix-2**: the near-miss metrics of Yuan et al. A predicted code
  matches a gold code of the same encounter when they agree up to n levels above the
  gold code's leaf, never above its 3-character category: I50.2 or I50.22 for gold
  I50.21 at n=1, I50.9 at n=2. Precision counts the predictions that match a gold
  code, recall the gold codes a prediction matches;
- **category**: the first 3 characters, e.g. E11 for E11.9;
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

## Comparison with published results

Two papers report results on these annotations:
[Yuan et al. (EMNLP 2025)](https://arxiv.org/abs/2510.07629), who made them, and
[Symphony (Edin et al., 2026)](https://arxiv.org/abs/2603.29709). The summary's first
table follows their protocol:

| | Yuan et al. | Symphony | This benchmark |
|---|---|---|---|
| Input | clinical note | clinical note | clinical note (`--source note`) |
| Codes | all of ICD-10-CM | all of ICD-10-CM (Table 3) | all of ICD-10-CM |
| Encounters | 67 train / 20 dev / 120 test | not stated | test, 120 (`--split test`) |
| Metrics | exact F1, Prefix-1, Prefix-2 | micro R / P / F1 × 100 | both |
| Runs | 1 | 5, mean ± SD | `--runs 1 2 3 4 5` |

Symphony's Table 1 counts 225 distinct codes "across all splits", exactly the union
of the three splits here, so both papers use this release. Neither says how codes
are normalized; here they are compared as written, with the dot.

Their results, for reference (exact codes, full code set):

| Source | System | Model | R | P | F1 |
|---|---|---|---|---|---|
| Symphony, Table 3 | LLM, no tools | Claude Opus 4.6 | 51.0 | 45.3 | 47.6 |
| | LLM, no tools | GPT 5.4 | 74.2 | 42.5 | 54.0 ± 1.4 |
| | LLM, with tools (coding MCP) | Claude Opus 4.6 | 60.3 | 58.1 | 59.0 |
| | LLM, with tools | GPT 5.4 | 72.4 | 36.4 | 48.4 ± 0.9 |
| | LLM, with tools | Gemini 3.1 Pro | 76.5 | 32.1 | 45.2 |
| | CLH (Corti) | | 56.0 | 58.5 | 57.2 ± 1.0 |
| | Symphony | | 75.0 | 56.0 | 64.1 ± 1.7 |
| Yuan et al., Tables 1, 5 | LLM, simple prompt | Claude Haiku 3 | | | 41.6 |
| | LLM, simple prompt | Claude Sonnet 3.5 v2 | | | 43.0 |
| | LLM, fine-tuned on train | Claude Haiku 3 | | | 56.9 |
| | + verification | Claude Haiku 3, fine-tuned | | | 57.6 |

Yuan et al.'s Prefix-1 and Prefix-2 F1 for the simple prompt are 51.3 and 54.6 with
Haiku 3, and 49.6 and 53.8 with Sonnet 3.5 v2. The F1 scores were checked twice
against the papers' HTML versions; recall, precision and Prefix-n were read once:
check them against the PDFs before quoting them.

What still differs:

- **Models.** Compare models of the same tier: the papers use Claude Opus 4.6,
  GPT 5.4 and Gemini 3.1 Pro.
- **Prompt.** `llm` uses its own coder prompt, not Yuan et al.'s one-line prompt
  (from Boyle et al., 2023) or Symphony's baselines.
- **Prefix-n.** Yuan et al. describe the metric in one sentence; the definition
  above is our reading of it.
- **Split.** Symphony doesn't say which encounters it scores. Yuan et al. fine-tune
  on the train split, so their scores are on dev or test, presumably test.

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
- `--runs 1 2 3 4 5`: repeated runs, one result file each, reported as mean ± SD.
- `--variant v4`: a label added to the method's name, to keep versions apart.
- `--concurrency` (default 32): encounters coded at the same time. Every run and
  method shares one pool, so 5 runs of the test set (600 encounters) take about 25
  minutes with `gpt-5-mini`. `--max-llm-calls` caps LLM calls in flight for a
  provider quota; `--request-timeout` (default 120 s) retries a stalled request.

Results go to `results/<split>__<source>__<method>__<model>[__run<N>].jsonl`, one line
per encounter: codes, and for `aatm` each diagnosis with its details, code,
confidence and rationale, and the uncertain diagnoses left out; LLM calls, tokens and
seconds. A rerun skips encounters already done and retries those that failed.
`--summary-only` scores every result file of a split and source.

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

## Results (2026-11-17)

`aatm`, `openai/gpt-5-mini` for extraction and coding, `embeddinggemma-300M` index in
pgvector, `max_tool_calls=8`.

### Test set: 120 encounters, 5 runs

| | Exact R | Exact P | Exact F1 | Prefix-1 F1 | Prefix-2 F1 |
|---|---|---|---|---|---|
| **`aatm` v4** (gpt-5-mini) | 76.6 ± 1.3 | 78.6 ± 1.1 | **77.6 ± 1.1** | 83.8 ± 1.3 | 87.7 ± 1.3 |
| `aatm` v1 (gpt-5-mini), 1 run | 63.6 | 59.5 | 61.4 | 75.0 | 81.4 |
| Symphony (Edin et al.) | 75.0 | 56.0 | 64.1 ± 1.7 | | |
| Claude Opus 4.6 + coding MCP (Edin et al.) | 60.3 | 58.1 | 59.0 | | |
| GPT 5.4, no tools (Edin et al.) | 74.2 | 42.5 | 54.0 ± 1.4 | | |
| Haiku 3 fine-tuned + verification (Yuan et al.) | | | 57.6 | | |

v4 on test, mean of 5 runs: category F1 89.1, standard-concept F1 82.2; 2.0 codes per
encounter (gold: 2.06), no invalid code, 3.2 non-billable codes per run (34 for v1),
no failed encounter. About 18,700 tokens per encounter.

What the comparison does and doesn't show:

- Same data, split, metric and number of runs as Symphony's Table 3, and a smaller,
  cheaper model than the papers' (Claude Opus 4.6, GPT 5.4, Gemini 3.1 Pro).
- **v4 was developed on the validation split, but not blind to test.** v1's errors
  on test were examined before the validation iterations (see the history of this
  README), and two of the v4 changes address errors first seen there: codes with
  children (headers) and complications inferred from labs (E11.65). Both are
  general ICD-10-CM rules, not test cases, and validation (78.6) and test (77.6)
  agree, but a strictly held-out number would need fresh data or the changes
  re-derived from train and valid alone.
- The coding conventions encode the annotators' practice, which follows the official
  guidelines (presumed hypertension–heart failure links, no uncertain diagnoses,
  injury 7th characters).

### Development on the validation split: 20 encounters, 2 runs each

| Version | Change | Exact R | Exact P | Exact F1 |
|---|---|---|---|---|
| v1 | as first run on test | 50.0 | 45.8 | 47.8 ± 2.3 |
| v2 | + a code with children is sent back once with them (agent) | 61.8 | 54.7 | 58.0 ± 1.7 |
| v3 | + certainty (uncertain not coded), assessment and plan first, other diagnoses and coding conventions given to the agent | 68.4 | 73.3 | 70.8 ± 4.5 |
| v4 | + which links ICD-10-CM presumes and where the combination code goes; no external-cause codes for injuries | 75.0 | 82.6 | **78.6 ± 1.2** |
| `llm` | the model alone, 1 run | 47.4 | 27.7 | 35.0 |

One code is worth about 2.6 F1 points on these 38 gold codes, and two runs of the
same version differed by up to 6.4 (v3): changes smaller than about 5 points can't be told
from noise. v1 to v4 are commits on this branch; v1 is the version before
`c22be60`.

v4's remaining validation errors are one-offs: annotations that infer a detail the
rules say not to infer (acute systolic heart failure from an ejection fraction of
45%), a rare code (Q93.88 *Other microdeletions* for Smith-Magenis syndrome), a
symptom coded as a diagnosis (painful swallowing as esophagitis), and the two
limitations below.

## Known limitations

- **One code per diagnosis.** Some diagnoses need two codes: a diabetic foot ulcer
  is E11.621 plus an L97 code for the ulcer's site ("use additional code" in the
  Tabular List). The agent returns one. Combination codes the conventions name
  (I11.0 for hypertension with heart failure) are found from the other diagnoses
  of the encounter, but the rest of ICD-10-CM's links live in its Alphabetic Index
  and Tabular notes, which the OMOP vocabularies don't include: the agent relies on
  the model's knowledge of them.
- **Encounter codes.** The gold codes include codes such as Z00.00 (annual exam) and
  Z09 (follow-up after completed treatment) that describe the visit rather than a
  diagnosis. `diagnoses.yaml` doesn't ask for them.
- **Conventions the search can't infer.** "High risk medication use" is coded Z79.899
  *Other long term (current) drug therapy*, a name that shares no words with it.

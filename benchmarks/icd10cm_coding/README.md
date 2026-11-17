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

## Results: test set (2026-11-17)

`aatm`, `openai/gpt-5-mini` for extraction and coding, `embeddinggemma-300M` index in
pgvector, `max_tool_calls=8`, all 120 test encounters, one run:

| | Exact R | Exact P | Exact F1 | Prefix-1 F1 | Prefix-2 F1 | Category F1 | Standard F1 |
|---|---|---|---|---|---|---|---|
| `aatm` (gpt-5-mini) | 63.6 | 59.5 | **61.4** | 75.0 | 81.4 | 82.5 | 69.7 |

For reference, from the table above: Symphony 64.1 ± 1.7, Claude Opus 4.6 with a
coding MCP 59.0, GPT 5.4 alone 54.0, fine-tuned Haiku 3 with verification 57.6. This
is one run with a small model, not five with frontier models: not yet a like-for-like
comparison.

- 2.2 codes per encounter (gold: 2.1), no invalid code, no failed encounter; 1 of 265
  extracted diagnoses got no code.
- 7.0 LLM calls, about 17,000 tokens per encounter. Seconds per encounter (85 on
  average) are not meaningful: the first 49 encounters ran before the vocabulary
  database was indexed (`aatm index-vocab`), which made lookups up to 10,000 times
  faster.

Where the errors come from:

- **Category headers.** 34 of 264 predicted codes are not billable, e.g. K21 for
  K21.9, J45.90 for J45.909, S83.421 without its 7th character; for 21 of them the
  gold code is a child of the header. The agent may finish on a code that still has
  children.
- **E11.65 instead of E11.9** (6 times in the first 102 encounters): "uncontrolled" diabetes or a high A1c coded
  as *with hyperglycemia*, where the annotators kept type 2 diabetes without
  complications.
- **Visit codes.** Z00.00 (annual exam) and Z09 (follow-up after completed
  treatment): see Known limitations. Status codes (Z94.0, Z95.1, Z98.890) and
  screening (Z11.3) were found.
- **Conventions the search can't infer.** "High risk medication use" is coded Z79.899
  *Other long term (current) drug therapy*; semantic search finds other "high risk"
  codes and the agent concluded there was none.
- **Symptoms and history.** Symptoms the annotators coded beside a diagnosis
  (M25.561 knee pain, R51.9 headache) are missed; conditions listed only under past
  history (mitral regurgitation after its repair) are coded.

Improvements to the definitions or the agent should be developed on the train and
valid splits, then scored once on test: the errors above come from the test set.

## Known limitations

- **Combination codes.** ICD-10-CM presumes hypertension and heart failure are linked
  and codes them together (I11.0). `aatm` codes each diagnosis on its own, so it
  can't produce a code that combines two of them; `llm` missed it as well.
- **Encounter codes.** The gold codes include codes such as Z09 (follow-up exam)
  that describe the visit rather than a diagnosis. `diagnoses.yaml` doesn't ask for
  them.

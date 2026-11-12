# Concept search benchmark

Compares ways of choosing the standard concept of a lab test name
([issue #33](https://github.com/precisiondata/aatm/issues/33)):

- `first-result`: the nearest retrieved candidate (`FirstResultSelector`);
- `llm-selector`: an LLM picks among the top 30 candidates (`OpenAILLMSelector`);
- `agent`: the concept search agent (`aatm.concept_search`), given the text with its
  value, unit and sentence;
- `agent-no-context`: the same agent, given the text only.

[How much context changes the answer](#how-much-does-context-change-the-answer) is
measured separately, on the same text in several contexts.

`lab_results.csv` holds 30 lab results with gold LOINC codes: abbreviations (Hb, K,
PLT, BUN), Portuguese names (creatinina, potássio, ureia, glicemia de jejum), and
urine and 24-hour tests. `also_accept` lists equivalent codes (e.g. platelets with or
without "by Automated count"). Every method searches standard LOINC concepts of the
Measurement domain. Every gold code was checked to exist as a standard Measurement
concept in the vocabulary database.

## Run

From the repository root, after `aatm init` (vocabulary database and embedding index):

```bash
uv run python benchmarks/concept_search/run_benchmark.py --retriever text-embedding-3-small --model openai/gpt-5-mini
```

`--methods` runs a subset; earlier results of the other methods are kept. Results go to
`results/<retriever>__<model>.csv`, one row per text and method, with the agent's
rationale, trace and usage.

## Results (2026-11-09)

`text-embedding-3-small` index, `openai/gpt-5-mini`, `max_tool_calls=8`:

| Method | Correct | LLM calls / text | Input tokens / text | Output tokens / text | Seconds / text |
|---|---|---|---|---|---|
| first-result | 1 / 30 | 0 | – | – | 7.7 |
| llm-selector | 14 / 30 | 1 | not reported | not reported | 13.5 |
| agent-no-context | 27 / 30 | 2.2 | 3,328 | 936 | 66.0 |
| **agent** | **29 / 30** | 2.2 | 3,379 | 903 | 28.0 |

Seconds are per text, with up to 8 agent searches running concurrently. Most of it is
filtered ChromaDB queries (about 5 s each on this 2.5-million-expression index), not
the LLM. The agent used 2.2 tool calls per text on average, at most 5, and never ran
out of budget; 26 answers were *high* confidence and 4 *medium*.

Its one miss: *BUN* → `6299-2` *Urea nitrogen [Mass/volume] in Blood* instead of
`3094-0` *in Serum or Plasma*.

## How much does context change the answer?

[Issue #58](https://github.com/precisiondata/aatm/issues/58): can one answer be reused
across contexts (a coarser cache key, #50), or the context be skipped (cheap tiers,
#57)? `run_context_benchmark.py` measures it on `context_cases.csv`: 15 lab test names
in 57 contexts, each with a gold LOINC code. In 10 texts the context changes the right
concept (*Hb 7.8 %* is HbA1c; *glucose* in a CSF panel, a urinalysis or fasting labs;
*creatinine* in µmol/L, 24-hour or spot urine; *Na* and *K* in urine electrolytes;
*troponin* I or high-sensitivity T; *CRP* high-sensitivity; *HbA1c* in mmol/mol). In
5 it doesn't (TSH, ALT, PLT, ferritin, INR, in different units, values and languages).

The agent codes every case under three variants, three times each, every case
searched separately:

- **full**: text, value, unit, operator, sentence and target, as `AgenticCoder` sends
  them;
- **unit**: text, target, operator and unit, without the value or the sentence;
- **text**: the text only.

```bash
uv run python benchmarks/concept_search/run_context_benchmark.py --retriever text-embedding-3-small --model openai/gpt-5-mini
```

### Results (2026-11-12)

`text-embedding-3-small` index, `openai/gpt-5-mini`, `max_tool_calls=8`, 513 searches,
none failed or out of budget.

| Variant | Accuracy (range of 3 runs) | Context matters (41 cases) | Same concept (16 cases) | High confidence | Input tokens / text |
|---|---|---|---|---|---|
| **full** | **95.9%** (94.7–96.5) | **95.1%** | 97.9% | 88% | 4,153 |
| unit | 76.0% (71.9–78.9) | 69.9% | 91.7% | 61% | 3,658 |
| text | 62.6% (57.9–66.7) | 49.6% | 95.8% | 26% | 3,207 |

**Context, not noise.** The gap between variants (20 and 33 points) is far larger
than the run-to-run range (at most 9 points). With the full context, 93% of cases got
the same answer in all three runs; 86% with less context.

**When two variants disagree on a case, the full context is right**: in 36 of 41
disagreements with *unit* (2 the other way) and 60 of 66 with *text* (3 the other
way).

**Consistency** (pairwise agreement between contexts of the same text, accepted
alternatives counted as one answer):

| | Same concept | Context matters |
|---|---|---|
| gold | 1.00 | 0.28 |
| full | 0.97 | 0.25 |
| unit | 0.88 | 0.60 |
| text | 0.93 | 0.86 |

With the full context the agent varies as much as the gold does. Without it, it gives
one answer where the gold has several.

**What the unit recovers, and what only the sentence has.** The unit alone is enough
when it implies the property or specimen: creatinine in µmol/L or g/24 h, protein in
mg/24 h, HbA1c in mmol/mol, Hb in %, urine WBC in /hpf (all right under *unit*, wrong
under *text*). It is not enough when only the sentence states the specimen or method:
urine or CSF glucose, CSF protein, urine albumin, spot urine creatinine, urine Na and
K, CSF WBC, fasting glucose and high-sensitivity CRP all became the serum or blood
concept under *unit*, in all three runs (high-sensitivity troponin T in one of three).

**Remaining errors with full context** (7 of 171): *Serial cTnI: troponin <0.01
ng/mL* → `49563-0` *Troponin I … by Detection limit <= 0.01 ng/mL* (3 runs; arguably
acceptable, the gold is the plain troponin I); *cTnI 0.03 ng/mL* → `89579-7`
high-sensitivity troponin I (1); urine *WBC 15 /hpf* → `30405-5` *Leukocytes [#/volume]
in Urine* (1); Portuguese *TSH 4,8 µUI/mL* → `3015-5` *in Blood* (1); *fasting labs:
glucose* → `2345-7` random glucose (1).

### Recommendation

**#50, a coarser cache key: don't.** Reusing the full-context answer of one note for
every case with the same text, target, attributes and unit drops accuracy from 95.9%
to 84.2%; keying on the text alone, to 59.3%:

| Cache key | Distinct keys (57 cases) | Accuracy |
|---|---|---|
| full context (exact) | 57 | 95.9% |
| text, target, attributes, unit | 41 | 84.2% |
| text | 16 | 59.3% |

The errors are exactly the specimen cases above, and are confident (urine potassium
cached as serum potassium). The only field the dataset never needed is the value, but
the value is also written in the sentence, so dropping it from the key alone would
rarely produce a hit. Keep the exact key for coders that read context; the exact key
still helps reruns and templated notes.

**#57, cheap tiers: they may skip the agent only when the text says everything.** A
deterministic lookup or a retrieval result sees the text alone. On texts whose concept
doesn't depend on context, the text alone is as good as the full context (95.8% vs
97.9%); on texts whose concept does, it is right half the time (49.6%). So a cheap
tier is safe for a text when:

- the text itself names the specimen, method and property, e.g. *urine
  albumin/creatinine ratio*, *24h urine protein*, *fasting glucose*; or
- neither the unit nor the sentence contradicts the default specimen and property:
  the unit agrees with the concept's property (mass, moles, units per volume, %), and
  the sentence has no specimen or method cue (*urine*, *urinalysis*, *CSF*, *24-hour*,
  *spot*, *fasting*, *high-sensitivity*, *IFCC*, …).

Everything else should go to the agent with the full context. Calibrate the tiers'
thresholds on `context_cases.csv` as well as `lab_results.csv`: in `lab_results.csv`
every text has one context, so it cannot show this failure.

Cost of the full context: 14% more input tokens than *unit* and 29% more than *text*
per search, and 2.6 LLM calls per text instead of 2.4 and 2.2.

# Attribute labelling benchmark

Should a decision model label closed-set entity attributes, and in which mode
([issue #81](https://github.com/precisiondata/aatm/issues/81))? Every mention of
`cases.csv` is labelled by:

- `extractor`: the extractor's own labels (`SimpleLLMExtractor`, no labeller);
- `verify`: the same extraction, each label checked by a decision model
  (`DecisionAttributeLabeller(mode="verify")`); a label is written only when both
  agree;
- `replace`: an extraction whose prompt leaves the attributes out, labelled by the
  decision model (`mode="replace"`).

Each mention ends as **correct**, **wrong**, **review** (the label was withheld for a
person to check) or **missed** (the extractor did not find the mention). `verify` and
`replace` are also scored at `min_probability` 0.5, 0.8 and 0.9, from the same calls.

`cases.csv` holds 50 mentions in 44 short notes, written by hand (no patient data;
edit `make_cases.py` and run it to change them):

- 26 smoking statuses (`smoking_status.status`, 6 labels): plain statements, forms
  ("Tobacco use: yes"), Portuguese, and harder ones: "lifelong non-smoker" (`never`,
  not `not_current`), "relapsed to smoking last month, about 5 cigarettes a day",
  a family member's smoking or vaping next to the patient's;
- 24 ECOG labels (`score`, an `ordered` attribute, and `score_range_min` / `_max`):
  digits, words ("Zubrod score of one", "grade II", "ECOG zero"), ranges (the single
  `score` is then not stated), two time points in one sentence, and an eligibility
  threshold next to the patient's grade.

## Run

From the repository root, with the keys in `.env`:

```bash
uv run python benchmarks/attribute_labels/run_benchmark.py --model openai/gpt-5-mini --decision-models openrouter/typesafe/jev-1.13 openai/gpt-6-luna gemini/gemini-decision-1
```

Extractions are stored in `results/` and reused (`--fresh` redoes them), so adding a
decision model costs only its own calls. Results: `results/<model>__<decision
models>.csv`, one row per setup and mention with the label written, the decision
model's answer and its probability, and a `.json` summary.

## Results (2026-11-12)

`openai/gpt-5-mini` extractor; seconds and cost include the extraction (44 calls,
8 at a time) and the labelling (45 decision calls, 8 at a time).

| Setup | Correct | Wrong | Review | Missed | Precision of written labels | USD / 1,000 mentions | Seconds |
|---|---|---|---|---|---|---|---|
| extractor | **50** | 0 | 0 | 0 | 1.00 | 1.88 | 39.4 |
| verify jev-1.13 | **50** | 0 | 0 | 0 | 1.00 | 1.91 | 47.7 |
| verify jev-1.13 @0.8 | 48 | 0 | 2 | 0 | 1.00 | 1.91 | 47.7 |
| replace jev-1.13 | **50** | 0 | 0 | 0 | 1.00 | **1.64** | 47.3 |
| replace jev-1.13 @0.8 | 49 | 0 | 1 | 0 | 1.00 | 1.64 | 47.3 |
| verify gpt-6-luna | 48 | 0 | 2 | 0 | 1.00 | 1.96 | 42.8 |
| replace gpt-6-luna | 48 | 2 | 0 | 0 | 0.96 | 1.68 | 40.9 |
| replace gpt-6-luna @0.5 | 48 | 1 | 1 | 0 | 0.98 | 1.68 | 40.9 |
| verify gemini-decision-1 | 49 | 0 | 1 | 0 | 1.00 | 1.88* | 44.4 |
| replace gemini-decision-1 | 49 | 1 | 0 | 0 | 0.98 | 1.61* | 42.1 |
| replace gemini-decision-1 @0.5 | 49 | 0 | 1 | 0 | 1.00 | 1.61* | 42.1 |

\* Without the Gemini decision calls: they have no price in the script and the API
reports none. At 0.9, every model sends 9–13 correct labels to review.

How well the probabilities match the answers (`replace`, all 50 answers):

| Decision model | Accuracy | Brier score | Mean probability when right | when wrong | Labelling: calls, seconds, USD |
|---|---|---|---|---|---|
| jev-1.13 (OpenRouter) | 50 / 50 | 0.006 | 0.95 | – | 45, 8.8 s, 0.0017 |
| gpt-6-luna | 48 / 50 | 0.029 | 0.96 | 0.67 | 45, 2.4 s, 0.0040 |
| gemini-decision-1 | 49 / 50 | 0.027 | 0.91 | 0.47 | 45, 3.6 s, – |

The errors:

- **gpt-6-luna, "ECOG 1-2"**: `score_range_min` "not stated" (probability 0.98) and
  `score_range_max` `1` (0.36). The first is a confident error no threshold catches.
- **gemini-decision-1, "Relapsed to smoking last month"** (the line goes on: "about 5
  cigarettes a day"): `current_unknown_frequency` instead of `current_every_day`
  (0.47), caught at a 0.5 threshold.

## What this says

- **This set is too easy to rank the setups by accuracy.** The extractor alone labels
  all 50 mentions right, so a decision model can only match it or add errors. Harder
  or longer notes, more labels, or a smaller extractor model are needed to see
  whether `verify` catches the extractor's mistakes.
- **`verify` never wrote a wrong label.** Every disagreement went to review: 0–2 of
  50 mentions with no threshold, at a cost of about 2 % more than the extractor
  alone.
- **`replace` is about 13 % cheaper** (1.64 vs 1.88 USD per 1,000 mentions), because
  the extraction prompt and output lose the attribute descriptions and labels; the
  decision calls cost almost nothing next to the extraction (Jev: 0.0017 USD for 45
  calls). But `replace` writes whatever the decision model answers: with Jev that was
  50 / 50, with gpt-6-luna and gemini-decision-1 it wrote 1–2 wrong labels.
- **Thresholds:** 0.5 caught the low-probability errors; 0.8 and 0.9 mostly sent
  correct labels to review. The probabilities are not comparable across models
  (Gemini's are lower overall), so a threshold has to be set per model.

**Decision: keep `verify` as the default.** `replace` with Jev matched the extractor
here at a lower cost, but 50 easy mentions are not enough to trust one decision
model's labels without the extractor's check. Revisit with a harder set.

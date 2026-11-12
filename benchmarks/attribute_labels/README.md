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

Each mention ends as:

- **correct** or **wrong**: the label written (or its absence, for "not stated");
- **review**: the label was withheld for a person to check;
- **dropped**: the mention was found but rejected, because a required attribute got
  no label (drug `kind` is required);
- **missed**: the extractor did not find the mention (no case ended so).

`verify` and `replace` are also scored at `min_probability` 0.5, 0.8 and 0.9, from
the same calls.

`cases.csv` holds 112 mentions in 81 notes, written by hand (no patient data; edit
`make_cases.py` and run it to change them), in two sets:

- **basic**, 50 mentions in one-line notes:
  - 26 smoking statuses (`smoking_status.status`, 6 labels): plain statements, forms
    ("Tobacco use: yes"), Portuguese, "lifelong non-smoker" (`never`, not
    `not_current`), relapse, and a relative's smoking or vaping next to the
    patient's;
  - 24 ECOG labels (`score`, an `ordered` attribute, and `score_range_min` / `_max`):
    digits, words ("Zubrod score of one", "grade II"), ranges (the single `score` is
    then not stated), two time points, an eligibility threshold.
- **hard**, 62 mentions, often in longer notes:
  - 24 drug labels on 12 drugs: `kind` (administered / prescribed / taking,
    required) and `route` (13 labels, or not stated), with sections ("HOME
    MEDICATIONS:" then "PLAN:"), abbreviations (SL, IM, OU, MDI) and Portuguese;
  - 14 `family_history.relative` labels (16 labels): sections ("Father: MI at 60,
    type 2 diabetes"), maternal aunt, niece (`other_relative`), "one of her parents"
    (`parent`), Portuguese;
  - 6 smoking statuses: forms over several lines, "smoked 2 packs/day … until his
    CABG", "smoking again after quitting", smokeless tobacco;
  - 18 ECOG labels: "PS 1 per ECOG", KPS or a trial threshold next to the grade,
    "ECOG PS 2 to 3", "ECOG: II-III".

## Run

From the repository root, with the keys in `.env`:

```bash
uv run python benchmarks/attribute_labels/run_benchmark.py --models openai/gpt-5-mini openai/gpt-5-nano --decision-models openrouter/typesafe/jev-1.13 openai/gpt-6-luna gemini/gemini-decision-1
```

`--sets basic` or `--sets hard` runs one set. Extractions are stored in `results/`
and reused while the model, definitions and notes are unchanged (`--fresh` redoes
them), so adding a decision model costs only its own calls. Results:
`results/<model>__<decision models>.csv`, one row per setup and mention with the
label written, the decision model's answer and its probability, and a `.json`
summary.

## Results (2026-11-12)

Two extractors, `openai/gpt-5-mini` and the smaller `openai/gpt-5-nano`, and three
decision models: TypeSafe's `jev-1.13` (OpenRouter), `gpt-6-luna` (OpenAI Decisions
API) and `gemini-decision-1` (Gemini Decisions API). Every note is extracted with the
smoking status, ECOG, drug exposure and family history definitions together.

### Hard set (62 mentions)

| Extractor | Setup | Correct | Wrong | Review | Dropped | Precision |
|---|---|---|---|---|---|---|
| gpt-5-mini | extractor alone | **61** | 1 | 0 | 0 | 0.98 |
| | verify jev-1.13 | 55 | **0** | 7 | 0 | 1.00 |
| | verify gpt-6-luna | 58 | **0** | 4 | 0 | 1.00 |
| | verify gemini-decision-1 | 56 | **0** | 6 | 0 | 1.00 |
| | replace jev-1.13 | 57 | 1 | 0 | 4 | 0.98 |
| | replace gpt-6-luna | 58 | 4 | 0 | 0 | 0.94 |
| | replace gemini-decision-1 | 57 | 1 | 0 | 4 | 0.98 |
| gpt-5-nano | extractor alone | 37 | 9 | 0 | 16 | 0.80 |
| | verify jev-1.13 | 35 | **0** | 11 | 16 | 1.00 |
| | verify gpt-6-luna | 34 | 1 | 11 | 16 | 0.97 |
| | verify gemini-decision-1 | 37 | 1 | 8 | 16 | 0.97 |
| | replace jev-1.13 | 57 | 1 | 0 | 4 | 0.98 |
| | replace gpt-6-luna | 56 | 4 | 0 | 2 | 0.93 |
| | replace gemini-decision-1 | **59** | 1 | 0 | 2 | 0.98 |

Precision is the share of written labels that are right.

### Basic set (50 mentions)

| Extractor | Setup | Correct | Wrong | Review | Dropped |
|---|---|---|---|---|---|
| gpt-5-mini | extractor alone | **50** | 0 | 0 | 0 |
| | verify jev / luna / gemini | 50 / 48 / 49 | 0 / 0 / 0 | 0 / 2 / 1 | 0 |
| | replace jev / luna / gemini | 50 / 48 / 49 | 0 / 2 / 1 | 0 | 0 |
| gpt-5-nano | extractor alone | 49 | 1 | 0 | 0 |
| | verify jev / luna / gemini | 48 / 47 / 46 | 0 / 0 / 0 | 2 / 3 / 4 | 0 |
| | replace jev / luna / gemini | **50** / 48 / 47 | 0 / 1 / 3 | 0 / 1 / 0 | 0 |

### Cost and time (both sets, 112 mentions)

| Setup | gpt-5-mini: USD / 1,000 mentions | gpt-5-nano: USD / 1,000 mentions |
|---|---|---|
| extractor alone | 2.05 | 0.77 |
| verify jev-1.13 | 2.08 | 0.80 |
| verify gpt-6-luna | 2.11 | 0.83 |
| replace jev-1.13 | **1.65** | 0.70 |
| replace gpt-6-luna | 1.68 | 0.74 |
| replace gemini-decision-1 | 1.62* | **0.67*** |

\* Without the Gemini decision calls: they have no price in the script and the API
reports none. The decision calls are cheap next to the extraction: 88–89 calls cost
0.003 USD with Jev and 0.007 USD with gpt-6-luna. The extraction takes 1.5 min
(gpt-5-mini) to 3 min (gpt-5-nano) for 81 notes, 8 at a time; labelling adds 4–16 s.

### Thresholds and calibration

At `min_probability` 0.8, every setup sends 2–16 more labels to review, nearly all of
them correct: with gpt-5-mini on the hard set, `verify` with Jev goes from 55 to 51
correct with no wrong label gained or lost. 0.5 changes little. The models'
probabilities are not comparable (Gemini's are lower overall), so a threshold has to
be set per model and checked on your notes.

| Decision model (replace, gpt-5-mini) | Accuracy, basic | Accuracy, hard | Brier, hard | Mean p right / wrong, hard |
|---|---|---|---|---|
| jev-1.13 | 50 / 50 | 59 / 62 | 0.040 | 0.96 / 0.76 |
| gpt-6-luna | 48 / 50 | 58 / 62 | 0.057 | 0.94 / 0.80 |
| gemini-decision-1 | 49 / 50 | 59 / 62 | 0.036 | 0.94 / 0.65 |

Wrong answers come with high probabilities (0.65–0.80 on average), which is why
thresholds catch few of them.

### Where the labels go wrong

- **The decision model sees only the mention's line.** In a "HOME MEDICATIONS:"
  section, `- lisinopril 10 mg PO daily` alone does not say the drug is taken at
  home, so Jev and Gemini answer drug `kind` "not stated" (p 0.69). In `verify` the
  label goes to review; in `replace`, `kind` is required, so the mention is
  **dropped**: the 4 dropped labels of `replace` are those two drugs. The extractor,
  which reads the whole note, gets them right.
- **The small extractor fails on required labels.** gpt-5-nano left out or
  mislabelled the required drug `kind` for 8 drugs, so validation rejected them (16
  dropped labels), and it left out the route of 6 drugs that state it. `verify` turns
  its wrong labels into reviews but cannot bring back the dropped mentions. In
  `replace`, nano's prompt has no attributes, it finds every drug, and the decision
  model labels them: 56–59 correct instead of 37.
- **ECOG ranges (gpt-6-luna)**: in 3 of the 4 ranges with a lower bound ("ECOG 1-2",
  "ECOG PS 2 to 3", "II-III") it answers `score_range_min` "not stated", with p up
  to 0.98. Gemini misses the lower bound of "II-III" (0.69); Jev gets every range
  right.
- **"Fentanyl 25 mcg/h patch changed today by nursing"**: Jev says `prescribed`
  (0.98); the gold label is `administered` (given by a health professional), which
  gpt-6-luna and Gemini give.

## What this says

- **No mode wins everywhere, so `mode` stays required.**
- **With a strong extractor, use `verify`.** gpt-5-mini alone made 1 mistake in 112
  labels. `verify` wrote no wrong label with any decision model, at 1–3 % more cost,
  but withheld 4–7 correct labels on the hard set for review. `replace` saved about
  20 % but wrote 1–4 wrong labels and dropped 2 drugs whose kind depends on a
  section heading.
- **With a small extractor, `replace` is far better.** gpt-5-nano alone got 37 of 62
  hard labels right and dropped 8 drugs. `replace` with Jev or Gemini got 57–59 right
  at the lowest cost of all (0.67–0.70 USD per 1,000 mentions), better than
  gpt-5-mini in `replace` and close to gpt-5-mini alone at a third of its cost.
  `verify` cannot help there: it only checks mentions the extractor kept.
- **Jev and Gemini were the most accurate decision models;** gpt-6-luna fails on
  most ECOG ranges.
- **Thresholds above 0.5 mostly withhold correct labels.**
- **Next:** give the labeller more of the note than the mention's line (e.g. the
  section heading, or a few lines before), so labels that depend on a section are
  not lost. Until then, prefer `verify` when labels depend on sections.

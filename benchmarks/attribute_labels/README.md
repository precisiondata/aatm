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

`cases.csv` holds 130 mentions in 83 notes, written by hand (no patient data; edit
`make_cases.py` and run it to change them), in two sets:

- **basic**, 50 mentions in one-line notes:
  - 26 smoking statuses (`smoking_status.status`, 6 labels): plain statements, forms
    ("Tobacco use: yes"), Portuguese, "lifelong non-smoker" (`never`, not
    `not_current`), relapse, and a relative's smoking or vaping next to the
    patient's;
  - 24 ECOG labels (`score`, an `ordered` attribute, and `score_range_min` / `_max`):
    digits, words ("Zubrod score of one", "grade II"), ranges (the single `score` is
    then not stated), two time points, an eligibility threshold.
- **hard**, 80 mentions, often in longer notes:
  - 44 drug labels on 22 drugs: `kind` (administered / prescribed / taking,
    required) and `route` (13 labels, or not stated), with abbreviations (SL, IM, OU,
    MDI) and Portuguese;
  - 18 `family_history.relative` labels (16 labels): maternal aunt, niece
    (`other_relative`), "one of her parents" (`parent`), Portuguese;
  - 7 smoking statuses: forms over several lines, "smoked 2 packs/day … until his
    CABG", "smoking again after quitting", smokeless tobacco;
  - 11 ECOG labels: "PS 1 per ECOG", KPS or a trial threshold next to the grade,
    "ECOG PS 2 to 3", "ECOG: II-III".

  18 of them (tag `section`) depend on a **section heading**, not on their own
  line: drugs listed under "HOME MEDICATIONS:", "GIVEN IN THE ED:" or "PLAN:" in
  an admission note, and diseases listed under "Mother:" or "Brother:" in a family
  history.

## Run

From the repository root, with the keys in `.env`:

```bash
uv run python benchmarks/attribute_labels/run_benchmark.py --models openai/gpt-5-mini openai/gpt-5-nano --decision-models openrouter/typesafe/jev-1.13 openai/gpt-6-luna gemini/gemini-decision-1 --contexts line section before3 section+before3 --thresholds 0.5 0.8
```

`--contexts` sets what the decision model reads besides the mention: `line` (its
line only), `section` (`section=True`: the nearest heading above), `before<n>`
(`lines_before=n`) or both. `--sets basic` or `--sets hard` runs one set.
Extractions are stored in `results/` and reused while the model, definitions and
notes are unchanged (`--fresh` redoes them), so adding a decision model or a context
costs only the decision calls. Results: `results/<model>__<decision models>.csv`,
one row per setup and mention with the label written, the decision model's answer
and its probability, and a `.json` summary.

## Results (2026-11-12)

Two extractors, `openai/gpt-5-mini` and the smaller `openai/gpt-5-nano`; three
decision models, TypeSafe's `jev-1.13` (OpenRouter), `gpt-6-luna` (OpenAI Decisions
API) and `gemini-decision-1` (Gemini Decisions API); four contexts. Every note is
extracted with the smoking status, ECOG, drug exposure and family history
definitions together. Cells are correct / wrong / review / dropped.

### Hard set, gpt-5-mini (80 mentions)

| Setup | line | section | before3 | section+before3 |
|---|---|---|---|---|
| extractor alone | 78 / 2 / 0 / 0 | | | |
| verify jev-1.13 | 53 / 0 / 27 / 0 | **75 / 0 / 5 / 0** | 72 / 0 / 8 / 0 | 72 / 0 / 8 / 0 |
| verify gpt-6-luna | 64 / 0 / 16 / 0 | 70 / 0 / 10 / 0 | 69 / 0 / 11 / 0 | 69 / 0 / 11 / 0 |
| verify gemini-decision-1 | 62 / 0 / 18 / 0 | **76 / 0 / 4 / 0** | 76 / 0 / 4 / 0 | 76 / 0 / 4 / 0 |
| replace jev-1.13 | 57 / 5 / 0 / 18 | **79 / 1 / 0 / 0** | 76 / 2 / 0 / 2 | 76 / 2 / 0 / 2 |
| replace gpt-6-luna | 67 / 9 / 0 / 4 | 73 / 5 / 0 / 2 | 72 / 6 / 0 / 2 | 72 / 6 / 0 / 2 |
| replace gemini-decision-1 | 65 / 5 / 0 / 10 | **79 / 1 / 0 / 0** | 79 / 1 / 0 / 0 | 79 / 1 / 0 / 0 |

### Hard set, gpt-5-nano (80 mentions)

gpt-5-nano alone: 48 correct, 6 wrong, 8 dropped (no valid drug `kind`) and 18
missed, 11 of them in the long sectioned notes. `verify` only checks what the
extractor kept, so it cannot do better than 48: with `section`, 44–48 correct and
no wrong label.

| Setup | line | section | before3 |
|---|---|---|---|
| replace jev-1.13 | 57 / 5 / 0 / 18 | **79 / 1 / 0 / 0** | 76 / 2 / 0 / 2 |
| replace gpt-6-luna | 66 / 8 / 0 / 6 | 74 / 4 / 0 / 2 | 73 / 5 / 0 / 2 |
| replace gemini-decision-1 | 61 / 5 / 0 / 14 | **79 / 1 / 0 / 0** | 79 / 1 / 0 / 0 |

In `replace`, the extractor's prompt has no attributes, and both extractors found
all 80 mentions; the decision model does the rest.

### Basic set (50 mentions)

One-line notes, so the context changes nothing. gpt-5-mini alone: 50 correct. With
gpt-5-mini, `verify` got 50 / 45 / 47 correct (Jev / Luna / Gemini) and no wrong
label, the rest in review; `replace` got 50 / 46 / 48 correct, 0 / 2 / 2 wrong.
gpt-5-nano alone missed 3 ECOG ranges; `replace` with Jev got 48 correct, 1 wrong.
These counts moved by a few labels from the earlier runs, after a new extraction:
run-to-run variation is of that size.

### Cost

| Setup (both sets, 130 mentions) | gpt-5-mini: USD / 1,000 mentions | gpt-5-nano |
|---|---|---|
| extractor alone | 1.82 | 0.73 |
| verify jev-1.13 / gpt-6-luna | 1.85 / 1.89 | 0.75 / 0.78 |
| replace jev-1.13 / gpt-6-luna / gemini* | 1.63 / 1.66 / **1.59** | 0.63 / 0.66 / **0.59** |

\* Without the Gemini decision calls, which have no published price.

**The context costs almost nothing.** Input tokens per decision call:

| Decision model | line | section | before3 |
|---|---|---|---|
| jev-1.13 | 1,007 | 1,009 | 1,012 |
| gpt-6-luna | 878 | 879 | 882 |
| gemini-decision-1 | 773 | 774 | 777 |

The question and its label descriptions are most of a call: a heading adds about
2 tokens, three short lines about 5. 100 Jev calls cost 0.0042 USD with the line
and 0.0043 USD with three lines before. These notes have short lines; with long
lines (a paragraph per line), `lines_before` costs more and sends more of the note
to the provider, while `section` stays one short line.

### Calibration (replace, gpt-5-mini, hard set)

| Decision model | Accuracy, line | Accuracy, section | Brier, line | Brier, section |
|---|---|---|---|---|
| jev-1.13 | 66 / 80 | **79 / 80** | 0.138 | **0.023** |
| gpt-6-luna | 69 / 80 | 74 / 80 | 0.120 | 0.062 |
| gemini-decision-1 | 70 / 80 | **79 / 80** | 0.100 | **0.017** |

In the earlier runs, thresholds above 0.5 mostly sent correct labels to review;
this run did not repeat them.

### Where the labels go wrong

- **Without context**, a drug under "HOME MEDICATIONS:" or "PLAN:" gets `kind`
  "not stated" (p ≈ 0.7), and a disease under "Mother:" gets no relative. In
  `verify` the label goes to review; in `replace` the drug is dropped (its `kind` is
  required) and the relative is left out.
- **The section key works better than the same heading among the lines before.**
  With `before3`, Jev still answers "not stated" for the kind of
  "ipratropium-albuterol nebulizer" (p 0.90) and for the relative of "breast cancer
  at 52", although "GIVEN IN THE ED:" and "Mother:" are among the three lines. Given
  as `section`, it gets both right. Adding the lines to the section does not help.
- **What context does not fix:** Jev labels a fentanyl patch "changed today by
  nursing" `prescribed` (gold: `administered`); Gemini labels "Tobacco: yes, 1/2
  ppd" `current_unknown_frequency` (p 0.44); gpt-6-luna misses the lower bound of
  ECOG ranges.
- **gpt-5-mini's own mistakes** (2): Timolol eye drops to "continue" labelled
  `prescribed` (gold: `taking`), and hydrocortisone cream "to the rash" with no
  route (gold: `topical`). `verify` sends both to review.

## What this says

- **Give the labeller the section heading (`section=True`).** It removes the
  biggest failure of the line-only state for about 2 tokens per call. With it,
  `replace` with Jev or Gemini got 79 of 80 hard labels right with either extractor,
  better than gpt-5-mini alone (78). With Jev and Gemini, `verify` sent 4–5 correct
  labels to review instead of 18–27.
- **`lines_before` is not needed** on these notes: it was as good as or worse than
  the heading alone, and sends more of the note with every call.
- **No mode wins everywhere, so `mode` stays required.**
  - **`verify`** never wrote a wrong label with gpt-5-mini, at about 2 % more cost
    than the extractor alone, and sends disagreements to review. It cannot recover
    the mentions a weak extractor misses or drops.
  - **`replace` with `section=True`**, with Jev or Gemini, was the most accurate and
    the cheapest: gpt-5-nano plus a decision model (0.59–0.63 USD per 1,000
    mentions) beat gpt-5-mini alone (1.82 USD). It writes the decision model's
    answer unchecked, so measure the decision model on your notes first.
- **gpt-6-luna** was the weakest decision model here (ECOG ranges, sections).

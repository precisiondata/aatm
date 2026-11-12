"""Write `cases.csv`: synthetic note snippets with gold attribute labels.

Each row is one mention to label: the note it is in, the mention's text (its first
occurrence in the note), the attribute and the gold label. An empty gold label means
the note does not state the attribute for that mention (e.g. the single ECOG grade of
a range). Notes with several rows are extracted once.

The snippets are written by hand, not taken from patient records. Run this script
after editing the list, from the repository root:

    uv run python benchmarks/attribute_labels/make_cases.py
"""

import csv
from pathlib import Path

SMOKING = "smoking_status"
ECOG = "ecog_performance_status"

# (note, entity, mention, attribute, gold, tag)
CASES = [
    # --- smoking status ---------------------------------------------------------------
    (
        "Social history: Former smoker, quit 2015. Drinks socially.",
        SMOKING,
        "Former smoker",
        "status",
        "former",
        "plain",
    ),
    (
        "SH: smokes 1 ppd x 20 years.",
        SMOKING,
        "smokes 1 ppd",
        "status",
        "current_every_day",
        "abbreviation",
    ),
    ("Never smoker. No alcohol.", SMOKING, "Never smoker", "status", "never", "plain"),
    (
        "Patient denies tobacco use.",
        SMOKING,
        "denies tobacco use",
        "status",
        "not_current",
        "plain",
    ),
    (
        "Occasional cigar on weekends.",
        SMOKING,
        "Occasional cigar on weekends",
        "status",
        "current_some_day",
        "plain",
    ),
    (
        "Tobacco: current smoker.",
        SMOKING,
        "current smoker",
        "status",
        "current_unknown_frequency",
        "plain",
    ),
    (
        "Quit smoking 3 months ago after his MI.",
        SMOKING,
        "Quit smoking 3 months ago",
        "status",
        "former",
        "plain",
    ),
    ("Non-smoker.", SMOKING, "Non-smoker", "status", "not_current", "plain"),
    (
        "Hábitos: ex-tabagista há 10 anos.",
        SMOKING,
        "ex-tabagista",
        "status",
        "former",
        "portuguese",
    ),
    (
        "Tabagista ativo, 20 cigarros/dia.",
        SMOKING,
        "Tabagista ativo, 20 cigarros/dia",
        "status",
        "current_every_day",
        "portuguese",
    ),
    (
        "Nega etilismo. Nunca fumou.",
        SMOKING,
        "Nunca fumou",
        "status",
        "never",
        "portuguese",
    ),
    (
        "Fuma ocasionalmente, em festas.",
        SMOKING,
        "Fuma ocasionalmente",
        "status",
        "current_some_day",
        "portuguese",
    ),
    ("Smoker.", SMOKING, "Smoker", "status", "current_unknown_frequency", "plain"),
    ("Lifelong non-smoker.", SMOKING, "Lifelong non-smoker", "status", "never", "hard"),
    (
        "He smoked in college but has not smoked since.",
        SMOKING,
        "smoked in college but has not smoked since",
        "status",
        "former",
        "hard",
    ),
    (
        "Social smoker, a few cigarettes at parties.",
        SMOKING,
        "Social smoker",
        "status",
        "current_some_day",
        "plain",
    ),
    (
        "Smokes half a pack daily, trying to cut down.",
        SMOKING,
        "Smokes half a pack daily",
        "status",
        "current_every_day",
        "plain",
    ),
    (
        "Former heavy smoker (quit 2001), now vapes daily.",
        SMOKING,
        "Former heavy smoker",
        "status",
        "former",
        "distractor",
    ),
    (
        "Mother smoked for 30 years; the patient never smoked.",
        SMOKING,
        "never smoked",
        "status",
        "never",
        "distractor",
    ),
    (
        "Relapsed to smoking last month, about 5 cigarettes a day.",
        SMOKING,
        "Relapsed to smoking last month",
        "status",
        "current_every_day",
        "hard",
    ),
    (
        "Smokes cigarettes on some days, not daily.",
        SMOKING,
        "Smokes cigarettes on some days",
        "status",
        "current_some_day",
        "plain",
    ),
    (
        "Tobacco use: yes.",
        SMOKING,
        "Tobacco use: yes",
        "status",
        "current_unknown_frequency",
        "form",
    ),
    ("Tobacco use: never.", SMOKING, "Tobacco use: never", "status", "never", "form"),
    (
        "Stopped smoking 2 weeks ago.",
        SMOKING,
        "Stopped smoking 2 weeks ago",
        "status",
        "former",
        "plain",
    ),
    ("No tobacco.", SMOKING, "No tobacco", "status", "not_current", "plain"),
    (
        "Pipe smoker.",
        SMOKING,
        "Pipe smoker",
        "status",
        "current_unknown_frequency",
        "plain",
    ),
    # --- ECOG ---------------------------------------------------------------------------
    ("Assessment: ECOG 1.", ECOG, "ECOG 1", "score", "1", "plain"),
    ("ECOG PS: 0", ECOG, "ECOG PS: 0", "score", "0", "plain"),
    (
        "ECOG performance status is 2.",
        ECOG,
        "ECOG performance status is 2",
        "score",
        "2",
        "plain",
    ),
    ("PS ECOG 3, mostly in bed.", ECOG, "PS ECOG 3", "score", "3", "plain"),
    ("Zubrod score of one.", ECOG, "Zubrod score of one", "score", "1", "words"),
    (
        "WHO performance status 0.",
        ECOG,
        "WHO performance status 0",
        "score",
        "0",
        "plain",
    ),
    ("Fit for chemotherapy, ECOG 1-2.", ECOG, "ECOG 1-2", "score", "", "range"),
    (
        "Fit for chemotherapy, ECOG 1-2.",
        ECOG,
        "ECOG 1-2",
        "score_range_min",
        "1",
        "range",
    ),
    (
        "Fit for chemotherapy, ECOG 1-2.",
        ECOG,
        "ECOG 1-2",
        "score_range_max",
        "2",
        "range",
    ),
    ("ECOG 0 at diagnosis, now ECOG 2.", ECOG, "ECOG 0", "score", "0", "time points"),
    ("ECOG 0 at diagnosis, now ECOG 2.", ECOG, "ECOG 2", "score", "2", "time points"),
    (
        "Performance status (ECOG): 4",
        ECOG,
        "Performance status (ECOG): 4",
        "score",
        "4",
        "plain",
    ),
    ("ECOG grade II.", ECOG, "ECOG grade II", "score", "2", "words"),
    ("ECOG zero, works full time.", ECOG, "ECOG zero", "score", "0", "words"),
    ("ECOG PS 0-1.", ECOG, "ECOG PS 0-1", "score", "", "range"),
    ("ECOG PS 0-1.", ECOG, "ECOG PS 0-1", "score_range_min", "0", "range"),
    ("ECOG PS 0-1.", ECOG, "ECOG PS 0-1", "score_range_max", "1", "range"),
    ("ECOG remains 1 despite fatigue.", ECOG, "ECOG remains 1", "score", "1", "plain"),
    (
        "Eligibility requires ECOG <= 1; the patient's ECOG is 2.",
        ECOG,
        "ECOG is 2",
        "score",
        "2",
        "distractor",
    ),
    ("Estado funcional ECOG 2.", ECOG, "ECOG 2", "score", "2", "portuguese"),
    ("ECOG 4, bedbound.", ECOG, "ECOG 4", "score", "4", "plain"),
    ("ECOG PS = 1.", ECOG, "ECOG PS = 1", "score", "1", "plain"),
    (
        "ECOG performance status of 2-3.",
        ECOG,
        "ECOG performance status of 2-3",
        "score",
        "",
        "range",
    ),
    (
        "ECOG performance status of 2-3.",
        ECOG,
        "ECOG performance status of 2-3",
        "score_range_max",
        "3",
        "range",
    ),
]


def main() -> None:
    notes: dict[str, int] = {}
    path = Path(__file__).with_name("cases.csv")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["note_id", "note", "entity", "mention", "attribute", "gold", "tag"]
        )
        for note, entity, mention, attribute, gold, tag in CASES:
            if mention not in note:
                raise ValueError(f"{mention!r} is not in {note!r}")
            note_id = notes.setdefault(note, len(notes) + 1)
            writer.writerow([note_id, note, entity, mention, attribute, gold, tag])
    print(f"Wrote {len(CASES)} mentions in {len(notes)} notes to {path}")


if __name__ == "__main__":
    main()

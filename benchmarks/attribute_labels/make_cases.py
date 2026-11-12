"""Write `cases.csv`: synthetic note snippets with gold attribute labels.

Each row is one mention to label: the note it is in, the mention's text (its first
occurrence in the note), the attribute and the gold label. An empty gold label means
the note does not state the attribute for that mention (e.g. the single ECOG grade of
a range). Notes with several rows are extracted once.

Two sets: `basic` (short one-line notes, smoking status and ECOG) and `hard` (longer
notes, labels that depend on another line or section, drug `kind` and `route`,
family history `relative`, and harder smoking and ECOG phrasing).

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

DRUG = "drug_name"
FAMILY = "family_history"

# The hard set: longer notes where the label depends on another line or section,
# attributes with many labels (drug route: 13, family relative: 16), the drug `kind`
# (administered / prescribed / taking), and harder smoking and ECOG phrasing.
# (note, [(entity, mention, attribute, gold, tag), ...])
HARD = [
    # --- drugs: kind and route ---------------------------------------------------------
    (
        "HOME MEDICATIONS:\n- lisinopril 10 mg PO daily\n- atorvastatin 40 mg nightly\n"
        "PLAN:\n- start metformin 500 mg twice daily with meals",
        [
            (DRUG, "lisinopril 10 mg", "kind", "taking", "section"),
            (DRUG, "lisinopril 10 mg", "route", "oral", "abbreviation"),
            (DRUG, "atorvastatin 40 mg", "kind", "taking", "section"),
            (DRUG, "atorvastatin 40 mg", "route", "", "not stated"),
            (DRUG, "metformin 500 mg", "kind", "prescribed", "section"),
            (DRUG, "metformin 500 mg", "route", "", "not stated"),
        ],
    ),
    (
        "ED course: ceftriaxone 1 g IV given.\n"
        "Discharge Rx: azithromycin 250 mg by mouth daily for 4 days.",
        [
            (DRUG, "ceftriaxone 1 g", "kind", "administered", "plain"),
            (DRUG, "ceftriaxone 1 g", "route", "intravenous", "abbreviation"),
            (DRUG, "azithromycin 250 mg", "kind", "prescribed", "section"),
            (DRUG, "azithromycin 250 mg", "route", "oral", "plain"),
        ],
    ),
    (
        "Em uso domiciliar: Insulina NPH 10 UI SC à noite.",
        [
            (DRUG, "Insulina NPH", "kind", "taking", "portuguese"),
            (DRUG, "Insulina NPH", "route", "subcutaneous", "portuguese"),
        ],
    ),
    (
        "Albuterol MDI 2 puffs q4h PRN wheeze (home med).",
        [
            (DRUG, "Albuterol", "kind", "taking", "abbreviation"),
            (DRUG, "Albuterol", "route", "inhaled", "abbreviation"),
        ],
    ),
    (
        "Timolol 0.5% eye drops OU BID - continue.",
        [
            (DRUG, "Timolol 0.5%", "kind", "taking", "plain"),
            (DRUG, "Timolol 0.5%", "route", "ophthalmic", "plain"),
        ],
    ),
    (
        "Fentanyl 25 mcg/h patch changed today by nursing.",
        [
            (DRUG, "Fentanyl 25 mcg/h", "kind", "administered", "hard"),
            (DRUG, "Fentanyl 25 mcg/h", "route", "transdermal", "plain"),
        ],
    ),
    (
        "Ondansetron 4 mg SL given for nausea, may repeat once.",
        [
            (DRUG, "Ondansetron 4 mg", "kind", "administered", "plain"),
            (DRUG, "Ondansetron 4 mg", "route", "sublingual", "abbreviation"),
        ],
    ),
    (
        "New: hydrocortisone 1% cream to the rash BID x 7 days.",
        [
            (DRUG, "hydrocortisone 1% cream", "kind", "prescribed", "plain"),
            (DRUG, "hydrocortisone 1% cream", "route", "topical", "hard"),
        ],
    ),
    (
        "Patient stopped taking aspirin 81 mg last month due to bruising.",
        [
            (DRUG, "aspirin 81 mg", "kind", "taking", "hard"),
            (DRUG, "aspirin 81 mg", "route", "", "not stated"),
        ],
    ),
    (
        "Received paclitaxel 175 mg/m2, cycle 3, today; tolerated well.",
        [
            (DRUG, "paclitaxel", "kind", "administered", "plain"),
            (DRUG, "paclitaxel", "route", "", "not stated"),
        ],
    ),
    (
        "Fluticasone nasal spray, 2 sprays each nostril daily (home).",
        [
            (DRUG, "Fluticasone", "kind", "taking", "plain"),
            (DRUG, "Fluticasone", "route", "nasal", "plain"),
        ],
    ),
    (
        "Vitamin B12 1000 mcg IM was given in clinic today.",
        [
            (DRUG, "Vitamin B12 1000 mcg", "kind", "administered", "plain"),
            (DRUG, "Vitamin B12 1000 mcg", "route", "intramuscular", "abbreviation"),
        ],
    ),
    # --- family history: relative ----------------------------------------------------
    (
        "FAMILY HISTORY:\nMother: breast cancer at 45.\nFather: MI at 60, type 2 "
        "diabetes.\nMaternal aunt with ovarian cancer.",
        [
            (FAMILY, "breast cancer", "relative", "mother", "section"),
            # "MI" alone would match inside "FAMILY": anchor it on its line.
            (FAMILY, "MI at 60", "relative", "father", "section"),
            (FAMILY, "type 2 diabetes", "relative", "father", "section"),
            (FAMILY, "ovarian cancer", "relative", "aunt", "plain"),
        ],
    ),
    (
        "Family history of colon cancer.",
        [(FAMILY, "colon cancer", "relative", "", "not stated")],
    ),
    (
        "Paternal grandmother had Alzheimer disease.",
        [(FAMILY, "Alzheimer disease", "relative", "grandmother", "plain")],
    ),
    (
        "His son was diagnosed with type 1 diabetes at age 8.",
        [(FAMILY, "type 1 diabetes", "relative", "son", "plain")],
    ),
    (
        "História familiar: irmã com lúpus.",
        [(FAMILY, "lúpus", "relative", "sister", "portuguese")],
    ),
    (
        "A cousin on the father's side has epilepsy.",
        [(FAMILY, "epilepsy", "relative", "cousin", "distractor")],
    ),
    (
        "Niece with cystic fibrosis.",
        [(FAMILY, "cystic fibrosis", "relative", "other_relative", "hard")],
    ),
    (
        "One of her parents had glaucoma; she is not sure which.",
        [(FAMILY, "glaucoma", "relative", "parent", "hard")],
    ),
    (
        "Twin brother with schizophrenia.",
        [(FAMILY, "schizophrenia", "relative", "brother", "plain")],
    ),
    (
        "Uncle (mother's brother) died of a stroke.",
        [(FAMILY, "stroke", "relative", "uncle", "distractor")],
    ),
    (
        "Children are healthy except one daughter with migraine.",
        [(FAMILY, "migraine", "relative", "daughter", "plain")],
    ),
    # --- smoking status --------------------------------------------------------------
    (
        "SOCIAL HISTORY\nTobacco: Former\nQuit date: 2012\nPacks/day: 1",
        [(SMOKING, "Former", "status", "former", "form")],
    ),
    (
        "Tobacco: yes, 1/2 ppd",
        [(SMOKING, "Tobacco: yes, 1/2 ppd", "status", "current_every_day", "form")],
    ),
    (
        "Smoked 2 packs/day for 30 years until his CABG in 2019.",
        [
            (
                SMOKING,
                "Smoked 2 packs/day for 30 years until his CABG in 2019",
                "status",
                "former",
                "hard",
            )
        ],
    ),
    (
        "Currently smoking again after quitting in 2018.",
        [
            (
                SMOKING,
                "Currently smoking again",
                "status",
                "current_unknown_frequency",
                "hard",
            )
        ],
    ),
    (
        "Has cut down to a cigarette now and then.",
        [
            (
                SMOKING,
                "cut down to a cigarette now and then",
                "status",
                "current_some_day",
                "hard",
            )
        ],
    ),
    (
        "Fumante de 1 maço por dia.",
        [
            (
                SMOKING,
                "Fumante de 1 maço por dia",
                "status",
                "current_every_day",
                "portuguese",
            )
        ],
    ),
    (
        "Uses smokeless tobacco; never smoked cigarettes.",
        [(SMOKING, "never smoked cigarettes", "status", "never", "distractor")],
    ),
    # --- ECOG --------------------------------------------------------------------------
    (
        "PS 1 per ECOG.",
        [(ECOG, "PS 1 per ECOG", "score", "1", "words")],
    ),
    (
        "Performance status: ECOG 1 (KPS 80).",
        [(ECOG, "ECOG 1", "score", "1", "distractor")],
    ),
    (
        "ECOG <= 1 required for the trial; currently ECOG 0.",
        [(ECOG, "ECOG 0", "score", "0", "distractor")],
    ),
    (
        "Zubrod 3.",
        [(ECOG, "Zubrod 3", "score", "3", "words")],
    ),
    (
        "ECOG PS 2 to 3.",
        [
            (ECOG, "ECOG PS 2 to 3", "score", "", "range"),
            (ECOG, "ECOG PS 2 to 3", "score_range_min", "2", "range"),
            (ECOG, "ECOG PS 2 to 3", "score_range_max", "3", "range"),
        ],
    ),
    (
        "ECOG: II-III.",
        [
            (ECOG, "ECOG: II-III", "score", "", "range"),
            (ECOG, "ECOG: II-III", "score_range_min", "2", "range"),
            (ECOG, "ECOG: II-III", "score_range_max", "3", "range"),
        ],
    ),
    (
        "ECOG performance status: 1.\nKarnofsky 70.",
        [(ECOG, "ECOG performance status: 1", "score", "1", "distractor")],
    ),
    # --- long notes whose labels depend on a section heading ---------------------------
    (
        "ADMISSION NOTE\n"
        "HISTORY OF PRESENT ILLNESS:\n"
        "67-year-old man with COPD admitted with worsening dyspnea.\n"
        "HOME MEDICATIONS:\n"
        "- tiotropium 18 mcg inhaled daily\n"
        "- amlodipine 5 mg daily\n"
        "- omeprazole 20 mg daily\n"
        "GIVEN IN THE ED:\n"
        "- methylprednisolone 125 mg IV\n"
        "- ipratropium-albuterol nebulizer\n"
        "PLAN:\n"
        "- prednisone 40 mg daily for 5 days\n"
        "- doxycycline 100 mg BID for 5 days",
        [
            (DRUG, "tiotropium 18 mcg", "kind", "taking", "section"),
            (DRUG, "tiotropium 18 mcg", "route", "inhaled", "plain"),
            (DRUG, "amlodipine 5 mg", "kind", "taking", "section"),
            (DRUG, "amlodipine 5 mg", "route", "", "not stated"),
            (DRUG, "omeprazole 20 mg", "kind", "taking", "section"),
            (DRUG, "omeprazole 20 mg", "route", "", "not stated"),
            (DRUG, "methylprednisolone 125 mg", "kind", "administered", "section"),
            (DRUG, "methylprednisolone 125 mg", "route", "intravenous", "abbreviation"),
            (DRUG, "ipratropium-albuterol", "kind", "administered", "section"),
            (DRUG, "ipratropium-albuterol", "route", "inhaled", "plain"),
            (DRUG, "prednisone 40 mg", "kind", "prescribed", "section"),
            (DRUG, "prednisone 40 mg", "route", "", "not stated"),
            (DRUG, "doxycycline 100 mg", "kind", "prescribed", "section"),
            (DRUG, "doxycycline 100 mg", "route", "", "not stated"),
        ],
    ),
    (
        "FAMILY HISTORY\n"
        "Mother:\n"
        "- hypertension\n"
        "- breast cancer at 52\n"
        "Brother:\n"
        "- type 1 diabetes\n"
        "Maternal grandfather:\n"
        "- colon cancer",
        [
            (FAMILY, "hypertension", "relative", "mother", "section"),
            (FAMILY, "breast cancer", "relative", "mother", "section"),
            (FAMILY, "type 1 diabetes", "relative", "brother", "section"),
            (FAMILY, "colon cancer", "relative", "grandfather", "section"),
        ],
    ),
]


def rows() -> list[tuple[str, str, str, str, str, str, str]]:
    """Every case as (set, note, entity, mention, attribute, gold, tag)."""
    basic = [("basic", *case) for case in CASES]
    hard = [("hard", note, *mention) for note, mentions in HARD for mention in mentions]
    return basic + hard


def main() -> None:
    notes: dict[str, int] = {}
    cases = rows()
    path = Path(__file__).with_name("cases.csv")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["note_id", "set", "note", "entity", "mention", "attribute", "gold", "tag"]
        )
        for case_set, note, entity, mention, attribute, gold, tag in cases:
            if mention not in note:
                raise ValueError(f"{mention!r} is not in {note!r}")
            note_id = notes.setdefault(note, len(notes) + 1)
            writer.writerow(
                [note_id, case_set, note, entity, mention, attribute, gold, tag]
            )
    sets = {s: sum(1 for c in cases if c[0] == s) for s in ("basic", "hard")}
    print(f"Wrote {len(cases)} mentions ({sets}) in {len(notes)} notes to {path}")


if __name__ == "__main__":
    main()

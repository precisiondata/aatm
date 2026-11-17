"""Download the ACI-bench notes and their ICD-10-CM annotations, and merge them.

The annotations are those of Yuan et al., *Toward Reliable Clinical Coding with
Language Models: Verification and Lightweight Adaptation* (EMNLP 2025, Industry
Track): two experts coded every encounter of
[ACI-bench](https://github.com/wyim/aci-bench) (doctor-patient dialogues and the
clinical notes written from them) with ICD-10-CM.

Both repositories are pinned to a commit, so every run benchmarks the same data. The
data is not committed to this repository: the annotations are CC BY-NC 4.0 and
ACI-bench is CC BY 4.0. Run from the repository root:

```bash
uv run python benchmarks/icd10cm_coding/prepare_data.py
```

Writes ``data/{train,valid,test}.jsonl`` next to this script (67, 20 and 120
encounters), one encounter per line: ``encounter_id``, ``dataset`` (the ACI-bench
source file), ``dialogue``, ``note`` and ``icd10_codes`` (``code`` and
``description``), as the authors' ``merge_aci_annotations.py`` does.
"""

import argparse
import csv
import io
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"

ACI_BENCH = (
    "https://raw.githubusercontent.com/wyim/aci-bench/"
    "b909b2bb9cf1d19de08df15cddde7bd0179665e4/data/challenge_data/"
)
ANNOTATIONS = (
    "https://raw.githubusercontent.com/amazon-science/"
    "toward-clinical-coding-verification-adaptation/"
    "311f4afea6e6697f748ad23b62e8a0d70772736b/annotation/"
)

# split -> (ACI-bench files, annotation file). The three test sets share one file.
SPLITS = {
    "train": (["train.csv"], "code_train.csv"),
    "valid": (["valid.csv"], "code_valid.csv"),
    "test": (
        [
            "clinicalnlp_taskB_test1.csv",
            "clinicalnlp_taskC_test2.csv",
            "clef_taskC_test3.csv",
        ],
        "code_test.csv",
    ),
}


def fetch_csv(url: str) -> list[dict[str, str]]:
    with urllib.request.urlopen(url, timeout=60) as response:
        text = response.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def load_codes(annotation_file: str) -> dict[str, list[dict[str, str]]]:
    codes: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in fetch_csv(ANNOTATIONS + annotation_file):
        codes[row["Document ID"]].append(
            {"code": row["ICD10"].strip(), "description": row["Description"].strip()}
        )
    return codes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=Path, default=DATA)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split, (aci_files, annotation_file) in SPLITS.items():
        codes = load_codes(annotation_file)
        records = []
        for aci_file in aci_files:
            for row in fetch_csv(ACI_BENCH + aci_file):
                records.append(
                    {
                        "encounter_id": row["encounter_id"],
                        "dataset": row["dataset"],
                        "dialogue": row["dialogue"],
                        "note": row["note"],
                        "icd10_codes": codes.get(row["encounter_id"], []),
                    }
                )
        ids = [r["encounter_id"] for r in records]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{split}: duplicate encounter ids")
        unmatched = set(codes) - set(ids)
        if unmatched:
            raise ValueError(
                f"{split}: annotations without a note: {sorted(unmatched)}"
            )

        path = args.output_dir / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_codes = sum(len(r["icd10_codes"]) for r in records)
        coded = sum(bool(r["icd10_codes"]) for r in records)
        print(
            f"{split}: {len(records)} encounters ({coded} with codes), "
            f"{n_codes} codes -> {path}"
        )


if __name__ == "__main__":
    main()

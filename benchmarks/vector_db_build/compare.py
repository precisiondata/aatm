"""Compare two `run_build.py` results: timings, memory and search agreement.

Usage:

    python benchmarks/vector_db_build/compare.py before.json after.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

STAGES = (
    "embedding_function_call",
    "calibrate_batch_size",
    "calibrate_max_batch_tokens",
    "chromadb_add (writer process)",
    "wait_for_writer",
    "wait_for_final_writes",
    "wait_for_prepared_chunk",
    "chromadb_get_existing_ids",
    "build_expression_metadata",
    "embedding_function_init",
)


def search_agreement(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    """Top-1 agreement and mean overlap of the top-10 ids, over shared queries."""
    queries = sorted(set(a) & set(b))
    top1 = sum(a[q]["ids"][0] == b[q]["ids"][0] for q in queries)
    overlap = sum(
        len(set(a[q]["ids"]) & set(b[q]["ids"])) / len(a[q]["ids"]) for q in queries
    )
    return {
        "queries": len(queries),
        "top1_agreement": top1 / len(queries),
        "mean_top10_overlap": overlap / len(queries),
    }


def main() -> None:
    before, after = (
        json.loads(Path(p).read_text(encoding="utf-8")) for p in sys.argv[1:3]
    )

    def row(name: str, x: Any, y: Any, ratio: bool = False) -> str:
        change = ""
        if ratio and isinstance(x, (int, float)) and isinstance(y, (int, float)) and y:
            change = f"{x / y:.2f}×"
        return f"| {name} | {x} | {y} | {change} |"

    print(f"| | {before['label']} | {after['label']} | before / after |")
    print("|---|---|---|---|")
    print(row("Build time (s)", before["build_seconds"], after["build_seconds"], True))
    print(row("Rows", before["rows"], after["rows"]))
    print(row("Rows/s", before["rows_per_second"], after["rows_per_second"]))
    print(row("Batch setting", before["batch_setting"][0], after["batch_setting"][0]))
    for key in ("dedicated_peak_gib", "shared_baseline_gib", "shared_peak_gib"):
        print(row(f"GPU {key}", before["gpu_memory"][key], after["gpu_memory"][key]))
    print(row("Database size (GiB)", before["db_size_gib"], after["db_size_gib"]))
    for stage in STAGES:
        x = before["profile_totals_seconds"].get(stage, "–")
        y = after["profile_totals_seconds"].get(stage, "–")
        if x != "–" or y != "–":
            print(row(f"`{stage}` (s)", x, y, True))
    print(
        row(
            "Warnings",
            len(before["warnings_during_build"]),
            len(after["warnings_during_build"]),
        )
    )
    print()
    print(
        "Search agreement:",
        json.dumps(search_agreement(before["search"], after["search"])),
    )
    if "repair" in after:
        print("Repair:", json.dumps(after["repair"]))


if __name__ == "__main__":
    main()

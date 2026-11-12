"""Compare attribute labels from the extractor alone, `verify` and `replace`.

Issue #81: should a decision model label closed-set attributes (smoking status,
ECOG grade), and in which mode? Every mention of `cases.csv` is labelled by:

- `extractor`: the extractor's own labels (`SimpleLLMExtractor`, no labeller);
- `verify`: the same extraction, with each label checked by the decision model
  (`DecisionAttributeLabeller(mode="verify")`); a label is kept when both agree;
- `replace`: an extraction whose prompt leaves the attributes out, labelled by the
  decision model (`mode="replace"`).

Each mention ends as `correct`, `wrong`, `review` (the label was withheld for a
person to check), `dropped` (found, but rejected because a required attribute got
no label) or `missed` (the extractor did not find it). `verify` and `replace`
are also scored at several `min_probability` thresholds, from the same calls.

Run from the repository root (keys in `.env`):

    uv run python benchmarks/attribute_labels/run_benchmark.py \\
        --models openai/gpt-5-mini --decision-models openrouter/typesafe/jev-1.13

Extractions are stored in `results/` and reused by later runs (`--fresh` redoes
them), so trying another decision model costs only its own calls.
"""

import argparse
import asyncio
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import dotenv

from aatm.data_models.extractors import (
    EntityDefinitions,
    EntityValidationResult,
    segment_text,
)
from aatm.extractors import DecisionAttributeLabeller, SimpleLLMExtractor
from aatm.extractors.attribute_labeller import AttributeDecision
from aatm.llm.usage import TokenUsage, track_usage
from aatm.omop.builder import _load_definitions

HERE = Path(__file__).parent
DEFINITIONS = [
    "smoking_status",
    "ecog_performance_status",
    "drug_exposure",
    "family_history",
]
SETS = ["basic", "hard"]
#: USD per million input / output tokens, for models whose provider reports no cost.
PRICES = {
    "openai/gpt-5-mini": (0.25, 2.0),
    "openai/gpt-5-nano": (0.05, 0.40),
    "openai/gpt-6-luna": (0.10, 0.0),
    "openrouter/typesafe/jev-1.13": (0.042, 0.0),
}
OUTCOMES = ["correct", "wrong", "review", "dropped", "missed"]


def slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_")


def load_cases() -> list[dict[str, str]]:
    with (HERE / "cases.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def notes_of(cases: list[dict[str, str]]) -> dict[str, str]:
    return {c["note_id"]: c["note"] for c in cases}


# --- extraction ------------------------------------------------------------------------


async def extract(
    model: str,
    definitions: EntityDefinitions,
    notes: dict[str, str],
    path: Path,
    fresh: bool,
) -> tuple[dict[str, EntityValidationResult], dict[str, Any]]:
    """Extract every note, or load the extraction stored at `path` when it was made
    with the same model, definitions and notes."""
    key = hashlib.sha256(
        json.dumps(
            [model, definitions.model_dump(mode="json", exclude={"omop"}), notes],
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if path.exists() and not fresh:
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("key") == key:
            results = {
                k: EntityValidationResult.model_validate(v)
                for k, v in stored["results"].items()
            }
            return results, stored["stats"]
    extractor = SimpleLLMExtractor(
        model_id=model,
        entity_definitions=definitions,
        on_error="record",
        max_concurrency=8,
    )
    started = time.perf_counter()
    with track_usage() as usage:
        outputs = await extractor(list(notes.values()))
    stats = usage_stats(usage, time.perf_counter() - started)
    results = dict(zip(notes, outputs))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model": model,
                "key": key,
                "stats": stats,
                "results": {k: r.model_dump(mode="json") for k, r in results.items()},
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return results, stats


def usage_stats(usage: TokenUsage, seconds: float) -> dict[str, Any]:
    return {
        "calls": usage.calls,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost": usage.cost(PRICES),
        "seconds": seconds,
    }


# --- labelling ---------------------------------------------------------------------------


async def label(
    labeller: DecisionAttributeLabeller,
    results: dict[str, EntityValidationResult],
    notes: dict[str, str],
    definitions: EntityDefinitions,
) -> tuple[
    dict[str, EntityValidationResult],
    dict[tuple[str, str, str], dict[str, AttributeDecision]],
    dict[str, Any],
]:
    """Label every result; also return each mention's decisions and the usage."""
    decisions: dict[tuple[str, str, str], dict[str, AttributeDecision]] = {}
    decide = labeller.decide

    async def recording(entity, attributes, mention, line):  # type: ignore[no-untyped-def]
        answers = await decide(entity, attributes, mention, line)
        decisions[(entity.key, mention, line)] = answers
        return answers

    labeller.decide = recording  # type: ignore[method-assign]
    started = time.perf_counter()
    with track_usage() as usage:
        labelled = await asyncio.gather(
            *(
                labeller.label(result, segment_text(notes[k]), definitions)
                for k, result in results.items()
            )
        )
    stats = usage_stats(usage, time.perf_counter() - started)
    return dict(zip(results, labelled)), decisions, stats


# --- scoring -------------------------------------------------------------------------------


def overlaps(start: int, end: int, other_start: int, other_end: int) -> bool:
    return start < other_end and other_start < end


def outcome(
    case: dict[str, str], result: EntityValidationResult
) -> tuple[str, Optional[str], Optional[Any]]:
    """The case's outcome, the label written, and the mention (to find its decision)."""
    start = case["note"].index(case["mention"])
    end = start + len(case["mention"])
    gold = case["gold"] or None
    for entity in result.validated_entities:
        if entity.key == case["entity"] and overlaps(
            start, end, entity.start, entity.end
        ):
            reviewed = any(
                r.entity_id == entity.id and r.attribute_key == case["attribute"]
                for r in result.attribute_reviews
            )
            value = entity.get_attribute(case["attribute"])
            if reviewed:
                return "review", value, entity
            return ("correct" if value == gold else "wrong"), value, entity
    for rejected in result.unmatched_entities:
        if (
            rejected.key == case["entity"]
            and rejected.start is not None
            and rejected.end is not None
            and overlaps(start, end, rejected.start, rejected.end)
        ):
            if "required_attribute_under_review" in rejected.reasons:
                return "review", None, rejected
            if {
                "required_attribute_missing",
                "invalid_required_attribute",
            } & set(rejected.reasons):
                # Found, but rejected for want of a required label.
                return "dropped", None, rejected
    return "missed", None, None


def segment_line(note: str, line_id: str) -> str:
    return next((s.text for s in segment_text(note) if s.id == line_id), "")


def score(
    cases: list[dict[str, str]],
    results: dict[str, EntityValidationResult],
    decisions: Optional[dict[tuple[str, str, str], dict[str, AttributeDecision]]],
    threshold: Optional[float],
) -> list[dict[str, Any]]:
    """One row per case. With a threshold, a label whose decision probability is
    below it goes to review, as `min_probability` would send it."""
    rows = []
    for case in cases:
        result = results[case["note_id"]]
        kind, value, entity = outcome(case, result)
        decision: Optional[AttributeDecision] = None
        if decisions is not None and entity is not None:
            key = (entity.key, entity.text, segment_line(case["note"], entity.line))
            decision = decisions.get(key, {}).get(case["attribute"])
        if (
            threshold is not None
            and kind in ("correct", "wrong")
            and (
                decision is None
                or decision.probability is None
                or decision.probability < threshold
            )
        ):
            kind = "review"
        rows.append(
            {
                "note_id": case["note_id"],
                "mention": case["mention"],
                "attribute": case["attribute"],
                "set": case["set"],
                "tag": case["tag"],
                "gold": case["gold"],
                "outcome": kind,
                "written": value or "",
                "decision": "" if decision is None else (decision.label or ""),
                "probability": ""
                if decision is None or decision.probability is None
                else round(decision.probability, 4),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(r["outcome"] for r in rows)
    written = counts["correct"] + counts["wrong"]
    return {
        **{o: counts[o] for o in OUTCOMES},
        "mentions": len(rows),
        "precision": counts["correct"] / written if written else None,
    }


def calibration(
    cases: list[dict[str, str]],
    results: dict[str, EntityValidationResult],
    decisions: dict[tuple[str, str, str], dict[str, AttributeDecision]],
) -> dict[str, Any]:
    """How well the decision model's probabilities match its accuracy (replace)."""
    pairs = []
    for case in cases:
        _, _, entity = outcome(case, results[case["note_id"]])
        if entity is None:
            continue
        key = (entity.key, entity.text, segment_line(case["note"], entity.line))
        decision = decisions.get(key, {}).get(case["attribute"])
        if decision is None or decision.probability is None:
            continue
        right = (decision.label or "") == case["gold"]
        pairs.append((decision.probability, right))
    if not pairs:
        return {"answers": 0}
    right = [p for p, ok in pairs if ok]
    wrong = [p for p, ok in pairs if not ok]
    return {
        "answers": len(pairs),
        "accuracy": len(right) / len(pairs),
        "brier": sum((p - ok) ** 2 for p, ok in pairs) / len(pairs),
        "mean_probability_right": sum(right) / len(right) if right else None,
        "mean_probability_wrong": sum(wrong) / len(wrong) if wrong else None,
    }


# --- main ------------------------------------------------------------------------------------


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def table(summary: dict[str, Any]) -> str:
    lines = [
        "| Setup | Correct | Wrong | Review | Dropped | Missed "
        "| Precision of written labels |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in summary.items():
        lines.append(
            f"| {name} | {s['correct']} | {s['wrong']} | {s['review']} "
            f"| {s['dropped']} | {s['missed']} | {fmt(s['precision'])} |"
        )
    return "\n".join(lines)


def cost_table(stats: dict[str, Any], mentions: int) -> str:
    lines = [
        "| Setup | USD / 1,000 mentions | Seconds | Decision calls |",
        "|---|---|---|---|",
    ]
    for name, s in stats.items():
        cost = s.get("cost")
        lines.append(
            f"| {name} | {fmt(None if cost is None else cost / mentions * 1000, 3)} "
            f"| {fmt(s.get('seconds'), 1)} | {s.get('decision_calls', 0)} |"
        )
    return "\n".join(lines)


async def run_model(
    args: argparse.Namespace,
    model: str,
    cases: list[dict[str, str]],
    notes: dict[str, str],
    definitions: EntityDefinitions,
) -> None:
    out = HERE / "results"
    full, full_stats = await extract(
        model,
        definitions,
        notes,
        out / f"extraction__{slug(model)}.json",
        args.fresh,
    )
    # The extraction of replace mode: the labelled attributes left out of the prompt.
    any_labeller = DecisionAttributeLabeller(
        args.decision_models[0], mode="replace", max_concurrency=args.concurrency
    )
    stripped, stripped_stats = await extract(
        model,
        any_labeller.prompt_definitions(definitions),
        notes,
        out / f"extraction_without_attributes__{slug(model)}.json",
        args.fresh,
    )

    per_case = {"extractor": score(cases, full, None, None)}
    stats: dict[str, Any] = {"extractor": full_stats}
    report: dict[str, Any] = {"model": model, "decision_models": {}}

    for decision_model in args.decision_models:
        name = decision_model.split("/")[-1]
        for mode, extracted, extraction_stats in (
            ("verify", full, full_stats),
            ("replace", stripped, stripped_stats),
        ):
            labeller = DecisionAttributeLabeller(
                decision_model, mode=mode, max_concurrency=args.concurrency
            )
            labelled, decisions, label_stats = await label(
                labeller, extracted, notes, definitions
            )
            stats[f"{mode} {name}"] = {
                **{
                    k: (extraction_stats.get(k) or 0) + (label_stats.get(k) or 0)
                    for k in ("cost", "seconds")
                },
                "decision_calls": label_stats["calls"],
            }
            for threshold in [None, *args.thresholds]:
                setup = f"{mode} {name}" + (f" @{threshold}" if threshold else "")
                per_case[setup] = score(cases, labelled, decisions, threshold)
            if mode == "replace":
                report["decision_models"][decision_model] = {
                    "calibration": {
                        case_set: calibration(
                            [c for c in cases if c["set"] == case_set],
                            labelled,
                            decisions,
                        )
                        for case_set in SETS
                    },
                    "usage": label_stats,
                }

    summary = {
        case_set: {
            setup: summarize([r for r in rows if r["set"] == case_set])
            for setup, rows in per_case.items()
        }
        for case_set in SETS
    }
    report["summary"] = summary
    report["stats"] = stats

    name = (
        slug(model)
        + "__"
        + "+".join(slug(m.split("/")[-1]) for m in args.decision_models)
    )
    with (out / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["setup", *per_case["extractor"][0]])
        writer.writeheader()
        for setup, rows in per_case.items():
            for row in rows:
                writer.writerow({"setup": setup, **row})
    (out / f"{name}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"\n## {model}")
    for case_set in SETS:
        count = sum(1 for c in cases if c["set"] == case_set)
        print(f"\n### {case_set} ({count} mentions)\n")
        print(table(summary[case_set]))
    print(f"\n### Cost, both sets ({len(cases)} mentions)\n")
    print(cost_table(stats, len(cases)))
    print()
    for decision_model, data in report["decision_models"].items():
        print(decision_model, json.dumps(data["calibration"]))
    by_tag: dict[str, Counter[str]] = defaultdict(Counter)
    for setup, rows in per_case.items():
        if "@" in setup:
            continue
        for row in rows:
            if row["outcome"] != "correct":
                by_tag[setup][f"{row['set']}/{row['tag']}:{row['outcome']}"] += 1
    print()
    for setup, counter in by_tag.items():
        print(setup, dict(counter))


async def run(args: argparse.Namespace) -> None:
    cases = load_cases()
    if args.sets:
        cases = [c for c in cases if c["set"] in args.sets]
    notes = notes_of(cases)
    definitions = _load_definitions(DEFINITIONS)
    print(f"{len(cases)} mentions in {len(notes)} notes")
    for model in args.models:
        await run_model(args, model, cases, notes, definitions)


def main() -> None:
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=["openai/gpt-5-mini"])
    parser.add_argument(
        "--decision-models", nargs="+", default=["openrouter/typesafe/jev-1.13"]
    )
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.5, 0.8, 0.9])
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--sets", nargs="*", choices=SETS, help="default: all")
    parser.add_argument("--fresh", action="store_true", help="extract the notes again")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()

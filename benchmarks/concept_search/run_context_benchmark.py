"""Measure how much a text's context changes the concept search agent's choice.

`context_cases.csv` holds lab test names that each appear in several contexts: the same
text ("Hb", "glucose", "creatinine") in different sentences, with different values and
units. In some groups the right concept is the same in every context; in others the
context changes it ("Hb 7.8 %" is HbA1c, "glucose" in a CSF panel is CSF glucose).

The agent codes every case under three context variants
([issue #58](https://github.com/precisiondata/aatm/issues/58)):

- ``full``: the text, its related values (value, unit, operator), the sentence and the
  target column, as `aatm.omop.AgenticCoder` sends them;
- ``unit``: the text, the target, the attributes and the unit, without the value or
  the sentence;
- ``text``: the text only.

Each variant runs ``--repeats`` times, every case searched separately (no
deduplication), to separate the effect of context from run-to-run variation. Run from
the repository root, after ``aatm init``:

```bash
uv run python benchmarks/concept_search/run_context_benchmark.py --retriever text-embedding-3-small --model openai/gpt-5-mini
```

Writes one row per case, variant and repeat to
``results/context__<retriever>__<model>.csv`` (``--variants`` and ``--repeats`` rerun a
subset; earlier rows of other variants and repeats are kept), then prints accuracy,
consistency, run-to-run agreement, which answer is right when variants disagree, and
the accuracy a coarser cache key would give. ``--summary-only`` prints them from the
saved results.
"""

import argparse
import asyncio
import itertools
import time
from pathlib import Path

# pandas before the retrievers (torch): on some Windows setups, loading pyarrow after
# torch crashes the interpreter.
import pandas as pd

from aatm.concept_search import CodingContext, ConceptSearchAgent, ConceptSearchTools

HERE = Path(__file__).parent
DOMAIN, VOCABULARIES = "Measurement", ["LOINC"]
TARGET = "measurement.measurement_concept_id"
VARIANTS = ["full", "unit", "text"]


def load_cases() -> pd.DataFrame:
    cases = pd.read_csv(HERE / "context_cases.csv", dtype=str, keep_default_na=False)
    cases["accepted"] = [
        {gold, *filter(None, also.split(";"))}
        for gold, also in zip(cases.gold, cases.also_accept)
    ]
    # A group's gold changes with context when its cases don't share an accepted code.
    cases["context_matters"] = cases.group.map(
        cases.groupby("group").accepted.agg(lambda s: not set.intersection(*s))
    )
    return cases


def context(case: pd.Series, variant: str) -> CodingContext:
    """The context the agent gets for a case, with the keys the lab definitions use."""
    if variant == "text":
        return CodingContext(text=case.text)
    related = {"operator": case.operator} if case.operator else {}
    if variant == "full" and case.value:
        related["lab_value"] = case.value
    if case.unit:
        related["lab_unit"] = case.unit
    return CodingContext(
        text=case.text,
        related=related,
        sentence=case.sentence if variant == "full" else None,
        target=TARGET,
    )


def cache_key(case: pd.Series, variant: str) -> str:
    return context(case, variant).cache_key()


async def run_variant(
    agent: ConceptSearchAgent, cases: pd.DataFrame, variant: str, concurrency: int
) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(case: pd.Series) -> dict:
        async with semaphore:
            r = await agent.search(context(case, variant), DOMAIN, VOCABULARIES)
        return {
            "concept_code": r.concept.concept_code if r.concept else None,
            "concept_name": r.concept.concept_name if r.concept else None,
            "confidence": r.confidence,
            "llm_calls": r.llm_calls,
            "tool_calls": r.tool_calls,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "seconds": r.seconds,
            "budget_exhausted": r.budget_exhausted,
            "rationale": r.rationale,
            "error": r.error,
        }

    return await asyncio.gather(*(one(case) for _, case in cases.iterrows()))


def agreement(codes: pd.Series) -> float:
    """Share of pairs of answers that are the same concept."""
    pairs = list(itertools.combinations(codes.fillna("none"), 2))
    return sum(a == b for a, b in pairs) / len(pairs) if pairs else float("nan")


def summarize(df: pd.DataFrame, cases: pd.DataFrame) -> None:
    variants = [v for v in VARIANTS if v in set(df.variant)]
    kinds = {False: "same concept", True: "context matters"}

    # Accuracy: mean over repeats, with the range across repeats.
    per_repeat = df.groupby(["variant", "repeat"]).correct.mean()
    by_kind = df.groupby(["variant", "context_matters"]).correct.mean().unstack()
    acc = pd.DataFrame(
        {
            "accuracy": per_repeat.groupby("variant").mean(),
            "min": per_repeat.groupby("variant").min(),
            "max": per_repeat.groupby("variant").max(),
            **{
                f"accuracy ({kinds[k]})": by_kind[k]
                for k in by_kind.columns
            },
            "high confidence": df.groupby("variant").confidence.apply(
                lambda s: (s == "high").mean()
            ),
            "input tokens": df.groupby("variant").input_tokens.mean(),
            "llm calls": df.groupby("variant").llm_calls.mean(),
        }
    ).loc[variants]
    print(f"\n## Accuracy ({df.repeat.nunique()} repeats, {len(cases)} cases)\n")
    print(acc.round(3).to_markdown())

    # Consistency: agreement between contexts of the same text, within a repeat.
    cons = (
        df.groupby(["variant", "repeat", "group", "context_matters"])
        .concept_code.apply(agreement)
        .groupby(["variant", "context_matters"])
        .mean()
        .unstack()
        .rename(columns=kinds)
        .loc[variants]
    )
    gold = cases.groupby(["group", "context_matters"]).gold.apply(agreement)
    cons.loc["gold"] = gold.groupby("context_matters").mean().rename(kinds)
    print("\n## Consistency across contexts of a text (pairwise agreement)\n")
    print(cons.round(3).to_markdown())

    # Run-to-run: agreement between repeats of the same case and variant.
    if df.repeat.nunique() > 1:
        runs = (
            df.groupby(["variant", "id"])
            .agg(
                agreement=("concept_code", agreement),
                stable=("concept_code", lambda s: s.fillna("none").nunique() == 1),
            )
            .groupby("variant")
            .mean()
            .loc[variants]
        )
        runs.columns = ["pairwise agreement", "cases always the same"]
        print("\n## Run-to-run agreement (same case and variant)\n")
        print(runs.round(3).to_markdown())

    # Disagreements: same case and repeat, two variants, different answers.
    wide = df.pivot_table(
        index=["id", "repeat"], columns="variant", values="correct", aggfunc="first"
    )
    codes = df.pivot_table(
        index=["id", "repeat"],
        columns="variant",
        values="concept_code",
        aggfunc="first",
    ).fillna("none")
    rows = []
    for a, b in itertools.combinations(variants, 2):
        differ = codes[a] != codes[b]
        rows.append(
            {
                "pair": f"{a} vs {b}",
                "disagreements": int(differ.sum()),
                f"only {a} right": int((differ & wide[a] & ~wide[b]).sum()),
                f"only {b} right": int((differ & ~wide[a] & wide[b]).sum()),
                "both wrong": int((differ & ~wide[a] & ~wide[b]).sum()),
            }
        )
    print("\n## When two variants disagree on a case, which is right\n")
    print(pd.DataFrame(rows).fillna(0).to_markdown(index=False))

    # Coarse cache keys: the full-context answer of one case reused for every case with
    # the same key, averaged over which case was coded first.
    full = df[df.variant == "full"].merge(
        cases[["id", "accepted"]], on="id", suffixes=("", "_case")
    )
    print("\n## Accuracy with a coarser cache key (full-context answers reused)\n")
    keyed = []
    for name, variant in [
        ("full context (exact)", "full"),
        ("text, target, attributes, unit", "unit"),
        ("text", "text"),
    ]:
        keys = cases.set_index("id").apply(lambda c: cache_key(c, variant), axis=1)
        full["key"] = full.id.map(keys)
        correct = 0.0
        for _, group in full.groupby(["repeat", "key"]):
            # Each member coded first with probability 1/n; its answer serves all n.
            for code in group.concept_code:
                correct += sum(code in a for a in group.accepted) / len(group)
        keyed.append(
            {
                "cache key": name,
                "distinct keys": keys.nunique(),
                "accuracy": correct / len(full),
            }
        )
    print(pd.DataFrame(keyed).round(3).to_markdown(index=False))

    wrong = df[~df.correct]
    print("\n## Errors (case, variant: wrong answers / repeats)\n")
    errors = (
        wrong.groupby(["id", "text", "gold", "variant"])
        .agg(times=("correct", "size"), answers=("concept_code", lambda s: ", ".join(sorted(set(s.fillna("none"))))))
        .reset_index()
    )
    errors["sentence"] = errors.id.map(cases.set_index("id").sentence)
    print(errors.to_markdown(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--retriever", default="text-embedding-3-small")
    parser.add_argument("--model", default="openai/gpt-5-mini")
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    parser.add_argument("--repeats", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--max-tool-calls", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--vector-db",
        default=None,
        help="'local', 'cloud' or 'pinecone'. Defaults to AATM_VECTOR_DB, else local.",
    )
    args = parser.parse_args()

    cases = load_cases()
    out = (
        HERE
        / "results"
        / f"context__{args.retriever}__{args.model.replace('/', '_')}.csv"
    )
    previous = (
        pd.read_csv(out, dtype={"id": str, "concept_code": str})
        if out.exists()
        else pd.DataFrame()
    )

    rows = []
    if not args.summary_only:
        from aatm.registries.retrievers import load_retriever
        from aatm.vector_db import resolve_vector_db_mode

        retriever = load_retriever(
            args.retriever, vector_db=resolve_vector_db_mode(args.vector_db)
        )
        agent = ConceptSearchAgent(
            args.model,
            ConceptSearchTools(retriever, vocabulary=".aatm/omop.db"),
            max_tool_calls=args.max_tool_calls,
        )
        for repeat, variant in itertools.product(args.repeats, args.variants):
            print(f"Repeat {repeat}, variant {variant}: {len(cases)} cases...")
            wall = time.perf_counter()
            found = asyncio.run(run_variant(agent, cases, variant, args.concurrency))
            print(f"  {time.perf_counter() - wall:.0f} s")
            for (_, case), result in zip(cases.iterrows(), found):
                rows.append(
                    {
                        "variant": variant,
                        "repeat": repeat,
                        "id": case.id,
                        "group": case.group,
                        "text": case.text,
                        "gold": case.gold,
                        **result,
                    }
                )
            # Save after each run, so a stopped benchmark keeps what it finished.
            done = pd.DataFrame(rows)
            if not previous.empty:
                rerun = previous.set_index(["variant", "repeat"]).index.isin(
                    done.set_index(["variant", "repeat"]).index
                )
                done = pd.concat([previous[~rerun], done], ignore_index=True)
            out.parent.mkdir(exist_ok=True)
            done.to_csv(out, index=False)
    df = pd.read_csv(out, dtype={"id": str, "concept_code": str})

    df = df.merge(cases[["id", "accepted", "context_matters"]], on="id")
    df["correct"] = [c in a for c, a in zip(df.concept_code, df.accepted)]
    summarize(df, cases)
    print(f"\nPer-case results: {out}")


if __name__ == "__main__":
    main()

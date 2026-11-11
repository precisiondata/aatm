"""Compare concept selection strategies on lab results with gold LOINC codes.

Methods:

- ``first-result``: the nearest of the retrieved candidates (`FirstResultSelector`).
- ``llm-selector``: an LLM picks among the top 30 candidates (`OpenAILLMSelector`).
- ``agent``: the concept search agent, given the text with its value, unit and
  sentence (`aatm.concept_search.ConceptSearchAgent`).
- ``agent-no-context``: the same agent, given the text only.

Every method searches standard LOINC concepts of the Measurement domain. Run from the
repository root, after ``aatm init`` (vocabulary database and the embedding index):

```bash
uv run python benchmarks/concept_search/run_benchmark.py --retriever text-embedding-3-small --model openai/gpt-5-mini
```

Writes one row per text and method to ``results/<retriever>__<model>.csv`` and prints a
summary table (accuracy, LLM calls, tokens and seconds per text). ``--vector-db`` picks
the vector database (``local``, ``cloud`` or ``pinecone``, default ``AATM_VECTOR_DB``);
results from Chroma Cloud or Pinecone go to ``results/<retriever>__<model>__<vector-db>.csv``.
"""

import argparse
import asyncio
import time
from pathlib import Path

# pandas before the retrievers (torch): on some Windows setups, loading pyarrow after
# torch crashes the interpreter.
import pandas as pd

from aatm.concept_search import CodingContext, ConceptSearchAgent, ConceptSearchTools
from aatm.omop.coding import retriever_filter
from aatm.registries.retrievers import load_retriever
from aatm.selectors import FirstResultSelector, OpenAILLMSelector
from aatm.vector_db import resolve_vector_db_mode

HERE = Path(__file__).parent
DOMAIN, VOCABULARIES = "Measurement", ["LOINC"]
WHERE = retriever_filter(DOMAIN, VOCABULARIES)
METHODS = ["first-result", "llm-selector", "agent", "agent-no-context"]


def load_cases() -> pd.DataFrame:
    cases = pd.read_csv(HERE / "lab_results.csv", dtype=str, keep_default_na=False)
    cases["accepted"] = [
        {gold, *filter(None, also.split(";"))}
        for gold, also in zip(cases.gold, cases.also_accept)
    ]
    return cases


def context(row: pd.Series, with_context: bool) -> CodingContext:
    if not with_context:
        return CodingContext(text=row.text)
    related = {k: row[k] for k in ("value", "unit") if row[k]}
    return CodingContext(
        text=row.text,
        related=related,
        sentence=row.sentence,
        target="measurement.measurement_concept_id",
    )


def run_selector(method: str, cases: pd.DataFrame, retriever, model: str) -> list[dict]:
    selector = (
        FirstResultSelector()
        if method == "first-result"
        else OpenAILLMSelector(model.split("/", 1)[1])
    )
    rows = []
    for _, case in cases.iterrows():
        start = time.perf_counter()
        retrieved = retriever.retrieve([case.text], where=WHERE, top_k=30)
        (selected,) = selector.select(retrieved).results
        rows.append(
            {
                "concept_code": selected.std_vocabulary_code,
                "concept_name": selected.std_concept_name,
                "score": 1 - selected.distance
                if selected.distance is not None
                else None,
                "llm_calls": 0 if method == "first-result" else 1,
                "input_tokens": None,  # not reported by the legacy selectors
                "output_tokens": None,
                "seconds": time.perf_counter() - start,
            }
        )
    return rows


async def run_agent(
    cases: pd.DataFrame, retriever, model: str, with_context: bool, max_tool_calls: int
) -> list[dict]:
    agent = ConceptSearchAgent(
        model,
        ConceptSearchTools(retriever, vocabulary=".aatm/omop.db"),
        max_tool_calls=max_tool_calls,
    )
    results = await agent.search_many(
        [(context(case, with_context), None) for _, case in cases.iterrows()],
        DOMAIN,
        VOCABULARIES,
    )
    return [
        {
            "concept_code": r.concept.concept_code if r.concept else None,
            "concept_name": r.concept.concept_name if r.concept else None,
            "score": r.score,
            "llm_calls": r.llm_calls,
            "tool_calls": r.tool_calls,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "seconds": r.seconds,
            "budget_exhausted": r.budget_exhausted,
            "rationale": r.rationale,
            "trace": r.trace,
            "error": r.error,
        }
        for r in results
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--retriever", default="text-embedding-3-small")
    parser.add_argument("--model", default="openai/gpt-5-mini")
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--max-tool-calls", type=int, default=8)
    parser.add_argument(
        "--vector-db",
        default=None,
        help="'local', 'cloud' or 'pinecone'. Defaults to AATM_VECTOR_DB, else local.",
    )
    args = parser.parse_args()

    cases = load_cases()
    vector_db = resolve_vector_db_mode(args.vector_db)
    retriever = load_retriever(args.retriever, vector_db=vector_db)
    rows = []
    for method in args.methods:
        print(f"Running {method} on {len(cases)} texts...")
        wall = time.perf_counter()
        if method.startswith("agent"):
            found = asyncio.run(
                run_agent(
                    cases,
                    retriever,
                    args.model,
                    with_context=method == "agent",
                    max_tool_calls=args.max_tool_calls,
                )
            )
        else:
            found = run_selector(method, cases, retriever, args.model)
        wall = time.perf_counter() - wall
        for (_, case), result in zip(cases.iterrows(), found):
            rows.append(
                {
                    "method": method,
                    "id": case.id,
                    "text": case.text,
                    "gold": case.gold,
                    "correct": result["concept_code"] in case.accepted,
                    "wall_seconds": wall,
                    **result,
                }
            )

    df = pd.DataFrame(rows)
    suffix = "" if vector_db == "local" else f"__{vector_db}"
    out = (
        HERE
        / "results"
        / f"{args.retriever}__{args.model.replace('/', '_')}{suffix}.csv"
    )
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        # Keep the results of the methods not rerun.
        previous = pd.read_csv(out, dtype={"id": str, "concept_code": str})
        previous = previous[~previous.method.isin(args.methods)]
        df = pd.concat([previous, df], ignore_index=True)
        order = {m: i for i, m in enumerate(METHODS)}
        df = df.sort_values("method", key=lambda s: s.map(order), kind="stable")
    df.to_csv(out, index=False)

    summary = df.groupby("method", sort=False).agg(
        correct=("correct", "sum"),
        texts=("correct", "size"),
        llm_calls=("llm_calls", "mean"),
        input_tokens=("input_tokens", "mean"),
        output_tokens=("output_tokens", "mean"),
        seconds=("seconds", "mean"),
        wall_seconds=("wall_seconds", "first"),
    )
    summary["accuracy"] = summary.correct / summary.texts
    print(summary.round(2).to_markdown())
    print(f"\nPer-text results: {out}")
    wrong = df[~df.correct][["method", "text", "gold", "concept_code", "concept_name"]]
    print("\nErrors:\n" + wrong.to_markdown(index=False))


if __name__ == "__main__":
    main()

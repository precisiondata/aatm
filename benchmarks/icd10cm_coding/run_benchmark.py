"""Benchmark LLMs at coding clinical encounters with ICD-10-CM, with and without AATM.

The data is ACI-bench with the ICD-10-CM annotations of Yuan et al. (EMNLP 2025);
``prepare_data.py`` downloads it. Each encounter has a clinical note (and the
dialogue it was written from) and the codes two experts gave it.

Methods:

- ``llm``: the model reads the note and answers with ICD-10-CM codes, as in the
  paper's baseline. No AATM involved.
- ``aatm``: `aatm.extractors.SimpleLLMExtractor` with ``diagnoses.yaml`` extracts the
  diagnoses a coder would report, with the details that change their code; the
  concept search agent (same model) codes each one to an ICD-10-CM code, searching
  ICD10CM concepts only (``standard=False``).

Scoring (``--summary-only`` rescores saved results), micro precision, recall and F1
over encounters:

- **code**: exact ICD-10-CM codes; **category**: their first 3 characters.
- **standard**: gold and predicted codes replaced by the standard concepts they map
  to ("Maps to"), the concepts an OMOP CDM would hold.
- ``llm (verified)``: the ``llm`` answer without the codes the vocabulary database
  doesn't know as valid ICD-10-CM. AATM only answers with codes it knows.

The ``aatm`` method, standard scores and verification need ICD10CM in the vocabulary
database (``.aatm/omop.db``): include it in the Athena bundle given to ``aatm init``.

Run from the repository root:

```bash
uv run python benchmarks/icd10cm_coding/prepare_data.py
uv run python benchmarks/icd10cm_coding/run_benchmark.py --model openai/gpt-5-mini --split valid
```

Writes one line per encounter to ``results/<split>__<source>__<method>__<model>.jsonl``
(predictions, rationale, usage, seconds), then prints the scores of every result file
of that split and source. Encounters already in a result file are skipped, so an
interrupted run resumes; ``--overwrite`` starts over.
"""

import argparse
import asyncio
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

import dotenv

# pandas before torch (the retrievers): on some Windows setups, loading pyarrow after
# torch crashes the interpreter.
import pandas as pd
from pydantic import BaseModel, Field

HERE = Path(__file__).parent
DATA = HERE / "data"
RESULTS = HERE / "results"
VOCABULARY_DB = Path(".aatm/omop.db")
DEFINITIONS = HERE / "diagnoses.yaml"
METHODS = ["llm", "aatm"]

SYSTEM = """\
You are a certified medical coder. Read the clinical documentation of one outpatient
encounter and assign the ICD-10-CM diagnosis codes a professional coder would report
for it.

- Code the conditions, signs and symptoms that were evaluated, treated, monitored or
  that affected the care given in this encounter, including chronic conditions
  that were addressed.
- Do not code conditions that are negated, ruled out, resolved, only in the family
  history, or only mentioned without bearing on this encounter.
- Do not code a sign or symptom that is routinely part of a diagnosis you code.
- Use the most specific valid, billable ICD-10-CM code the documentation supports
  (laterality, type, complications). Write codes with the dot, e.g. "E11.9".
- For each code, quote the words of the documentation that support it.
"""


class AssignedCode(BaseModel):
    code: str = Field(description='An ICD-10-CM code, with the dot, e.g. "E11.9".')
    description: str = Field(description="The code's ICD-10-CM description.")
    evidence: str = Field(description="The words of the documentation that support it.")


class AssignedCodes(BaseModel):
    codes: list[AssignedCode]


# ---------------------------------------------------------------- data


def load_split(split: str) -> list[dict[str, Any]]:
    path = DATA / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run benchmarks/icd10cm_coding/prepare_data.py first."
        )
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def normalize_code(code: str) -> str:
    """``"e119 "`` -> ``"E11.9"``: upper case, no spaces, a dot after 3 characters."""
    code = re.sub(r"[^A-Z0-9.]", "", code.upper())
    code = code.replace(".", "")
    return f"{code[:3]}.{code[3:]}" if len(code) > 3 else code


def slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "_", model)


def result_path(split: str, source: str, method: str, model: str, run: int = 1) -> Path:
    """Run 1 has no suffix; later runs of the same method end in ``__run<N>``."""
    suffix = "" if run == 1 else f"__run{run}"
    return RESULTS / f"{split}__{source}__{method}__{slug(model)}{suffix}.jsonl"


def read_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- methods


class LLMMethod:
    """The model answers with ICD-10-CM codes."""

    def __init__(self, model: str, settings: Optional[dict[str, Any]]):
        from aatm.llm import create_llm

        self.llm = create_llm(model, settings=settings)

    async def __call__(self, text: str) -> dict[str, Any]:
        answer = await self.llm.generate(
            system=SYSTEM, user=text, output_type=AssignedCodes
        )
        return {
            "codes": [normalize_code(c.code) for c in answer.codes],
            "details": [c.model_dump() for c in answer.codes],
        }


class AatmMethod:
    """Extract the encounter's diagnoses, then code each to ICD-10-CM with the agent.

    The extraction uses ``diagnoses.yaml``; each diagnosis goes to the concept search
    agent with its line of the note and its details, restricted to ICD10CM concepts
    (``standard=False``).
    """

    def __init__(
        self,
        model: str,
        settings: Optional[dict[str, Any]],
        retriever: Any,
        max_tool_calls: int,
    ):
        from aatm.concept_search import ConceptSearchAgent, ConceptSearchTools
        from aatm.data_models.extractors import EntityDefinitions
        from aatm.extractors import SimpleLLMExtractor

        self.extractor = SimpleLLMExtractor(
            model_id=model,
            entity_definitions=EntityDefinitions.from_yaml(DEFINITIONS),
            settings=settings,
        )
        self.agent = ConceptSearchAgent(
            model,
            ConceptSearchTools(retriever, VOCABULARY_DB),
            max_tool_calls=max_tool_calls,
            settings=settings,
        )

    async def __call__(self, text: str) -> dict[str, Any]:
        from aatm.concept_search import CodingContext

        (extraction,) = await self.extractor([text])
        diagnoses = [r for r in extraction.to_records() if r.key == "diagnosis"]
        contexts = [
            CodingContext(
                text=r.text,
                related={
                    f"detail_{n}": d.text
                    for n, d in enumerate(r.children.get("diagnosis_detail", []), 1)
                },
                sentence=r.line,
                target="ICD-10-CM diagnosis code of an outpatient encounter",
            )
            for r in diagnoses
        ]
        results = await self.agent.search_many(
            [(c, None) for c in contexts], vocabularies=["ICD10CM"], standard=False
        )
        details = [
            {
                "text": c.text,
                "related": c.related,
                "code": r.concept.concept_code if r.concept else None,
                "name": r.concept.concept_name if r.concept else None,
                "confidence": r.confidence,
                "tool_calls": r.tool_calls,
                "rationale": r.rationale,
                "error": r.error,
            }
            for c, r in zip(contexts, results)
        ]
        return {
            "codes": list(dict.fromkeys(d["code"] for d in details if d["code"])),
            "details": details,
        }


async def run(
    method: Any,
    records: list[dict[str, Any]],
    source: str,
    path: Path,
    concurrency: int,
    timeout: float,
) -> None:
    from aatm.llm import track_usage

    # Encounters that failed are dropped from the file and run again.
    previous = read_results(path)
    kept = [r for r in previous if not r.get("error")]
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(kept) < len(previous):
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in kept)
    done = {r["encounter_id"] for r in kept}
    todo = [r for r in records if r["encounter_id"] not in done]
    print(
        f"{len(done)} encounters already in {path.name}, {len(todo)} to run "
        f"({len(previous) - len(kept)} failed before)"
    )
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    finished = 0

    async def one(record: dict[str, Any]) -> None:
        nonlocal finished
        async with semaphore:
            start = time.perf_counter()
            with track_usage() as usage:
                try:
                    answer = await asyncio.wait_for(method(record[source]), timeout)
                    error = None
                except Exception as e:  # noqa: BLE001 - one note must not stop a run
                    answer, error = {}, f"{type(e).__name__}: {e}"
            line = {
                "encounter_id": record["encounter_id"],
                **answer,
                "error": error,
                "llm_calls": usage.calls,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "seconds": round(time.perf_counter() - start, 2),
            }
        async with lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
            finished += 1
            status = f"error: {error}" if error else "ok"
            print(
                f"  [{finished}/{len(todo)}] {record['encounter_id']} {status}",
                flush=True,
            )

    await asyncio.gather(*(one(r) for r in todo))


# ---------------------------------------------------------------- scoring


class Icd10cm:
    """ICD-10-CM codes in the vocabulary database, and the standard concepts they map to."""

    def __init__(self, path: Path = VOCABULARY_DB):
        if not path.exists():
            raise SystemExit(f"{path} not found: run `aatm init` first.")
        self.db = sqlite3.connect(path)
        (n,) = self.db.execute(
            "SELECT COUNT(*) FROM concept WHERE vocabulary_id = 'ICD10CM'"
        ).fetchone()
        self.available = bool(n)
        if not n:
            print(
                f"{path} has no ICD10CM concepts: codes are scored as written, but "
                "can't be checked or mapped to standard concepts. To score those, "
                "download a vocabulary bundle from Athena that includes ICD10CM and "
                "run `aatm init --vocab-zip <link>` again."
            )
        self._concepts: dict[str, Optional[dict[str, Any]]] = {}
        self._maps: dict[int, set[int]] = {}

    def lookup(self, codes: set[str]) -> None:
        """Load the concepts of `codes` and their "Maps to", in two queries."""
        new = sorted(codes - set(self._concepts))
        if not new:
            return
        marks = ",".join("?" * len(new))
        found = {
            code: {"concept_id": int(cid), "class": cls, "invalid": invalid}
            for cid, code, cls, invalid in self.db.execute(
                "SELECT concept_id, concept_code, concept_class_id, invalid_reason "
                f"FROM concept WHERE vocabulary_id = 'ICD10CM' AND concept_code IN ({marks})",
                new,
            )
        }
        for code in new:
            self._concepts[code] = found.get(code)
        ids = [c["concept_id"] for c in found.values()]
        if ids:
            marks = ",".join("?" * len(ids))
            for a, b in self.db.execute(
                "SELECT concept_id_1, concept_id_2 FROM concept_relationship "
                "WHERE relationship_id = 'Maps to' AND invalid_reason IS NULL "
                f"AND concept_id_1 IN ({marks})",
                ids,
            ):
                # Ids may be stored as text: compare them as integers.
                self._maps.setdefault(int(a), set()).add(int(b))

    def valid(self, code: str) -> bool:
        """A current ICD-10-CM code (billable or not)."""
        c = self._concepts.get(code)
        return c is not None and c["invalid"] is None

    def billable(self, code: str) -> bool:
        c = self._concepts.get(code)
        return self.valid(code) and "nonbill" not in (c["class"] or "").lower()

    def standard(self, codes: set[str]) -> set[int]:
        out: set[int] = set()
        for code in codes:
            c = self._concepts.get(code)
            if c:
                out |= self._maps.get(c["concept_id"], set())
        return out


def prf(gold: list[set], pred: list[set]) -> dict[str, float]:
    tp = sum(len(g & p) for g, p in zip(gold, pred))
    n_pred = sum(len(p) for p in pred)
    n_gold = sum(len(g) for g in gold)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _undotted(code: str) -> str:
    return code.replace(".", "")


def prefix_prf(gold: list[set], pred: list[set], n: int) -> dict[str, float]:
    """Prefix-n of Yuan et al. (2025): exact match, accepting near misses.

    A predicted code matches a gold code of the same encounter when they agree up to
    `n` levels above the gold code's leaf: on the gold code without its last `n`
    characters, never shorter than its 3-character category. I50.2 or I50.22 for
    gold I50.21 matches at n=1, I50.9 at n=2. Precision counts the predicted codes
    that match a gold code, recall the gold codes that a prediction matches.
    """

    def matches(p: str, g: str) -> bool:
        p, g = _undotted(p), _undotted(g)
        k = max(3, len(g) - n)
        return p[:k] == g[:k]

    matched_pred = sum(
        any(matches(p, g) for g in gs) for gs, ps in zip(gold, pred) for p in ps
    )
    matched_gold = sum(
        any(matches(p, g) for p in ps) for gs, ps in zip(gold, pred) for g in gs
    )
    n_pred = sum(len(p) for p in pred)
    n_gold = sum(len(g) for g in gold)
    precision = matched_pred / n_pred if n_pred else 0.0
    recall = matched_gold / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def parse_result_name(path: Path) -> tuple[str, str, str, str, int]:
    """``split, source, method, model, run`` of a result file."""
    match = re.fullmatch(r"(.+?)__(.+?)__(.+?)__(.+?)(?:__run(\d+))?", path.stem)
    if match is None:
        raise ValueError(f"Not a result file name: {path.name}")
    split, source, method, model, run = match.groups()
    return split, source, method, model, int(run or 1)


def summarize(split: str, source: str) -> None:
    records = {r["encounter_id"]: r for r in load_split(split)}
    gold_codes = {
        eid: {normalize_code(c["code"]) for c in r["icd10_codes"]}
        for eid, r in records.items()
    }
    files = sorted(RESULTS.glob(f"{split}__{source}__*.jsonl"))
    if not files:
        print(f"No results for {split} / {source} in {RESULTS}")
        return
    icd = Icd10cm()
    runs = {}
    for path in files:
        _, _, method, model, repeat = parse_result_name(path)
        runs[(method, model, repeat)] = {
            r["encounter_id"]: r for r in read_results(path)
        }
    icd.lookup(
        set().union(*gold_codes.values())
        | {c for run in runs.values() for r in run.values() for c in r.get("codes", [])}
    )
    gold_unmapped = sorted(
        c for c in set().union(*gold_codes.values()) if not icd.standard({c})
    )
    if icd.available and gold_unmapped:
        print(f"Gold codes without a standard concept: {', '.join(gold_unmapped)}")

    rows = []
    for (method, model, repeat), run in runs.items():
        ids = [eid for eid in records if eid in run]
        gold = [gold_codes[eid] for eid in ids]
        gold_std = [icd.standard(g) for g in gold]
        common = {
            "model": model,
            "run": repeat,
            "encounters": len(ids),
            "errors": sum(bool(run[eid].get("error")) for eid in ids),
            "input tokens": sum(run[eid]["input_tokens"] or 0 for eid in ids)
            / max(len(ids), 1),
            "output tokens": sum(run[eid]["output_tokens"] or 0 for eid in ids)
            / max(len(ids), 1),
            "seconds": sum(run[eid]["seconds"] for eid in ids) / max(len(ids), 1),
        }
        variants = {method: lambda codes: codes}
        # AATM only answers with codes of the vocabulary database.
        if method == "llm" and icd.available:
            variants["llm (verified)"] = lambda codes: {
                c for c in codes if icd.valid(c)
            }
        for name, keep in variants.items():
            pred = [keep(set(run[eid].get("codes", []))) for eid in ids]
            codes = [c for p in pred for c in p]
            rows.append(
                {
                    "method": name,
                    **common,
                    **{f"code {k}": v for k, v in prf(gold, pred).items()},
                    **{
                        f"prefix-{n} {k}": v
                        for n in (1, 2)
                        for k, v in prefix_prf(gold, pred, n).items()
                    },
                    **{
                        f"category {k}": v
                        for k, v in prf(
                            [{c[:3] for c in g} for g in gold],
                            [{c[:3] for c in p} for p in pred],
                        ).items()
                    },
                    "codes / encounter": len(codes) / max(len(ids), 1),
                }
            )
            if icd.available:
                rows[-1].update(
                    {
                        f"standard {k}": v
                        for k, v in prf(
                            gold_std, [icd.standard(p) for p in pred]
                        ).items()
                    }
                )
                rows[-1]["invalid codes"] = sum(not icd.valid(c) for c in codes)
                rows[-1]["non-billable codes"] = sum(
                    icd.valid(c) and not icd.billable(c) for c in codes
                )

    df = pd.DataFrame(rows)
    keys = ["method", "model"]
    partial = df[df.encounters < len(records)]
    if len(partial):
        print(
            "Not comparable to the papers, which score every encounter of the split: "
            + ", ".join(
                f"{r.method} {r.model} run {r.run} ({r.encounters}/{len(records)})"
                for r in partial.itertuples()
            )
        )

    # As in Symphony (Edin et al., 2026, Table 3): micro recall, precision and F1
    # x 100, mean ± standard deviation over runs.
    def mean_sd(values: pd.Series) -> str:
        values = values * 100
        if len(values) < 2:
            return f"{values.mean():.1f}"
        return f"{values.mean():.1f} ± {values.std():.1f}"

    paper = df.groupby(keys).agg(
        runs=("run", "count"),
        **{
            f"{label} {m}": (f"{metric} {m}", mean_sd)
            for label, metric in [("exact", "code"), ("prefix-1", "prefix-1")]
            for m in ("recall", "precision", "f1")
        },
        **{"prefix-2 f1": ("prefix-2 f1", mean_sd)},
    )
    print(f"\n## {split}, {source} ({len(records)} encounters): paper metrics\n")
    print(paper.to_markdown())

    details = df.drop(columns="run").groupby(keys).mean(numeric_only=True)
    print(f"\n## {split}, {source}: all metrics (mean over runs)\n")
    print(details.round(3).to_markdown())


# ---------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ICD-10-CM coding of ACI-bench encounters."
    )
    parser.add_argument("--model", help='e.g. "openai/gpt-5-mini"')
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument(
        "--source",
        choices=["note", "dialogue"],
        default="note",
        help="What the model reads: the clinical note or the dialogue.",
    )
    parser.add_argument(
        "--limit", type=int, help="Only the first N encounters, for a quick check."
    )
    parser.add_argument(
        "--retriever",
        default="text-embedding-3-small",
        help="Embedding model of the agent's semantic search (aatm method).",
    )
    parser.add_argument(
        "--vector-db",
        help="'local', 'cloud', 'pinecone' or 'pgvector'. Defaults to AATM_VECTOR_DB, "
        "else local.",
    )
    parser.add_argument("--max-tool-calls", type=int, default=8)
    parser.add_argument(
        "--temperature", type=float, help="Leave unset for reasoning models."
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="Seconds per encounter. The aatm method codes every condition of a note.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        nargs="+",
        default=[1],
        help="Run numbers, e.g. 1 2 3 4 5: Symphony reports the mean and standard "
        "deviation of 5 runs. Each run has its own result file.",
    )
    parser.add_argument(
        "--variant",
        help="A label added to the method's name, e.g. v2, to keep the results of "
        "versions of the definitions or the agent apart.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    dotenv.load_dotenv()

    if not args.summary_only:
        if not args.model:
            parser.error("--model is required unless --summary-only")
        records = load_split(args.split)[: args.limit]
        settings = (
            {"temperature": args.temperature} if args.temperature is not None else None
        )
        for name in args.methods:
            if name == "llm":
                method: Any = LLMMethod(args.model, settings)
            else:
                from aatm.registries.retrievers import load_retriever
                from aatm.vector_db import resolve_vector_db_mode

                vector_db = resolve_vector_db_mode(args.vector_db)
                retriever = load_retriever(args.retriever, vector_db=vector_db)
                method = AatmMethod(
                    args.model, settings, retriever, args.max_tool_calls
                )
                # The retriever changes the results: it is part of the method's name.
                name = f"aatm-{args.retriever}" + (
                    "" if vector_db == "local" else f"-{vector_db}"
                )
            if args.variant:
                name += f"-{args.variant}"
            for repeat in args.runs:
                path = result_path(args.split, args.source, name, args.model, repeat)
                if args.overwrite and path.exists():
                    path.unlink()
                print(
                    f"\n# {name}, {args.model}, {args.split}, {args.source}, "
                    f"run {repeat}"
                )
                asyncio.run(
                    run(
                        method,
                        records,
                        args.source,
                        path,
                        args.concurrency,
                        args.timeout,
                    )
                )
    summarize(args.split, args.source)


if __name__ == "__main__":
    main()

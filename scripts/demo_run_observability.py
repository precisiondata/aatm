"""Show what a long run logs, with no API keys: a fake extractor and coder.

    uv run python scripts/demo_run_observability.py

The fake models sleep to look like real calls and report made-up token counts, so
the progress lines, the per-stage summary and the cost estimate look as they would
with real models. Output goes to a temporary folder, printed at the end.
"""

import asyncio
import random
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from aatm.llm.usage import record_usage  # noqa: E402
from aatm.logs import configure_logging  # noqa: E402
from aatm.omop import OmopExtractionPipeline, OmopRun  # noqa: E402
from test_omop_coding import FakeCoder, UnitLookup, lab, lab_definitions  # noqa: E402
from test_omop_mapping import extract  # noqa: E402

TESTS = ["Hb", "glucose", "BMI", "creatinine", "sodium", "potassium"]


class SlowExtractor:
    """Finds two or three lab tests per note, in about half a second."""

    model_id = "openai/gpt-5-mini"

    def __init__(self, definitions):
        self.definitions = definitions

    async def __call__(self, texts):
        results = []
        for text in texts:
            await asyncio.sleep(random.uniform(0.3, 0.7))
            if "fail" in text:
                raise RuntimeError("503 Service Unavailable")
            record_usage(
                random.randint(1500, 2500), random.randint(150, 400), self.model_id
            )
            names = [w.strip(",") for w in text.split() if w.strip(",") in TESTS]
            entities = [lab(f"e{i}", name) for i, name in enumerate(names, start=1)]
            results.append(extract(self.definitions, text, entities))
        return results


class SlowCoder(FakeCoder):
    """Codes each text with one "agent" call of about 50 ms."""

    def code(self, texts, domain=None, vocabularies=None):
        for _ in texts:
            time.sleep(0.05)
            record_usage(
                random.randint(3000, 6000),
                random.randint(200, 600),
                "anthropic/claude-haiku-5-5",
            )
        return super().code(texts, domain=domain, vocabularies=vocabularies)


def notes(n):
    for i in range(1, n + 1):
        names = random.sample(TESTS, k=random.randint(2, 3))
        text = ", ".join(f"{name} {random.randint(5, 140)}" for name in names)
        if i % 37 == 0:
            text += " fail"
        yield {"note_id": i, "text": text}


async def main():
    random.seed(7)
    configure_logging("INFO")
    output = Path(tempfile.mkdtemp(prefix="aatm-demo-"))
    definitions = lab_definitions()
    pipeline = OmopExtractionPipeline(
        definitions,
        extractor=SlowExtractor(definitions),
        vocabulary=UnitLookup(),
        mapper=SlowCoder(),
    )
    summary = await pipeline.run(
        notes(120),  # a generator: the total is given so that the ETA can be shown
        output,
        chunk_size=10,
        total=120,
        log_every=2,
        prices={"openai/gpt-5-mini": (0.25, 2.00)},  # the coder's model is unpriced
    )
    print("\n--- summary.stages['extraction'].usage.by_model ---")
    print(summary.stages["extraction"].usage.by_model)
    (saved,) = OmopRun(output).summaries()[-1:]
    print(f"\nSaved summary: {next((output / 'runs').glob('*.json'))}")
    print(f"Read back with OmopRun(...).summaries(): completed={saved.completed}")


if __name__ == "__main__":
    asyncio.run(main())

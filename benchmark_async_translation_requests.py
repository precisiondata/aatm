import asyncio
import time
from typing import List

from aatm.translators import GeminiTranslator
from aatm.data_models import Translation


class AsyncGeminiTranslator(GeminiTranslator):
    async def call_concurrent(self, texts: List[str]) -> List[Translation]:
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(
                    self.client.aio.models.generate_content(
                        model=self.model,
                        contents=f'Translate the following text into English: "{t}"',
                    )
                )
                for t in texts
            ]
        return [t.result() for t in tasks]

    async def call_sequential(self, texts: List[str]) -> List[Translation]:
        results = []
        for t in texts:
            r = await self.client.aio.models.generate_content(
                model=self.model,
                contents=f'Translate the following text into English: "{t}"',
            )
            results.append(r)
        return results


async def benchmark(translator: AsyncGeminiTranslator, texts: List[str]):
    # Warm-up (important for network + TLS)
    await translator.call_sequential(texts[:1])

    print(f"\nRunning benchmark with {len(texts)} requests\n")

    t0 = time.perf_counter()
    await translator.call_sequential(texts)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    await translator.call_concurrent(texts)
    t3 = time.perf_counter()

    seq_time = t1 - t0
    conc_time = t3 - t2

    print(f"Sequential time : {seq_time:.2f}s")
    print(f"Concurrent time : {conc_time:.2f}s")
    print(f"Speedup         : {seq_time / conc_time:.2f}×")


if __name__ == "__main__":
    texts = [f"Sentence number {i}" for i in range(20)]

    translator = AsyncGeminiTranslator(
        model="models/gemini-2.5-flash"  # adjust if needed
    )

    asyncio.run(benchmark(translator, texts))

# Sharding benchmark

Does splitting a run into shards, one process each (`aatm.omop.run_sharded`), make it
faster? ([issue #56](https://github.com/precisiondata/aatm/issues/56))

Every run here has the **same total concurrency** (extraction calls in flight, agent
searches, rate limits), divided among the shards. Only the split into processes
changes, so a gain comes from the split itself, not from doing more at once.

## Setup

[`run_benchmark.py`](run_benchmark.py) builds synthetic notes from the 30 lab texts of
the [concept search benchmark](../concept_search/README.md): 6 filler sentences and 6
lab results per note (`Labs 03/15/2025: serum creatinine 1.4 mg/dl.`), with about 300
distinct test names (specimens, casings) drawn Zipf-like.

Real: the `SimpleLLMExtractor` (prompt, parsing, span validation), the rate limiter,
the builder with the vocabulary database (`.aatm/omop.db`), the Parquet writes and the
ledger. Simulated: the LLM, which answers from the note's lab lines after a log-normal
latency (no API call), and, in some scenarios, the coder.

| Scenario | Coder |
|---|---|
| `instant` | Answers at once: measures the framework. |
| `retrieval` | `TerminologyMapper`, embeddinggemma-300M and a 50k-row ChromaDB index, on the GPU. |
| `agent` | Each distinct text waits ~2 s (the agent's LLM calls), 16 at once in all. |
| `rate-limited` | `instant`, with the extraction model capped at `--rpm` in all. |

Two settings:

- **Hosted API**: 2,000 notes, 4 s mean latency, 64 calls in flight in all.
- **Fast model** (e.g. a local vLLM server): 5,000 notes, 0.3 s latency, 256 calls in
  flight.

Coding cache off. Machine: 24 cores, 128 GB RAM, RTX 4090, Windows 11. One repeat per
setting: differences under ~5% are noise.

```bash
uv run python benchmarks/sharding/run_benchmark.py --notes 2000 --latency 4 --concurrency 64 --shards 1 2 4 8 --rpm 1200 --label hosted-api
uv run python benchmarks/sharding/run_benchmark.py --notes 5000 --latency 0.3 --concurrency 256 --shards 1 2 4 8 --scenarios instant retrieval --label fast-model
uv run python benchmarks/sharding/run_benchmark.py --notes 2000 --latency 4 --concurrency 64 --shards 1 4 --chunk-size 1000 --scenarios instant retrieval --label hosted-api-chunk1000
uv run python benchmarks/sharding/run_benchmark.py --notes 5000 --latency 0.3 --concurrency 256 --shards 1 8 --chunk-size 1000 --scenarios instant retrieval --label fast-model-chunk1000
uv run python benchmarks/sharding/run_benchmark.py --notes 1000 --latency 4 --concurrency 64 --shards 1 4 --rpm 300 --scenarios rate-limited --label hosted-api-rpm300
```

The tables below were measured with a runner that extracted a chunk at a time and
waited for its slowest call; the runner now streams notes through extraction (see
[Streaming extraction](#streaming-extraction)). Those runs are saved with
`-streaming` labels:

```bash
uv run python benchmarks/sharding/run_benchmark.py --notes 2000 --latency 4 --concurrency 64 --shards 1 4 --scenarios instant retrieval --label hosted-api-streaming
uv run python benchmarks/sharding/run_benchmark.py --notes 5000 --latency 0.3 --concurrency 256 --shards 1 8 --scenarios instant retrieval --label fast-model-streaming
uv run python benchmarks/sharding/run_benchmark.py --notes 5000 --latency 0.3 --concurrency 256 --shards 1 --chunk-size 1000 --scenarios instant retrieval --label fast-model-streaming-chunk1000
```

## Results (2026-11-11)

Notes per minute, wall time (process start-up and model loading included). Full rows,
with stage times, CPU time and memory, are in [`results/`](results/).

### Hosted API

| Scenario | Chunk size | 1 shard | 2 | 4 | 8 |
|---|---|---|---|---|---|
| instant | 100 | 421 | 614 | 754 | 811 |
| instant | 1000 | **851** | | 779 | |
| retrieval | 100 | 383 | 591 | 661 | 731 |
| retrieval | 1000 | **715** | | 740 | |
| agent | 100 | 243 | 256 | 252 | 229 |
| rate-limited, 300/min | 100 | 306 | | 298 | |

The ceiling is 64 calls / 4 s = 960 notes/min.

### Fast model

| Scenario | Chunk size | 1 shard | 2 | 4 | 8 |
|---|---|---|---|---|---|
| instant | 100 | 5,900 | 10,500 | 17,200 | 22,900 |
| instant | 1000 | **20,000** | | | 27,700 |
| retrieval | 100 | 1,310 | 2,310 | 3,780 | 5,070 |
| retrieval | 1000 | **4,770** | | | 7,530 |

Memory: about 0.2 GB per process with the instant coder, 2.2 GB with the embedding
model (17 GB for 8 retrieval shards).

### Streaming extraction

The runner used to extract a chunk with all its calls at once and wait for the
slowest before starting the next chunk, so the calls in flight drained at the end of
every chunk. It now keeps `max_concurrency` calls in flight all the time, starting a
new one as soon as one ends, stores each result as it comes back, and builds chunks
as they fill up. Notes per minute, before and after:

| Setting | Scenario | Chunk size | Shards | Before | After |
|---|---|---|---|---|---|
| Hosted API | instant | 100 | 1 | 421 | **889** |
| Hosted API | instant | 100 | 4 | 754 | 864 |
| Hosted API | retrieval | 100 | 1 | 383 | **770** |
| Hosted API | retrieval | 100 | 4 | 661 | 775 |
| Fast model | instant | 100 | 1 | 5,900 | 11,900 |
| Fast model | instant | 1000 | 1 | 20,000 | **24,200** |
| Fast model | instant | 100 | 8 | 22,900 | 28,400 |
| Fast model | retrieval | 100 | 1 | 1,310 | 1,220 |
| Fast model | retrieval | 1000 | 1 | 4,770 | 4,440 |
| Fast model | retrieval | 100 | 8 | 5,070 | 3,770, 4,450 |

With a hosted API, one process with 100-note chunks now runs at 93% of the 960
notes/min ceiling (run time; the rest is the latency tail of the last calls), and
shards add nothing. With retrieval on a fast model, coding a chunk at a time is the
limit either way, and the differences are within the noise of single runs (the
8-shard run, repeated, gave 3,770 and 4,450).

## What it shows

1. **With a hosted API, sharding on one machine does not help.** The CPU is nearly
   idle: one process used 11 CPU seconds in a 285 s run. The apparent gains with
   100-note chunks came from the runner, not from more processes: a chunk's
   extraction waited for its slowest call before the next chunk started, so the
   slots in flight drained at the end of every chunk, and more shards hid the gaps.
   The runner now streams notes through extraction: one process reaches 889
   notes/min with 100-note chunks, more than 8 shards did, with an eighth of the
   memory.
2. **Under a rate limit, sharding cannot help**: the limit is the provider's, and
   `run_sharded` divides it among the processes (306 vs 298 notes/min).
3. **The agentic coder is bound by its concurrency and latency**, not by processes:
   with 16 searches in all, 1 to 8 shards give the same speed. Raise the coder's
   `max_concurrency` instead.
4. **Sharding does help when a single process is busy**: with a fast model, one
   process reaches 20,000 notes/min and 8 shards 27,700 (1.4x); with local embedding
   retrieval, 4,770 and 7,530 (1.6x), where coding a chunk at a time is the limit.
   The cost is memory: one embedding model per process.
5. Sharding remains the way to **use several machines or cluster jobs**, where each
   shard is an independent job that resumes on its own.

### Recommendations

- Hosted API: one process. Raise `max_concurrency` and the rate limits to your quota;
  `chunk_size` no longer limits extraction.
- Fast or local models: larger chunks (1,000) still help, since each chunk's build
  has a fixed cost.
- Agentic coder: raise its `max_concurrency`.
- Fast or local models, local retrieval: shard on one machine, within its memory.
- Several machines or a cluster: shard, one shard per job.

### Caveats

- The LLM and the agent are simulated: real APIs add jitter, 429s and varying output
  lengths, and the real agent does some CPU work per text.
- The index is a 50k-row subset; the full index is slower to search and larger.
- `--mapper-batch-size 16` works around ChromaDB failing with "too many SQL
  variables" on filtered queries of 64 or more texts at once.

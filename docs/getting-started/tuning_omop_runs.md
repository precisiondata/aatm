# Tuning large OMOP runs

A full run with `pipeline.run` has many settings, but most runs need two or three of
them. This page sorts them into tiers: start with tier 1, move down only when a
[plan](#check-the-plan-first) or a pilot says you need to.

## Which setup fits your case

| Your case | Use | Why |
|---|---|---|
| Hosted API, the run can wait up to 24 h | `batch=True` | About half the price; no concurrency or rate limits to tune |
| Hosted API, you need it today | `max_concurrency` and `set_rate_limits` | One process keeps that many calls in flight; the provider's quota is the limit |
| Local model (Ollama, vLLM) | `max_concurrency` set to what the server serves at once; shards across GPUs or machines | The server, not the API, is the limit |
| Several machines or cluster jobs | `shard=(index, count)` | Each machine runs its own part of the dataset |

Sharding a hosted-API run on one machine rarely helps: the processes share one rate
limit, and one process already keeps `max_concurrency` calls in flight.

## Check the plan first

`pipeline.plan` is a dry run: no model call, no note read beyond counting them. It
prints the settings the run would use and where each came from, the notes left to do,
an estimate when there is a pilot, and warnings with a fix:

```python
from aatm.omop import OmopExtractionPipeline

pipeline = OmopExtractionPipeline(
    "ecog_performance_status", model_id="openai/gpt-5-mini", max_concurrency=1
)
print(pipeline.plan("notes.parquet", "output/ecog"))
```

```text
Run plan for output/ecog
notes: 120,000 in the source; 120,000 to extract
settings:
  max_concurrency  1    [argument]  extraction calls in flight; max_concurrency=1
  chunk_size       100  [default]   notes built and written together: it bounds memory, not extraction
estimate: none without a pilot; run a few hundred notes into a scratch folder and pass pilot=<that folder>
warnings:
  - max_concurrency=1 with a hosted API: about 6 notes/min (at about 10 s per call), far below what the API serves
    fix: set max_concurrency=16 or more, with aatm.llm.set_rate_limits to stay under your quota; or batch=True if the run can wait a day
```

`plan` takes the same arguments as `run` (`chunk_size`, `batch`, `shard`, `total`...),
so you can compare setups before paying for one. `plan.settings`, `plan.estimate` and
`plan.warnings` hold the report as data.

**A pilot gives the estimate.** Run a few hundred notes like the dataset into a
scratch folder, with prices, then plan the full run from it:

```python
sample = "notes_sample.parquet"  # a few hundred notes, like the full dataset
await pipeline.run(sample, "output/pilot", prices={"openai/gpt-5-mini": (0.25, 2.00)})
print(pipeline.plan("notes.parquet", "output/ecog", pilot="output/pilot"))
```

```text
estimate (from a pilot of 300 notes): 241,200 LLM calls, 310,450,000 input and 52,300,000 output tokens, cost 182.21, about 9h40m (207 notes/min)
  about 4.6 s per extraction call in the pilot, 16 in flight
```

Calls, tokens and cost per note are scaled to the notes left; time per call is scaled
to the planned concurrency, and paced by `requests_per_minute` when you set one. With
`batch=True`, the extraction cost is halved and the time is "within 24 hours". When
`output_dir` already holds a run, the notes it did are left out, and its own summaries
serve as the pilot.

## Tier 1: always

These decide what the run produces and what it costs. Set them for every run.

- **Definitions**: which entities to extract and how they map to OMOP
  ([Writing entity definitions](entity_definitions.md)).
- **Model**: `model_id`, the extraction model. Try a small one on the pilot first.
- **Coder**: `coder=`, how extracted texts get concept ids
  ([the agent](coding_text.md#searching-with-an-agent) or a terminology mapper), and
  `min_score` in the mappings, the confidence below which `0` is stored.
- **`cache=True`**: each distinct text is coded once, across chunks, runs and rebuilds.
  Without it, the agent coder pays again for repeated texts in every chunk and on every
  rebuild; `plan` warns about it.

The throughput settings need no change for most runs: with no `max_concurrency`, a run
keeps 16 extraction calls in flight with a hosted API and 4 with a local server
(Ollama, or an `openai-compatible` server on `localhost`), whatever the `chunk_size`.

## Tier 2: large runs

Pick **one** of these two, as in the [table above](#which-setup-fits-your-case).

**Live, faster: `max_concurrency` and rate limits.** Raise the calls in flight, and
set your provider's quota so the run stays under it instead of retrying 429 errors:

```python
from aatm.llm import set_rate_limits

set_rate_limits("openai", requests_per_minute=5000, tokens_per_minute=4_000_000)
pipeline = OmopExtractionPipeline(
    "ecog_performance_status", model_id="openai/gpt-5-mini", max_concurrency=64,
    cache=True,
)
```

The extraction concurrency is the lowest of the extractor's `max_concurrency` and the
model's (from `set_rate_limits`). With neither, but a `requests_per_minute`, it is
derived from that rate: enough calls to use it when a call takes up to 30 s, at most
256. The rate limits cover every call to the model, extraction and coding together.

Coding has its own concurrency: the coder's `max_concurrency` (8 by default), on top of
the extraction calls, so more calls than `max_concurrency` can be in flight. An
attribute labeller adds its `max_concurrency` calls per note being extracted. When the
same model serves all of them, `set_rate_limits("<model>", max_concurrency=...)` caps
them together.

**Offline, cheaper: `batch=True`.** Extraction goes through the provider's batch
endpoint (OpenAI, Anthropic, Gemini): about half the price, results within 24 hours.
Concurrency and rate limits no longer matter for extraction; coding stays live. See
[Batch extraction](large_datasets.md#batch-extraction-about-half-the-price).

## Tier 3: experts

Change these only when a plan, a pilot or the progress logs point at them.

- **`chunk_size`** (100): notes built and written together, one part file per table
  per chunk. It bounds memory and lets the builder group repeated texts per coding
  call; it does **not** limit extraction. Raise it (1000) for fewer, larger files or
  larger batch jobs; lower it if memory is tight or Gemini refuses a batch job as too
  large.
- **Shards** (`shard=(index, count)`, `run_sharded`): several processes or machines,
  each with its own part of the dataset. They pay off with a local model or local
  embedding retrieval that keeps one process busy, and to spread a run over machines.
  `run_sharded` divides the rate limits among its processes.
- **`max_batches`** (20) and **`poll_every`** (60 s): batch jobs pending at once, and
  seconds between polls. Raise `poll_every` for long jobs; lower `max_batches` if your
  provider caps the jobs or enqueued tokens you may have.
- **`max_attempts`** (3): times a failing note is tried, across runs.

## Reading a running job

Each run logs a progress line every minute with the throughput and time left, and saves
its summary in `runs/` (see [Processing large datasets](large_datasets.md)).
The summary breaks the time down by stage: when extraction's notes/min is far below
`max_concurrency` × 60 / seconds per call, the rate limits or 429 retries are the
limit; when building is the slower stage, the coder is, and `cache=True` or a larger
coder `max_concurrency` helps.

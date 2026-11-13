# Processing large datasets

`await pipeline(notes)` holds every note and row in memory, and a failure partway
loses the model calls already made. For a full dataset, use `pipeline.run`: it reads
the notes in chunks, writes each chunk's rows to Parquet files as it goes, and records
its progress, so a run that stops can be resumed.

```python
import asyncio
from aatm.omop import OmopExtractionPipeline, OmopRun

pipeline = OmopExtractionPipeline(
    "ecog_performance_status",
    model_id="openai/gpt-5-mini",
    max_concurrency=16,  # notes extracted at once (16 is the default for a hosted API)
    cache=True,          # code each distinct text once, across chunks and runs
)
summary = asyncio.run(
    pipeline.run(
        "notes.parquet",               # or .csv, a folder of Parquet files, a DataFrame...
        "output/ecog",
        chunk_size=200,
        columns={"note_text": "text"}, # source column -> ClinicalNote field
    )
)
print(summary)  # notes built, skipped and failed, rows per table, cache hit rate
```

## From the command line

`aatm omop run` does the same, for shell scripts, cron jobs
and schedulers (SLURM, Kubernetes), with no Python wrapper:

```bash
aatm omop run notes.parquet output/ecog   --definitions ecog_performance_status   --model openai/gpt-5-mini --max-concurrency 16 --cache   --chunk-size 200 --columns note_text=text   --price openai/gpt-5-mini=0.25,2.00
```

Repeat `--definitions` for several files (predefined names or your own YAML files) and
`--columns` for several renames. Coding `map` sources takes `--coder mapper` (the
terminology mapper, with `--retriever` and `--selector`) or `--coder agent` (the
concept search agent, with `--coder-model`, `--retriever` and `--max-tool-calls`); the
retriever defaults to the one `aatm init` built. `--min-score` applies to `map`
sources without a `min_score` of their own, and `--labeller <decision model>
--labeller-mode verify --min-probability 0.8` labels attributes with a decision model.
`aatm omop run --help` lists every option with its default.

For anything longer than a few flags, write the options in a YAML file, under their
names with underscores, and commit it so the run can be repeated. Flags override it:

```yaml
# run.yaml
source: notes.parquet
output: output/ecog
definitions: [ecog_performance_status, ./body_weight.yaml]
model: openai/gpt-5-mini
coder: agent
columns: {note_text: text, patient_id: person_id}
prices: {openai/gpt-5-mini: [0.25, 2.00]}
chunk_size: 200
cache: true
```

```bash
aatm omop run --config run.yaml
aatm omop run --config run.yaml --limit 50 --json   # a quick pilot on 50 notes
```

Progress is logged to standard error; standard output gets the path of the run's
summary, or the summary itself with `--json`. The exit code tells a scheduler how it
went: `0` when every note is built, `2` when the run completed but some notes failed
(`aatm omop failures output/ecog` lists them; running again retries them), `1` for an
error, `130` when stopped with Ctrl+C. Running the same command again resumes.

## Extraction settings

`max_concurrency`, `settings` (generation settings of the
extraction model, e.g. `{"temperature": 0}`) and `attribute_labeller` (a
[decision model](../reference/decision_models.md) labelling attributes) are arguments of
the pipeline. Any other `SimpleLLMExtractor` argument, such as `instructions` or
`on_error`, is passed through too; a misspelt name raises a `TypeError` that lists the
valid ones. With `extractor=` (an extractor you built yourself), set them on that
extractor instead: the pipeline raises rather than ignore them.

## Check the plan first

`pipeline.plan("notes.parquet", "output/ecog")` is a dry
run: no model call. It prints the settings the run would use and where each came from,
the notes left to do, the calls, tokens, cost and time a pilot run predicts, and
warnings with a fix, such as a low `max_concurrency` with a hosted API. Which settings
matter for your case, and which you can leave alone, is in
[Tuning large OMOP runs](tuning_omop_runs.md).

## Sources

A CSV or Parquet file, a folder of Parquet files, a DataFrame, an iterable
of `ClinicalNote` objects or dicts, or a database query:

```python
import sqlite3
from aatm.omop import notes_from_sql

con = sqlite3.connect("cdm.db")  # or a SQLAlchemy engine for other databases
source = notes_from_sql(
    "SELECT note_id, note_text AS text, person_id, note_date, visit_occurrence_id "
    "FROM note WHERE note_date >= '2024-01-01'",
    con,
)
summary = await pipeline.run(source, "output/ecog")
```

From the command line, `--sql` takes the query and `--db` the database's SQLAlchemy
URL (`sqlite:///cdm.db` works without SQLAlchemy; other databases need it and their
driver):

```bash
aatm omop run output/ecog --sql "SELECT note_id, note_text AS text, person_id FROM note"   --db postgresql+psycopg://user@host/cdm --definitions ecog_performance_status   --model openai/gpt-5-mini
```

Columns that are not `ClinicalNote` fields are ignored. Every note needs a unique
**`note_id`**: it is how a resumed run knows what is done.

## Reading the output

Each table, and the provenance, issues and mappings of its rows,
is a folder of Parquet files:

```python
run = OmopRun("output/ecog")
run.table("measurement")    # a DataFrame
run.provenance()            # note_id, entity_id and span of each row (table, row_id)
run.issues()
run.mappings()
run.attribute_reviews()     # labels a decision model could not confirm, if you use one
run.failures()              # notes that could not be extracted or built, and why
run.status()                # {"built": 9850, "failed": 3}
run.dataset("measurement")  # a pyarrow Dataset, to filter without loading it all
```

From the command line, `aatm omop status output/ecog` prints the notes built and
failed, the rows per table, the issues by reason and the last run's summary, and
`aatm omop failures output/ecog` the failed notes with their error (both take
`--json`).

DuckDB, Polars and Spark read the folders directly, e.g.
`SELECT * FROM 'output/ecog/measurement/*.parquet'`.

## Resuming

Run the same call again. Notes already built are skipped; notes extracted
but not yet written are built from their stored extraction result, without another
model call. Each extraction result is stored as soon as it comes back, so a run that
stops loses none of the calls it paid for. A note that fails (an API error once retries are spent, a result that
cannot be built) is recorded with its error and the run goes on; the next run tries it
again, up to `max_attempts` times in all (3 by default).

## Changing the mappings

Extraction results are stored apart from the tables
(in `output/ecog/ledger.db`). After editing an `omop` block, rebuild the tables from
them, with no extraction calls:

```python
await pipeline.rebuild("output/ecog")
```

```bash
aatm omop rebuild output/ecog --definitions ./ecog_performance_status.yaml
```

`aatm omop run` saves the options it used in `output/ecog/options.yaml`, and
`rebuild` starts from them: give only what changed, such as the definitions or
`--coder agent`.

A run refuses to add rows built with other mappings to an existing folder until you
rebuild, and refuses a folder made with other entity definitions or another extractor:
use a new folder for those.

## Watching a long run

A progress line is logged every `log_every` seconds (60 by
default), when a chunk is done:

```text
Progress: 12,400 of 100,000 notes done (12.4%) in 41m10s, 301 notes/min; about 4h17m left;
12,380 built, 0 skipped, 20 failed; 31,950 LLM calls, 48,210,330 tokens; cost 14.2031;
cache hits 63%
```

The time left needs the number of notes in the source. It is read from Parquet files,
DataFrames and lists; for a CSV file or a query, pass `total=`. Logs are shown with
`aatm.logs.configure_logging()`.

At the end, the summary breaks the run down by stage (extraction, coding and
building): notes or texts handled, time, errors, LLM calls and tokens per model.
Give your models' prices, per million input and output tokens, to estimate the cost:

```python
summary = await pipeline.run(
    "notes.parquet",
    "output/ecog",
    prices={"openai/gpt-5-mini": (0.25, 2.00)},  # check your provider's current prices
)
summary.stages["extraction"].usage.by_model  # calls and tokens per model
summary.stages["coding"].cost                # coding is the agent's calls, if any
summary.cost
```

Each `run` and `rebuild` also saves its summary as JSON in `output/ecog/runs/`. The
file is updated at every progress line, so you can follow a run from another process,
and it is kept with `completed: false` and the error when a run stops.
`OmopRun("output/ecog").summaries()` reads them back, oldest first.

## Batch extraction, about half the price

For a large run that can wait, pass
`batch=True` to extract through the provider's batch endpoint (OpenAI, Anthropic and
Gemini). Batch jobs cost about 50% of live calls and return within 24 hours, often
much sooner:

```python
summary = await pipeline.run(
    "notes.parquet",
    "output/ecog",
    chunk_size=1000,   # one batch job per chunk
    batch=True,
    poll_every=300,    # seconds between status checks
    max_batches=20,    # jobs pending at once
)
OmopRun("output/ecog").batches()  # the jobs: id, status, notes
```

From the command line: `aatm omop run ... --batch --chunk-size 1000 --poll-every 300`.

Only extraction is batched: coding stays live. Each job is recorded in the ledger as
soon as it is submitted, so you can stop the run (or lose the machine) while jobs are
pending: the next run, with or without `batch=True`, collects them instead of
submitting the notes again. Requests that fail within a job are recorded as failed
notes and tried again by the next run, live or in a batch. Batch calls are counted as
`<model>@batch` in the usage, and priced at half the model's price unless `prices`
has an entry for that id. Gemini takes at most about 20 MB of prompts per job: lower
`chunk_size` if it refuses one.

## Several processes or machines

Split the dataset into shards that run
independently.

!!! note "When sharding helps"
    With a hosted API, a single process is not the limit: it spends its time waiting
    for the model, and the provider's rate limit applies to all processes together.
    Raise `max_concurrency` and the rate limits instead: one process keeps that many
    calls in flight all the time, whatever the `chunk_size`. Shard on one machine when a single process is busy (a
    fast or local model, local embedding retrieval), or to spread a run over several
    machines or cluster jobs. See the
    [sharding benchmark](https://github.com/precisiondata/aatm/tree/main/benchmarks/sharding).
 A note's shard is a stable hash of its `note_id`, so every worker finds
its share of the source on its own. On one machine, `run_sharded` runs the shards in
separate processes:

```python
# my_pipeline.py: worker processes import the factory, so define it in a module
from aatm.omop import OmopExtractionPipeline

def make_pipeline():
    return OmopExtractionPipeline(
        "ecog_performance_status", model_id="openai/gpt-5-mini", max_concurrency=16,
        cache=True,  # one cache file, shared by the processes
    )
```

```python
from aatm.llm import set_rate_limits
from aatm.omop import OmopRun, run_sharded
from my_pipeline import make_pipeline

if __name__ == "__main__":
    set_rate_limits("openai", requests_per_minute=5000)  # divided among the processes
    summary = run_sharded(
        make_pipeline, "notes.parquet", "output/ecog", num_shards=4, chunk_size=200
    )
    run = OmopRun("output/ecog")  # reads every shard as one run
    run.table("measurement")
```

Each shard is an ordinary run in its own folder (`output/ecog/shard-000-of-004/`...),
with its own ledger: it resumes, retries and rebuilds on its own, and
`pipeline.rebuild("output/ecog")` rebuilds them all. Calling `run_sharded` again
resumes every shard. A folder keeps the number of shards it was split into. If a shard
stops with an error, the others finish, and `ShardedRunError` reports it with the
combined summary.

Rate limits apply per process: `run_sharded` divides the limits you set among the
processes running at once (`processes`, all shards by default). Pass the source as a
path, a DataFrame or a list, which every process reads, keeping its own notes, or, for
a database, as a function that opens its own connection and returns `notes_from_sql`.
A query is filtered in the database, so each shard fetches only its notes:

```python
import sqlite3
from aatm.omop import notes_from_sql

def note_source():  # in my_pipeline.py, next to make_pipeline
    con = sqlite3.connect("cdm.db")
    return notes_from_sql("SELECT note_id, note_text AS text, person_id FROM note", con)

run_sharded(make_pipeline, note_source, "output/ecog", num_shards=4)
```

The filter is integer arithmetic on the note id, cast to `BIGINT`; for MySQL, pass
`notes_from_sql(..., cast="SIGNED")`. `shard_condition(index, count)` returns it, to
filter a query yourself.

On several machines (or jobs of a scheduler), run one shard on each with
`shard=(index, count)`, then gather the folders under one parent and read it with
`OmopRun`:

```python
from aatm.omop import shard_dir

index = int(os.environ["SHARD_INDEX"])
await pipeline.run("notes.parquet", shard_dir("output/ecog", index, 8), shard=(index, 8))
```

From the command line, `--shard <index>/<count>` writes to the shard's folder under
the output folder, e.g. as a SLURM array job:

```bash
aatm omop run --config run.yaml --shard "$SLURM_ARRAY_TASK_ID/8"
aatm omop status output/ecog   # every shard: notes built and failed, issues, last summary
```

Each machine keeps its own coding cache: entries are not shared across machines.

## Row ids

The primary keys of a run are derived from the note:
`note_id * 65536 + n`, where `n` is the position of the mention in the note. The same
note gets the same ids in any chunk, after a resume or a rebuild, and in separate runs
over parts of a dataset, such as shards; ids never repeat across tables. They need a 64-bit (`BIGINT`)
column. `OmopTableBuilder.build(..., row_ids="note")` numbers rows the same way.

!!! tip "Speed and cost"
    Most runs need no throughput setting: a run keeps 16 extraction calls in flight
    with a hosted API (4 with a local server), whatever the `chunk_size`. Check what a
    run would do, and what it would cost from a pilot, with `pipeline.plan(...)`; see
    [Tuning large OMOP runs](tuning_omop_runs.md) for which settings matter in your
    case.

    While a chunk is built (coded and written), extraction goes on, so the slower of
    the two stages sets the pace. This pays off when coding takes about as long as
    extraction, e.g. with the [agent](coding_text.md#searching-with-an-agent); without a mapper,
    building takes milliseconds and extraction is all the time. A few chunks are in
    memory at most: the one being built, up to two waiting, and the notes being
    extracted.


# Command line reference

Every `aatm` command, with its arguments and options. This page is generated from the
CLI itself (`scripts/gen_cli_docs.py`), so it matches `aatm <command> --help` for this
version. For how to use the commands together, see the
[high-level guide](../getting-started/high_level.md).

| Workflow | Commands |
|---|---|
| Setup | [`aatm init`](#aatm-init), [`aatm doctor`](#aatm-doctor), [`aatm vector-db check`](#aatm-vector-db-check), [`aatm vector-db upload`](#aatm-vector-db-upload) |
| Clinical notes to OMOP tables | [`aatm omop run`](#aatm-omop-run), [`aatm omop status`](#aatm-omop-status), [`aatm omop failures`](#aatm-omop-failures), [`aatm omop rebuild`](#aatm-omop-rebuild), [`aatm omop pilot log`](#aatm-omop-pilot-log), [`aatm omop pilot compare`](#aatm-omop-pilot-compare), [`aatm cache info`](#aatm-cache-info), [`aatm cache clear`](#aatm-cache-clear) |
| Entity definitions | [`aatm schema`](#aatm-schema) |
| Terminology mapping | [`aatm map`](#aatm-map), [`aatm search-ui`](#aatm-search-ui), [`aatm serve`](#aatm-serve) |

Global options, such as `--log-level`, go before the command name:
`aatm --log-level DEBUG omop run ...`.

`aatm omop run` exits with `0` when every note is built, `2` when the run completed
but some notes failed, `1` on an error and `130` when stopped with Ctrl+C, so
schedulers can act on it.

## `aatm`

**Usage**:

```console
$ aatm [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `-l, --log-level TEXT`: Log level: DEBUG, INFO, WARNING, ERROR or CRITICAL. Logs go to stderr.  [env var: AATM_LOG_LEVEL; default: (INFO)]
* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `init`: Set up the OMOP vocabularies: unpack an...
* `search-ui`: Open an interactive search over the...
* `map`: Map a CSV of source terms...
* `serve`: Serve terminology mapping as a REST API...
* `doctor`: Check whether local models can use a GPU,...
* `schema`: Write the JSON Schema of entity definition...
* `vector-db`: Check the vector database settings or...
* `cache`: Inspect or clear the coding cache...
* `omop`: Run, rebuild and inspect OMOP extraction runs

### `aatm init`

Set up the OMOP vocabularies: unpack an Athena bundle, build the vocabulary database (.aatm/omop.db) and the search index of an embedding model. Run once per project, before any other command.

**Usage**:

```console
$ aatm init [OPTIONS]
```

**Options**:

* `-vd, --vocab-dir PATH`: Path to the OMOP vocabularies directory.
* `-e, --embedding-model TEXT`: Name of the embedding model to use.
* `--embedding-dtype TEXT`: Precision for local embedding models: float32, bfloat16 or float16. Defaults to the precision of the model&#x27;s checkpoint. Half precision is faster and uses less GPU memory, but requires an NVIDIA GPU. Overrides AATM_EMBEDDING_DTYPE.
* `-sv, --standard-vocabs TEXT`: List of standard vocabularies to use. Available options: [&#x27;LOINC&#x27;, &#x27;SNOMED&#x27;, &#x27;RxNorm&#x27;].
* `--vector-db TEXT`: Where to build the vector database: &#x27;local&#x27; (ChromaDB on this machine), &#x27;cloud&#x27; (Chroma Cloud, configured with CHROMA_API_KEY), &#x27;pinecone&#x27; (configured with PINECONE_API_KEY) or &#x27;pgvector&#x27; (PostgreSQL, configured with AATM_PG_DSN). Overrides AATM_VECTOR_DB. Defaults to &#x27;local&#x27;.
* `--vocab-zip TEXT`: Athena vocabulary bundle to set up from: the download link Athena emailed, or the path of the downloaded zip. It is unpacked into --vocab-dir (default ./vocabularies).
* `--force`: With --vocab-zip, overwrite existing vocabulary files without asking.
* `--umls-api-key TEXT`: UMLS API key used to add the CPT4 names when the bundle includes CPT4. Read from UMLS_API_KEY when not given.  [env var: UMLS_API_KEY]
* `--help`: Show this message and exit.

### `aatm search-ui`

Open an interactive search over the vocabularies in the browser (Streamlit).

**Usage**:

```console
$ aatm search-ui [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

### `aatm map`

Map a CSV of source terms (SOURCE_TO_CONCEPT_MAP layout) to standard concepts; the result is &lt;output-dir&gt;/mapped_source_concepts.csv. Give the options as flags or in a --task-config-path file.

**Usage**:

```console
$ aatm map [OPTIONS]
```

**Options**:

* `-t, --task-config-path PATH`: A YAML or JSON file of the options below (names with underscores)
* `-i, --input-file TEXT`: The CSV of source terms, with the SOURCE_TO_CONCEPT_MAP columns
* `-o, --output-dir TEXT`: Where to write mapped_source_concepts.csv  [default: (output)]
* `-tr, --translator-id TEXT`: Translate the terms first: empty-translator or gemini-2.5-flash  [default: (no translation)]
* `-r, --retriever-id TEXT`: The embedding model whose index to search, e.g. text-embedding-3-small  [default: (the one aatm init built)]
* `-s, --selector-id TEXT`: How the concept is chosen: first-result-selector, an LLM (e.g. gpt-5-mini), the agent (e.g. agent-gpt-5-mini) or decision-&lt;model&gt;  [default: (first-result-selector)]
* `-rr, --reranker-id TEXT`: Reorder the candidates: bm25-reranker or qwen3-reranker-0.6b/-4b/-8b  [default: (no reranking)]
* `-b, --batch-size INTEGER`: Terms mapped together  [default: (100)]
* `-rl, --rate-limit INTEGER`: Terms mapped per minute at most  [default: (no limit)]
* `-l, --limit-to INTEGER`: Map only the first N terms, e.g. for a test  [default: (all)]
* `--help`: Show this message and exit.

### `aatm serve`

Serve terminology mapping as a REST API (FastAPI), configured by .aatm/api_config.yaml

**Usage**:

```console
$ aatm serve [OPTIONS]
```

**Options**:

* `-h, --host TEXT`: Host  [default: 0.0.0.0]
* `-p, --port TEXT`: Port  [default: 8000]
* `-m, --mode [prod|dev]`: Serving mode: &#x27;prod&#x27; or &#x27;dev&#x27;  [default: dev]
* `--reload`: Enables hot reload. Compatible with &#x27;dev&#x27; mode only.
* `-w, --workers INTEGER`: Number of workers
* `-r, --rate-limit INTEGER`: Maximum number of documents allowed per minute
* `-b, --batch-size INTEGER`: Batch size  [default: 100]
* `--help`: Show this message and exit.

### `aatm doctor`

Check whether local models can use a GPU, and print how to enable it.

**Usage**:

```console
$ aatm doctor [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

### `aatm schema`

Write the JSON Schema of entity definition files, for editor support offline.

**Usage**:

```console
$ aatm schema [OPTIONS] [DIRECTORY]
```

**Arguments**:

* `[DIRECTORY]`: Directory to write entity_definitions.schema.json to.  [default: .]

**Options**:

* `--help`: Show this message and exit.

### `aatm vector-db`

Check the vector database settings or upload a local index to Chroma Cloud, Pinecone or PostgreSQL

**Usage**:

```console
$ aatm vector-db [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `check`: Show where the vector database lives and...
* `upload`: Copy a local vector database to Chroma...

#### `aatm vector-db check`

Show where the vector database lives and test the connection

**Usage**:

```console
$ aatm vector-db check [OPTIONS]
```

**Options**:

* `--vector-db TEXT`: &#x27;local&#x27;, &#x27;cloud&#x27;, &#x27;pinecone&#x27; or &#x27;pgvector&#x27;. Overrides AATM_VECTOR_DB.
* `--help`: Show this message and exit.

#### `aatm vector-db upload`

Copy a local vector database to Chroma Cloud, Pinecone or PostgreSQL without re-embedding

**Usage**:

```console
$ aatm vector-db upload [OPTIONS]
```

**Options**:

* `-e, --embedding-model TEXT`: Name of the embedding model whose index to upload.  [required]
* `--local-dir PATH`: Local vector database directory. Defaults to the model&#x27;s path under .aatm/chroma_vector_dbs.
* `--batch-size INTEGER`: Rows per write. Capped at Chroma Cloud&#x27;s maximum batch size; split into requests within Pinecone&#x27;s limits.
* `--to TEXT`: Destination: &#x27;cloud&#x27; (Chroma Cloud), &#x27;pinecone&#x27; or &#x27;pgvector&#x27;.  [default: cloud]
* `--ids-file PATH`: Only upload the rows whose IDs this file lists, one per line, e.g. a subset that fits a free plan.
* `--help`: Show this message and exit.

### `aatm cache`

Inspect or clear the coding cache (.aatm/coding_cache.db)

**Usage**:

```console
$ aatm cache [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `info`: Show the entries, size and coders of the...
* `clear`: Delete the cached results and shrink the file

#### `aatm cache info`

Show the entries, size and coders of the cache

**Usage**:

```console
$ aatm cache info [OPTIONS]
```

**Options**:

* `-p, --path PATH`: The cache file  [default: (.aatm/coding_cache.db)]
* `--help`: Show this message and exit.

#### `aatm cache clear`

Delete the cached results and shrink the file

**Usage**:

```console
$ aatm cache clear [OPTIONS]
```

**Options**:

* `-p, --path PATH`: The cache file  [default: (.aatm/coding_cache.db)]
* `--coder TEXT`: Only delete this coder&#x27;s entries (see &#x27;aatm cache info&#x27;)
* `-y, --yes`: Do not ask for confirmation
* `--help`: Show this message and exit.

### `aatm omop`

Run, rebuild and inspect OMOP extraction runs

**Usage**:

```console
$ aatm omop [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `run`: Extract notes into OMOP tables, in chunks,...
* `rebuild`: Build the tables of a run again from its...
* `status`: Notes built and failed, issues by reason,...
* `failures`: The notes that failed: stage, error and...
* `pilot`: Read a pilot workspace: versioned runs on...

#### `aatm omop run`

Extract notes into OMOP tables, in chunks, resuming where an earlier run stopped. Exit codes: 0 every note built, 2 some notes failed, 1 error, 130 stopped

**Usage**:

```console
$ aatm omop run [OPTIONS] [SOURCE] [OUTPUT]
```

**Arguments**:

* `[SOURCE]`: The notes: a CSV or Parquet file or a folder of Parquet files (with --sql, the only argument is the output folder)
* `[OUTPUT]`: The output folder

**Options**:

* `--config PATH`: A YAML file of options (names with underscores); flags override it  [default: (none)]
* `-d, --definitions TEXT`: Entity definitions: a predefined name (e.g. condition_occurrence) or a YAML file; repeat for several  [default: (none)]
* `-m, --model TEXT`: The extraction model, e.g. openai/gpt-5-mini  [default: (none)]
* `--sql TEXT`: Read the notes from this query instead of a file (needs --db)  [default: (none)]
* `--db TEXT`: The SQLAlchemy URL of the database of --sql, e.g. sqlite:///cdm.db  [default: (none)]
* `-c, --columns TEXT`: Rename a source column to a note field: &lt;column&gt;=&lt;field&gt;, e.g. note_text=text  [default: (none)]
* `--vocabulary TEXT`: The OMOP vocabulary database built by &#x27;aatm init&#x27;  [default: (.aatm/omop.db)]
* `--coder TEXT`: How &#x27;map&#x27; sources are coded: none (stored as 0), mapper (the terminology mapper) or agent (a concept search agent)  [default: (none)]
* `--coder-model TEXT`: The agent&#x27;s model (--coder agent)  [default: (--model)]
* `--retriever TEXT`: The retriever of the coder, e.g. text-embedding-3-small; &#x27;none&#x27; gives the agent the vocabulary tools only  [default: (the one recorded by &#x27;aatm init&#x27;)]
* `--selector TEXT`: The mapper&#x27;s selector, a registry name (--coder mapper)  [default: (the first result)]
* `--max-tool-calls INTEGER`: Tool calls per text of the agent (--coder agent)  [default: (8)]
* `--min-score FLOAT`: Codes scored lower are stored as 0, for &#x27;map&#x27; sources without a min_score of their own  [default: (none)]
* `--labeller TEXT`: A decision model that labels the attributes of mentions, e.g. openrouter/typesafe/jev-1.13  [default: (none)]
* `--labeller-mode TEXT`: verify: keep a label only when the decision model agrees; replace: the decision model labels instead  [default: (verify)]
* `--min-probability FLOAT`: Labels the decision model is less sure of go to review  [default: (none)]
* `--max-concurrency INTEGER`: Extraction calls in flight at once  [default: (16 for a hosted API, 4 for a local server)]
* `--chunk-size INTEGER`: Notes built and written together  [default: (100)]
* `--max-attempts INTEGER`: Times a failing note is tried, across runs  [default: (3)]
* `--cache / --no-cache`: Code each distinct text once, across chunks and runs (.aatm/coding_cache.db)  [default: (off)]
* `--price TEXT`: A model&#x27;s price per million input and output tokens, for the cost: &lt;model&gt;=&lt;input&gt;,&lt;output&gt;; repeat for several  [default: (none)]
* `--batch / --live`: Extract through the provider&#x27;s batch endpoint: about half the price, within 24 hours  [default: (live)]
* `--poll-every FLOAT`: Seconds between polls of pending batch jobs  [default: (60.0)]
* `--max-batches INTEGER`: Batch jobs pending at once  [default: (20)]
* `--shard TEXT`: Process only shard &lt;index&gt;/&lt;count&gt; (from 0), e.g. 3/8; the output goes to &lt;output&gt;/shard-003-of-008  [default: (the whole dataset)]
* `--limit INTEGER`: Process only the first N notes of the source, e.g. for a pilot  [default: (all)]
* `--total INTEGER`: Notes in the source, for the time left (CSV files and queries)  [default: (counted for Parquet)]
* `--log-every FLOAT`: Seconds between progress logs  [default: (60.0)]
* `--json`: Print the summary as JSON on standard output
* `--help`: Show this message and exit.

#### `aatm omop rebuild`

Build the tables of a run again from its stored extraction results, with no extraction call (after changing the omop blocks or the coder). Options not given are those the run recorded in options.yaml

**Usage**:

```console
$ aatm omop rebuild [OPTIONS] OUTPUT
```

**Arguments**:

* `OUTPUT`: The output folder of the run  [required]

**Options**:

* `--config PATH`: A YAML file of options (names with underscores); flags override it  [default: (none)]
* `-d, --definitions TEXT`: Entity definitions: a predefined name (e.g. condition_occurrence) or a YAML file; repeat for several  [default: (none)]
* `-m, --model TEXT`: The extraction model, e.g. openai/gpt-5-mini  [default: (none)]
* `--vocabulary TEXT`: The OMOP vocabulary database built by &#x27;aatm init&#x27;  [default: (.aatm/omop.db)]
* `--coder TEXT`: How &#x27;map&#x27; sources are coded: none (stored as 0), mapper (the terminology mapper) or agent (a concept search agent)  [default: (none)]
* `--coder-model TEXT`: The agent&#x27;s model (--coder agent)  [default: (--model)]
* `--retriever TEXT`: The retriever of the coder, e.g. text-embedding-3-small; &#x27;none&#x27; gives the agent the vocabulary tools only  [default: (the one recorded by &#x27;aatm init&#x27;)]
* `--selector TEXT`: The mapper&#x27;s selector, a registry name (--coder mapper)  [default: (the first result)]
* `--max-tool-calls INTEGER`: Tool calls per text of the agent (--coder agent)  [default: (8)]
* `--min-score FLOAT`: Codes scored lower are stored as 0, for &#x27;map&#x27; sources without a min_score of their own  [default: (none)]
* `--labeller TEXT`: A decision model that labels the attributes of mentions, e.g. openrouter/typesafe/jev-1.13  [default: (none)]
* `--labeller-mode TEXT`: verify: keep a label only when the decision model agrees; replace: the decision model labels instead  [default: (verify)]
* `--min-probability FLOAT`: Labels the decision model is less sure of go to review  [default: (none)]
* `--chunk-size INTEGER`: Notes built and written together  [default: (1000)]
* `--cache / --no-cache`: Code each distinct text once, across chunks and runs (.aatm/coding_cache.db)  [default: (off)]
* `--price TEXT`: A model&#x27;s price per million input and output tokens, for the cost: &lt;model&gt;=&lt;input&gt;,&lt;output&gt;; repeat for several  [default: (none)]
* `--log-every FLOAT`: Seconds between progress logs  [default: (60.0)]
* `--json`: Print the summary as JSON on standard output
* `--help`: Show this message and exit.

#### `aatm omop status`

Notes built and failed, issues by reason, and the last summary

**Usage**:

```console
$ aatm omop status [OPTIONS] OUTPUT
```

**Arguments**:

* `OUTPUT`: The output folder of a run  [required]

**Options**:

* `--json`: Print the status as JSON
* `--help`: Show this message and exit.

#### `aatm omop failures`

The notes that failed: stage, error and attempts

**Usage**:

```console
$ aatm omop failures [OPTIONS] OUTPUT
```

**Arguments**:

* `OUTPUT`: The output folder of a run  [required]

**Options**:

* `--json`: Print the failures as JSON records
* `--help`: Show this message and exit.

#### `aatm omop pilot`

Read a pilot workspace: versioned runs on a fixed sample of notes (see aatm.omop.Pilot)

**Usage**:

```console
$ aatm omop pilot [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `log`: List the versions and what changed in each
* `compare`: Compare two versions (by default the...

##### `aatm omop pilot log`

List the versions and what changed in each

**Usage**:

```console
$ aatm omop pilot log [OPTIONS]
```

**Options**:

* `-p, --path PATH`: The pilot workspace folder  [default: pilot]
* `--diff`: Show the diff of the definitions
* `--help`: Show this message and exit.

##### `aatm omop pilot compare`

Compare two versions (by default the latest two)

**Usage**:

```console
$ aatm omop pilot compare [OPTIONS] [A] [B]
```

**Arguments**:

* `[A]`: The earlier version, e.g. v002
* `[B]`: The later version, e.g. v003

**Options**:

* `-p, --path PATH`: The pilot workspace folder  [default: pilot]
* `--help`: Show this message and exit.

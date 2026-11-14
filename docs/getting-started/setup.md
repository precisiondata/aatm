# Set up the vocabularies

Every AATM workflow reads the [OMOP vocabularies](https://athena.ohdsi.org/search-terms/start):
the terminology mapper searches them, the extraction pipeline codes its rows with them.
Setting them up is done once per project, with one command: `aatm init`.

It builds two things in a local `.aatm/` folder:

| What | File | Used by |
|---|---|---|
| The vocabulary database (concepts, synonyms, relationships) | `.aatm/omop.db` | Every workflow: code lookups, checks of every concept id, the agent's lexical search |
| A search index of concept names and synonyms, for one embedding model | `.aatm/chroma_vector_dbs/<model>/` (or Chroma Cloud, Pinecone, PostgreSQL) | The terminology mapper and the agent, to search by meaning |

---

## 1. Request the vocabularies on Athena

AATM cannot ship the vocabularies because several of them have their own licences. You
request them from [Athena](https://athena.ohdsi.org/vocabulary/list), which has no API
for this, so this step is done in the browser:

1. Log in to [Athena](https://athena.ohdsi.org/vocabulary/list) and open **Download**.
2. Select the vocabularies you need and accept the licences of the restricted ones.
   **SNOMED, LOINC, UCUM and RxNorm** cover every predefined entity definition.
3. Click **Download vocabularies** and name the bundle. Athena emails you a link to a
   zip file when it is ready, which can take a while.

## 2. Run `aatm init`

Pass it the link from Athena's email, or the zip if you already downloaded it:

```bash
aatm init --vocab-zip "https://athena.ohdsi.org/api/v1/vocabularies/zip/<bundle-id>"
```

```bash
aatm init --vocab-zip ~/Downloads/vocabulary_download_v5_<bundle-id>.zip
```

`aatm init` then:

1. downloads the bundle into `.aatm/downloads` (resuming if the connection drops) and
   unpacks it into `./vocabularies` (or `--vocab-dir`), asking before it overwrites
   files (`--force` skips the question);
2. checks that the required files (`CONCEPT.csv`, `CONCEPT_RELATIONSHIP.csv`,
   `CONCEPT_SYNONYM.csv`, `VOCABULARY.csv`, `DOMAIN.csv`) are there, and lists the
   vocabularies and versions in the bundle;
3. asks where to build the search index: a local ChromaDB, Chroma Cloud, Pinecone or
   PostgreSQL with pgvector (see [Vector database](env.md#vector-database)). Each option
   lists the models already built in it, and an option that misses a setting or a package
   is greyed out with what to set or install. `AATM_VECTOR_DB` is preselected;
   `--vector-db` skips the question;
4. builds the vocabulary database, `.aatm/omop.db`;
5. lists the search indexes already built there, asks which **embedding model** (built
   ones are marked) and **standard vocabularies** to index, and builds the search index;
6. records the model it built in `.aatm/config.yaml`, so every retriever, mapper and
   agent uses it unless told otherwise;
7. adds `.aatm` to your `.gitignore`.

To answer the questions up front, for scripts and CI:

```bash
aatm init \
  --vocab-zip "https://athena.ohdsi.org/api/v1/vocabularies/zip/<bundle-id>" \
  --vector-db local \
  --embedding-model text-embedding-3-small \
  --standard-vocabs SNOMED --standard-vocabs LOINC --standard-vocabs RxNorm
```

If you unpacked the zip yourself, put the CSV files in `./vocabularies` and run
`aatm init`, or point to them with `--vocab-dir path/to/vocabularies`.

### Which embedding model?

| Model | Runs | Notes |
|---|---|---|
| `text-embedding-3-small`, `text-embedding-3-large` | OpenAI API (`OPENAI_API_KEY`) | No GPU needed; building the index is paid per token |
| `gemini-embedding-001` | Google API (`GOOGLE_API_KEY`) | No GPU needed; building the index is paid per token |
| `embeddinggemma-300M` | On this machine | Free; a GPU makes the build much faster (check with `aatm doctor`) |
| `qwen3-06B`, `qwen3-4B` | On this machine | Free; larger models, a GPU is strongly recommended |

You can build more than one: run `aatm init --embedding-model <other>` again. The last
one built becomes the default. `aatm vector-db check` shows which ones are built and
which is the default.

### CPT4

When the bundle includes CPT4, its concept names must be downloaded with your own UMLS
licence. Install [Java](https://adoptium.net) and pass a
[UMLS API key](https://uts.nlm.nih.gov/uts/profile) with `--umls-api-key` or the
`UMLS_API_KEY` environment variable. Without a key, CPT4 concepts are kept without
names.

### Sharing the index with a team

By default the search index is a local ChromaDB folder of several GB. To build it once
and share it, or to search from machines without the disk space, build it in Chroma
Cloud, Pinecone or a PostgreSQL server with `--vector-db cloud`, `pinecone` or
`pgvector`. See
[Vector database](env.md#vector-database).

---

## 3. Set your API keys

Put the keys of the providers you use in a `.env` file at the project root; AATM loads
it automatically:

```env
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=...
```

See [Environment variables](env.md) for every setting.

---

## Troubleshooting

| You see | Fix |
|---|---|
| The vocabulary directory does not exist | Put the CSV files in `./vocabularies`, pass `--vocab-dir`, or use `--vocab-zip` |
| `--vocab-zip` says the link returned a web page or has expired | Request the bundle again on Athena and use the new link, or download the zip in your browser while logged in and pass its path |
| `--vocab-zip` says files already exist | Confirm the overwrite, pass `--force`, or unpack into another `--vocab-dir` |
| A warning about CPT4 | Install Java and pass `--umls-api-key` (see [CPT4](#cpt4)) |
| The embedding model is unsupported | Use one of the names in [Which embedding model?](#which-embedding-model) |
| `No vector database for '<model>'` later, when mapping | That model's index was not built: use one `aatm vector-db check` lists, or build it with `aatm init --embedding-model <model>` |
| The build is slow with a local model | Run `aatm doctor`: it says whether your GPU is used and how to fix it ([Installation](installation.md#gpu-acceleration-optional)) |

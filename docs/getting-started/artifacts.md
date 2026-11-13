# Files AATM creates

## The `.aatm/` folder

`aatm init` creates a `.aatm/` folder in the project and adds it to `.gitignore`. Every
command and class looks for it in the current directory, so run your scripts from the
project root.

```text
.aatm/
├── omop.db                 # the vocabulary database (concepts, synonyms, relationships)
├── config.yaml             # the embedding model and vector database built, used as defaults
├── chroma_vector_dbs/      # one local search index per embedding model
│   └── <model>/
├── datasets/               # the expressions the indexes are built from
├── downloads/              # Athena bundles fetched by aatm init --vocab-zip
└── coding_cache.db         # coding results, with cache=True (OMOP runs)
```

With `AATM_VECTOR_DB=cloud` or `AATM_VECTOR_DB=pinecone`, the search indexes live in
Chroma Cloud or Pinecone instead of `chroma_vector_dbs/` (see
[Environment variables](env.md#vector-database-local-chroma-cloud-or-pinecone)).

`aatm init` also unpacks the vocabulary CSV files into `./vocabularies` (or
`--vocab-dir`). They can be deleted once `omop.db` is built, unless you want to rebuild
it.

## Terminology mapping output

`aatm map` and `TerminologyMapper.map` write:

```text
output/
└── mapped_source_concepts.csv
```

## OMOP run output

`pipeline.run(source, "output/run")` and `aatm omop run ... output/run` write:

```text
output/run/
├── measurement/            # one folder of Parquet files per table
├── condition_occurrence/
├── provenance/             # the note and text span of each row
├── issues/                 # what could not be filled, and why
├── mappings/               # how each text was coded
├── attribute_reviews/      # labels a decision model could not confirm, with a labeller
├── ledger.db               # every note's status and extraction result: resume and rebuild
├── options.yaml            # the options of aatm omop run, reused by aatm omop rebuild
└── runs/                   # one JSON summary per run and rebuild
```

A sharded run has one such folder per shard (`shard-000-of-004/`...), read together by
`OmopRun("output/run")`. A `Pilot` workspace holds one run folder per version
(`v001-<hash>/`...), the sample, and a shared `coding_cache.db`.

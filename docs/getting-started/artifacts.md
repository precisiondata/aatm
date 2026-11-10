# Project artifacts created locally

AATM uses a local `.aatm/` directory to store generated assets such as:

```text
.aatm/
├── omop.db
├── datasets/
└── chroma_vector_dbs/
```

With `AATM_VECTOR_DB=cloud`, the vector database lives in Chroma Cloud instead of `chroma_vector_dbs/` (see [Environment variables](env.md#vector-database-location-chroma-cloud)).

Typical outputs are written to:

```text
output/
```
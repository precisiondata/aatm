# Project artifacts created locally

AATM uses a local `.aatm/` directory to store generated assets such as:

```text
.aatm/
├── omop.db
├── datasets/
└── chroma_vector_dbs/
```

With `AATM_VECTOR_DB=cloud` or `AATM_VECTOR_DB=pinecone`, the vector database lives in Chroma Cloud or Pinecone instead of `chroma_vector_dbs/` (see [Environment variables](env.md#vector-database-local-chroma-cloud-or-pinecone)).

Typical outputs are written to:

```text
output/
```
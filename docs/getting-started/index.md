# Getting Started

Welcome to **AATM** — the **Any-to-Any Terminology Mapper**.

AATM helps you map source terminology expressions to standardized concepts through a modular pipeline composed of:

- **translation**
- **retrieval**
- **reranking**
- **selection**

It is designed so you can start simple and progressively swap components as needed.

---

## What AATM does

At a high level, AATM takes source concepts such as:

- local clinical terms
- source vocabulary descriptions
- free-text expressions

and maps them to standardized concepts using a pipeline like this:

```text
source concept
  -> translator
  -> retriever
  -> reranker
  -> selector
  -> mapped concept
```

The default workflow is centered on OMOP-style terminology mapping.
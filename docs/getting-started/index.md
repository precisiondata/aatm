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

---

## Information extraction

AATM can also extract structured information from clinical documents, such as conditions, drugs, dates and routes, before it is mapped:

```text
clinical note
  -> numbered lines
  -> LLM extraction (guided by entity definitions)
  -> validation against the note
  -> validated entities and records
```

See the [Information extraction quick start](quick_start_extraction.md), and
[Writing entity definitions](entity_definitions.md) to describe your own entities in
YAML.
# Any-to-Any Terminology Mapping

**AATM** is an open-source Python library that turns healthcare text into standard
concepts of the [OMOP vocabularies](https://athena.ohdsi.org/search-terms/start). It
does three jobs, which build on each other:

| | You have | You get | Start with |
|---|---|---|---|
| **Clinical notes to OMOP tables** | Free-text notes | Rows of the OMOP CDM `measurement`, `observation`, `condition_occurrence`, `drug_exposure` and `device_exposure` tables, each traceable to the words it came from | [Quick start](getting-started/quick_start_omop.md) |
| **Information extraction** | Free-text notes | Conditions, drugs, dates, scores... found by an LLM and checked against the text | [Quick start](getting-started/quick_start_extraction.md) |
| **Terminology mapping** | Local codes and descriptions, in any language | A `SOURCE_TO_CONCEPT_MAP` to standard concepts | [Quick start](getting-started/quick_start_cli.md) |

## In a few lines

=== "Command line"

    ```bash
    pip install aatm
    aatm init --vocab-zip "<the link Athena emailed you>"
    aatm omop run notes.parquet output/ \
      --definitions condition_occurrence --definitions drug_exposure \
      --model openai/gpt-5-mini --coder agent --cache
    ```

=== "Python"

    ```python
    from aatm.omop import AgenticCoder, OmopExtractionPipeline

    pipeline = OmopExtractionPipeline(
        ["condition_occurrence", "drug_exposure"],
        model_id="openai/gpt-5-mini",
        coder=AgenticCoder("openai/gpt-5-mini"),
        cache=True,
    )
    summary = await pipeline.run("notes.parquet", "output/")
    ```

The run reads the notes in chunks, writes each table as Parquet files, and resumes where
it stopped. Every row keeps its provenance (the note and the span of text it came
from), and everything that could not be filled is reported with a reason.

## Pick your level

AATM is built in layers. Use the highest one that does what you need, and go down a
level only for the part you want to change:

| Level | What you write | Read |
|---|---|---|
| **Command line** | A YAML file of options; `aatm init`, `aatm omop run`, `aatm map` | [High-level guide](getting-started/high_level.md) |
| **Pipelines** | A few lines of Python: `OmopExtractionPipeline`, `Pilot`, `TerminologyMapper` | [High-level guide](getting-started/high_level.md) |
| **Building blocks** | Your own LLM client, coder, selector or vocabulary source, plugged into the pipelines | [Low-level guide](getting-started/low_level.md) |

What to extract, and where it goes in the tables, is never code: it is a YAML file of
[entity definitions](getting-started/entity_definitions.md), with eight predefined ones
to start from.

## How it works

Every job ends with the same step: connecting a text to a **standard concept**. The OMOP
vocabularies are a community-built network that links the world's health terminologies
and classifications through standard concepts. Once a text is linked to one, every
terminology connected to it is one step away.

![Standard vocabularies](./assets/std_vocabs_diagram.png)
/// caption
Standard vocabularies connect the world's terminologies
///

AATM makes that link in three steps: an optional **translation**, a **retrieval** of
candidate concepts from a search index of every concept name and synonym, and a
**selection** of the best one, by the closest match, an LLM, a decision model, or an
agent that searches the vocabularies with tools until it finds the concept.

![Terminology mapping pipeline](./assets/aatm-pipeline-diagram.jpg)
/// caption
Terminology mapping pipeline
///

For notes, an LLM first **extracts** the mentions your entity definitions describe.
AATM **validates** each one against the note, so only text that is really there
survives, **codes** it with the step above, and **builds** the table rows as the
[OMOP CDM conventions](https://ohdsi.github.io/CommonDataModel/cdm55.html) ask.

## Design principles

**Modular.** Translation, retrieval, reranking, selection, extraction, coding and table
building are separate components with small interfaces. Each can be used, tested and
replaced on its own.

**Extensible.** New embedding models, vector databases, LLM providers, selectors and
coders plug in without changing the rest. Most are a class with one method.

**Simple to start, open to tune.** The defaults run a full pipeline from one command;
every default can be overridden when your data needs it.

**Auditable.** Nothing an LLM returns is trusted as is: spans are grounded in the note,
codes are checked against the vocabularies, and every gap is reported rather than
filled with a guess.

## Where next

- New here? [Start here](getting-started/index.md), then [install](getting-started/installation.md).
- Upgrading? See the [release notes](release_notes.md).
- Looking for a class or a function? See the API reference.

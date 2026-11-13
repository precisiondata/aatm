# Coding extracted text

This page is about the `map` sources of an [`omop` block](omop_tables.md#reference-the-omop-block):
how extracted text gets a standard concept id, how to choose a coder, and how to keep
its results across runs.

Some concepts never change (body weight is always *Body weight*), but many depend on
what the note says: the name of a lab test, a condition. A `map` source sends the
extracted text to the same `TerminologyMapper` used for
[terminology mapping](quick_start_python_api.md), which searches the vocabularies and
picks the best standard concept.

The predefined `lab_results` definition codes the test name this way:

```yaml
      measurement_concept_id:
        from: text
        map: {}          # any standard concept of the column's domain (Measurement)
```

The YAML says **what** to code. Your code chooses **how**, by passing a **coder**,
because that involves search models, LLMs and API keys. `coder=` takes a configured
`TerminologyMapper`, the [agent](#searching-with-an-agent) (`AgenticCoder`), or a
[coder of your own](#writing-your-own-coder):

```python
import asyncio

from aatm.omop import ClinicalNote, OmopExtractionPipeline
from aatm.selectors import OpenAILLMSelector
from aatm.terminology_mapper import TerminologyMapper

mapper = TerminologyMapper(
    retriever="text-embedding-3-small",          # optional: defaults to the model built by `aatm init`
    selector=OpenAILLMSelector("gpt-5-mini"),    # an LLM picks among the candidates
)
mapper.retriever.top_k = 30                      # how many candidates it sees

pipeline = OmopExtractionPipeline("lab_results", model_id="openai/gpt-5-mini", coder=mapper)
notes = [ClinicalNote(text="Labs 03/15/2025: Hb 11.2 g/dL, creatinine 1.4 mg/dl.", person_id=10)]
tables = asyncio.run(pipeline(notes))
tables.mappings_dataframe()
```

`mappings_dataframe()` has one row per coded text, so you can review every choice:

| row_id | column | text | method | concept_id | concept_name | score | stored |
|---|---|---|---|---|---|---|---|
| 1 | measurement_concept_id | Hb | map | 3000963 | Hemoglobin [Mass/volume] in Blood | 0.30 | 3000963 |
| 1 | unit_concept_id | g/dL | lookup | 8713 | gram per deciliter | | 8713 |

How it works:

- **The search is limited to a domain, not a vocabulary.** By default it is the domain
  the column requires (Measurement here). `map: {domain: Condition}` sets it
  explicitly, and `map: {vocabularies: [LOINC]}` narrows the search further if your
  data must use one vocabulary.
- **Each distinct text is coded once per run**: a hundred notes mentioning "Hb" make
  one mapper query.
- **`min_score`** (`map: {min_score: 0.8}`) stores `0` and reports `low_confidence`
  below that score. Scores depend on the retriever, so look at real scores in
  `mappings_dataframe()` before choosing a threshold.
- **Every concept is checked against the vocabulary database.** A concept that is not
  standard is stored as `0` and reported `unmapped`; one of another domain is reported
  `domain_mismatch`. The indexes `aatm init` builds hold only standard concepts, but a
  retriever over an index of your own may not.

!!! note "`mapper=` is now `coder=`"
    Earlier versions called this argument `mapper=`. It still works, with a
    `DeprecationWarning`, and will be removed in a later minor version.

!!! warning "Review the mappings"
    Short names are ambiguous. In a small test with nine lab results, the default
    selector, which takes the closest match, coded only 2 test names correctly; with
    `OpenAILLMSelector` choosing among 30 candidates, 7 of 9 were correct. Without a
    vocabulary filter, some tests map to general SNOMED procedures such as *Potassium
    measurement* rather than the more specific LOINC *Potassium [Moles/volume] in Serum
    or Plasma*; both are valid standard concepts.

## Searching with an agent

The mapper makes one search with the text alone, then picks from what it found. That
fails in two ways: the right concept is not among the candidates ("creatinine" brings
back urine and 24-hour tests, but not serum creatinine), or the text does not say which
variant it is, while the rest of the note does (`mg/dL` means a mass concentration,
`mmol/L` a molar one).

`AgenticCoder` codes each text with an LLM agent instead. The agent gets the text
**with its context**, and searches the vocabularies with tools until it finds the
concept or concludes there is none:

```python
import asyncio

from aatm.omop import AgenticCoder, ClinicalNote, OmopExtractionPipeline

coder = AgenticCoder(
    "openai/gpt-5-mini",                 # any aatm.llm model id
    retriever="text-embedding-3-small",  # optional: defaults to the model built by `aatm init`
    max_tool_calls=8,                    # per text
)
pipeline = OmopExtractionPipeline("lab_results", model_id="openai/gpt-5-mini", coder=coder)
notes = [ClinicalNote(text="Labs 03/15/2025: Hb 11.2 g/dL, creatinine 1.4 mg/dl.", person_id=10)]
tables = asyncio.run(pipeline(notes))
tables.mappings_dataframe()[["text", "concept_name", "score", "rationale", "trace", "usage"]]
```

**The context.** The builder gives the agent, for each coded text: the other values of
the mention (its attributes, and the first text of each child, e.g. the value
`1.4` and unit `mg/dl` of a lab test), the line of the note it comes from, and the
target column (`measurement.measurement_concept_id`). The domain and vocabularies of
the `map` source still restrict the answer. Nothing changes in the YAML.

**The tools.**

| Tool | What it does |
|---|---|
| Semantic search | The retriever's embedding search, with the agent's own queries: expansions ("Hb" → "hemoglobin"), translations ("potássio" → "potassium"), the specimen or unit added |
| Lexical search | Concept names and synonyms containing every word of a query, in `.aatm/omop.db` |
| Code lookup | A code in any vocabulary, with the standard concepts it maps to |
| Concept details | Synonyms, "Maps to" and the LOINC axes (component, property, time, system, scale, method), to check that the specimen and property match the unit |
| Hierarchy | Direct parents and children, to move to a more general or more specific concept |

The answer is checked against the vocabulary: it must be a standard, valid concept of
the required domain and vocabularies, or the agent is asked again.

**What you get back.** `mappings_dataframe()` gains three columns:

- `rationale`: the agent's reason, e.g. *"mg/dl is a mass concentration; serum is the
  usual specimen for creatinine"*;
- `trace`: every action with its query and the candidates it returned;
- `usage`: tool calls, LLM calls, input and output tokens, seconds, and whether the
  budget ran out.

**The score** is the agent's confidence on a fixed scale, so `min_score` means the same
thing whatever the retriever:

| Level | Score | Meaning |
|---|---|---|
| high | 0.9 | Matches the text and everything the context states |
| medium | 0.6 | The right concept, but a detail the context does not state was assumed |
| low | 0.3 | The closest concept available, with a known mismatch or real ambiguity |

`min_score: 0.5` keeps *high* and *medium*. When the budget runs out the agent must
answer from what it has seen, and its score is capped at *low*.

**Caching.** Within a batch, the plain mapper codes each distinct text once. The agent
codes each distinct text *in its context* once: "creatinine 1.4 mg/dL" in two notes is
one search, but "creatinine 1.4 mg/dL" and "urine creatinine 80 mg/dL" are two. To keep
results across batches and runs, see [Keeping results across runs](#keeping-results-across-runs).

### Choosing between the mapper and the agent

On 30 lab results with gold LOINC codes (abbreviations, Portuguese names, 24-hour and
urine tests; LOINC Measurement concepts only; `text-embedding-3-small` index;
`gpt-5-mini`):

| Method | Correct | LLM calls per text | Tokens per text (in / out) |
|---|---|---|---|
| Nearest candidate (`FirstResultSelector`) | 1 / 30 | 0 | – |
| `OpenAILLMSelector` over the top 30 | 14 / 30 | 1 | not reported |
| Agent, text only | 27 / 30 | 2.2 | 3,300 / 940 |
| **Agent, text with context** | **29 / 30** | 2.2 | 3,400 / 900 |

The agent used 2.2 tool calls per text on average (at most 5 of the 8 allowed) and
never ran out of budget. Its one miss was *BUN*, coded to urea nitrogen in blood rather
than in serum or plasma. The LLM selector failed mostly because the right concept was
not among its 30 candidates ("creatinine", "K", "potássio"). With context, the agent
also told *Hb* (mass concentration, not "by calculation") and the albumin/creatinine
*mass* ratio (from `mg/g`) apart.

Most of the time per text goes to the vector search, not to the LLM: a ChromaDB query
filtered by domain takes about 5 seconds on a full index (2.5 million expressions),
against a few milliseconds unfiltered.

The benchmark and its gold codes are in `benchmarks/concept_search/`; run it with your
own retriever and model before choosing.

- **The plain mapper** costs one search (and one LLM call with an LLM selector) per
  distinct text, and is the choice when texts are explicit ("Creatinine in serum") or
  cost and speed matter most.
- **The agent** costs a few LLM calls per distinct text in context, and is the choice
  when texts are short, abbreviated or not in English, and when variants of a test
  differ by specimen or unit.

The agent is also available as the selector of a `TerminologyMapper`, so CSV
terminology mapping and a mapper you already configured can use it:
`TerminologyMapper(retriever=..., selector=AgenticSelector("openai/gpt-5-mini"))`. In a
builder, that mapper also receives the context.

## Writing your own coder

A coder is any object with a `code` method that returns one `CodedText` per text, in
order (the `TextCoder` protocol). The builder calls it once per domain and
vocabularies, with each distinct text once:

```python
from aatm.omop import CodedText, OmopExtractionPipeline


class DictionaryCoder:
    """Codes the texts of a local dictionary: text -> concept_id."""

    def __init__(self, concept_ids: dict[str, int]):
        self.concept_ids = concept_ids

    def code(self, texts, domain=None, vocabularies=None):
        return [
            CodedText(text=t, concept_id=self.concept_ids.get(t.lower()), score=1.0)
            for t in texts
        ]


pipeline = OmopExtractionPipeline(
    "lab_results",
    model_id="openai/gpt-5-mini",
    coder=DictionaryCoder({"hb": 3000963, "creatinine": 3016723}),
)
```

Leave `concept_id` as `None` when nothing fits; the builder stores `0` and reports
`unmapped`. You need not fill `concept_name`, `domain_id` or `standard`: the builder
takes them from the vocabulary database. Set `error` when coding failed, so the result
is not cached.

Two optional protocols add what the built-in coders do:

- **`ContextualTextCoder`**: a `uses_context` attribute (or property) and a `contexts`
  argument to `code`. When `uses_context` is true, the builder passes one
  `CodingContext` per text (related values such as the unit, the sentence of the note,
  the target column), and codes each distinct text *in its context* once.
- **`CacheableCoder`**: a `cache_identity()` method returning JSON-compatible values
  that cover everything that changes the results (models, settings, prompts, index),
  so `cache=True` can reuse them. See [custom coders](#keeping-results-across-runs) in
  the cache section.

```python
class ContextualDictionaryCoder(DictionaryCoder):
    uses_context = True

    def code(self, texts, domain=None, vocabularies=None, contexts=None):
        # contexts[i].related holds the mention's other values, e.g. {"lab_unit": "g/dL"}
        return super().code(texts, domain, vocabularies)

    def cache_identity(self):
        return {"coder": "ContextualDictionaryCoder", "concept_ids": self.concept_ids}
```

The builder checks the coder when it is created: an object that is neither a
`TerminologyMapper` nor has a `code` method (a retriever, a selector, a model name)
raises a `TypeError` before any note is processed. `isinstance(obj, TextCoder)` and
the other two protocols tell you which parts an object implements, and
`mapper.as_coder()` returns the `TerminologyMapperCoder` a `TerminologyMapper` is
wrapped in.

## Keeping results across runs

Coding is usually the most expensive step. Without a cache, every batch and every rerun
(after a crash, or to rebuild the tables after changing a mapping) pays for the same
texts again. Turn on the coding cache to store each result on disk and reuse it:

```python
from aatm.omop import CodingCache, OmopExtractionPipeline

pipeline = OmopExtractionPipeline(
    "lab_results", model_id="openai/gpt-5-mini", coder=coder, cache=True
)
tables = await pipeline(notes)
tables.cache_stats  # CacheStats(hits=..., misses=..., stored=...), and .hit_rate
```

| `cache=` | Effect |
|---|---|
| `None` or `False` (default) | No cache |
| `True` | `.aatm/coding_cache.db` |
| `"path/to/cache.db"` | That file |
| `CodingCache(path, max_entries=..., max_bytes=...)` | That file, with a size limit |

The same argument works on `OmopTableBuilder`. In `mappings_dataframe()`, `cached`
tells which concepts came from the cache; their `usage` was spent in the run that
first coded them.

**A result is only reused for exactly the same input.** The cache never changes what a
run produces:

- The agent (and a mapper with `AgenticSelector`) reads context, so its key is the text
  **with its full context**: the related values (such as the lab value and unit), the
  line of the note and the target column. "Hb 11.2 g/dL" in two notes with the same
  line is one search; the same "Hb" on a different line is searched again. Hits
  therefore come from reruns, rebuilds and repeated lines (templated notes), rather
  than from the same text in different notes.
- The plain mapper ignores context, so its key is the text, domain and vocabularies:
  each distinct text is coded once, across all runs.
- Every key includes **the coder's identity**: its class, models, generation settings,
  retriever and index, tool budget, the agent's prompts, the server URL of
  OpenAI-compatible providers and the aatm version. Changing any of these, or upgrading
  aatm, starts from an empty cache for that coder. The old entries stay in the file
  until they are evicted or cleared.
- The identity also covers **the data the coder reads**: the vocabulary releases in
  the vocabulary database (e.g. `LOINC 2.78`) and its size, and the id and size of the
  ChromaDB collection. Rebuilding either from other data starts from an empty cache,
  and a warning names the old entries so you can clear them.
- Results that ended in an error (e.g. an API failure) are not stored, so they are
  retried on the next run.

Each batch codes its most frequent texts first and writes results as it goes, so a
run that stops partway keeps what it paid for.

**Size.** By default the cache grows without limit. Set `max_entries` (results kept) or
`max_bytes` (bytes of results kept; the file is somewhat larger) to cap it. When a write
goes over a limit, the entries used least recently are deleted first:

```python
cache = CodingCache(".aatm/coding_cache.db", max_entries=1_000_000, max_bytes=2 * 1024**3)
```

**Clearing.** Delete everything and shrink the file, or one coder's entries:

```python
cache.info()   # entries, size on disk, and the coders they belong to
cache.clear()  # everything
cache.clear(coder="<hash from info()>")
```

or from the command line:

```bash
aatm cache info
aatm cache clear            # asks for confirmation; --yes to skip it
aatm cache clear --coder <hash>
```

!!! warning "Rebuilt data the cache cannot see"
    The data of a retriever that is neither a ChromaDB collection nor a Pinecone index
    (a retriever of your own, for example) cannot be fingerprinted. When the cache opens, a warning names such
    components: `changes to the content of ... cannot be detected`. If you rebuild
    their index, run `aatm cache clear`, or results coded against the old data will be
    reused without any error.

    Model aliases are another blind spot: `openai/gpt-5-mini` names whatever the
    provider currently serves under that name. Pin a dated model version when results
    must stay reproducible, and clear the cache when you change it.

**Custom coders.** A coder of your own is cached when it has a `cache_identity()`
method (the `CacheableCoder` protocol, see
[Writing your own coder](#writing-your-own-coder)) returning JSON-compatible values
that cover everything that changes its results. Otherwise wrap it yourself and say
what it is:
`CachedCoder(my_coder, cache, identity={"name": "my_coder", "version": 2})`.

## Rate limits for long runs

No rate limits are set by default, and rate-limit and transient errors are retried on
their own. A chunked run (`pipeline.run`) keeps 16 extraction calls in flight with a
hosted API and 4 with a local server, unless you set `max_concurrency`. For large
datasets, or a provider quota you must stay under, set limits once before the run;
they cover extraction and coding together, and a run derives its extraction
concurrency from them:

```python
from aatm.llm import set_rate_limits

set_rate_limits("openai", max_concurrency=32, requests_per_minute=5000)
```

See [Rate limits and retries](quick_start_extraction.md#rate-limits-and-retries-optional)
for every option.


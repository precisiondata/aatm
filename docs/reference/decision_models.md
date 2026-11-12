# Decision models

Decision models answer **closed-set questions** about a state (a text or a JSON
record): pick one of several options, place it on an ordered scale, or answer yes or
no. They return the answer **with a probability per option**, faster and cheaper than
a generative LLM call, and write no text.

Many steps of a mapping or extraction pipeline are decisions of this kind: choosing a
candidate concept, labelling an attribute, checking that a mention is not negated.
Two steps use them so far:
[`DecisionSelector`](#selecting-concepts-decisionselector) and
[`DecisionAttributeLabeller`](#labelling-attributes-decisionattributelabeller); see
[issue #77](https://github.com/precisiondata/aatm/issues/77) for the others
planned on top of this interface.

```python
import asyncio

from aatm.llm import Question, create_decision_model

model = create_decision_model("openrouter/typesafe/jev-1.13")
decision = asyncio.run(
    model.decide(
        {"text": "creatinine", "unit": "mg/dl", "value": "1.4"},
        [
            Question.choice(
                "specimen",
                "Which specimen was the test most likely done on?",
                {"serum": "Serum or plasma", "urine": "Urine", "unknown": "Not stated"},
            ),
            Question.score(
                "abnormal", "How abnormal is the value?", ["normal", "mild", "severe"]
            ),
            Question.predicate("is_lab", "Is the text the name of a lab test?"),
        ],
    )
)
decision["specimen"].value          # "serum"
decision["specimen"].probabilities  # {"serum": 0.82, "urine": 0.11, "unknown": 0.07}
decision["abnormal"].label          # the most likely level
decision["is_lab"].probability      # probability of yes
```

## Backends

| Model id | Backend | Key | Probabilities |
|---|---|---|---|
| `openai/gpt-6-luna` | [OpenAI Decisions API](https://developers.openai.com/api/docs/guides/decisions) | `OPENAI_API_KEY` | yes |
| `gemini/gemini-decision-1` | [Gemini Decisions API](https://aistudio.google.com/docs/decisions-api) (`v1beta`) | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | yes |
| `openrouter/typesafe/jev-1.13` | [OpenRouter decisions endpoint](https://openrouter.ai/docs/guides/community/jev) (alpha), serving TypeSafe's Jev | `OPENROUTER_API_KEY` | yes |
| `jev-compatible/<model>` + `base_url` | Any endpoint with Jev's request shape: TypeSafe, Eden AI, a self-hosted model | `api_key=` | yes |
| `llm/<llm id>`, e.g. `llm/anthropic/claude-haiku-5-5` | Any `aatm.llm.create_llm` model, through structured output | the LLM's | **no** |

`llm/` models let pipeline steps run with any provider, or offline with a local
model, but their answers carry no probabilities: thresholds on probabilities do not
apply to them.

Calls share the rate limits and retries of `aatm.llm.limits`, and are counted by
`aatm.llm.track_usage`. OpenRouter reports each call's cost, which
`TokenUsage.cost` uses when you give no price for the model.

!!! warning "Pin versions"
    Aliases such as `jev-latest` move: the answers behind cached results change. Use
    a versioned model id (`typesafe/jev-1.13`); aatm warns about `latest` and
    `preview` aliases.

!!! warning "Data governance"
    Decision backends send the state, e.g. a sentence of a clinical note, to the
    provider: OpenAI, Google, OpenRouter and TypeSafe, or Eden AI. Before sending protected
    health information, check the provider's data processing agreement (and BAA,
    where required) and its data retention. Use `llm/` with a local model to keep
    data on your machine.

!!! note "Probabilities are not accuracy"
    A confident answer can be wrong, and some models' probabilities shift with the
    order of the options. Calibrate thresholds on your own labelled data.

## Selecting concepts: `DecisionSelector`

`DecisionSelector` is a selector of the terminology mapper. For each text, it asks
the decision model one choice question: which of the top-k retrieved candidate
concepts means the same as the text, or `none`. Expressions of the same standard
concept (synonyms) are one option, so their probability is not split.

```python
from aatm.omop import OmopTableBuilder
from aatm.terminology_mapper import TerminologyMapper

mapper = TerminologyMapper(
    retriever="text-embedding-3-small",
    selector="decision-openrouter/typesafe/jev-1.13",
)
builder = OmopTableBuilder(definitions, mapper=mapper)
```

Registered names are `decision-openai/gpt-6-luna`, `decision-gemini/gemini-decision-1`
and `decision-openrouter/typesafe/jev-1.13`; any other `decision-<model id>` works
too, e.g. `decision-llm/ollama/llama3.1:8b`. To set `top_k`, the question, or the
model's `api_key`, `base_url` and `settings`, build the selector:

```python
from aatm.selectors import DecisionSelector

selector = DecisionSelector(
    "jev-compatible/jev-1.13",
    top_k=5,
    model_kwargs={"base_url": "https://jev.example.org/v1/decisions"},
)
```

**The probability of the chosen concept is the mapping's score**: the mapper's
`confidence_score` and the builder's `score`. A `min_score` in a definition is
therefore a threshold on the model's probability: `map: {min_score: 0.8}` keeps the
concepts chosen with at least 80 %, and stores `0` with a `low_confidence` issue
for the others. Each mapping's trace has the provider's model version and the
probability of every option.

!!! warning "`llm/` models give no score"
    With an `llm/` model the selector still chooses, but the mappings have no score
    (`confidence_score` is empty): a `min_score` rejects all of them. aatm warns
    when you create such a selector.

The coding cache (`aatm.omop.CachedCoder`) keys results on the selector's
`cache_identity()`: the model id, `top_k`, the question and the endpoint. Changing
any of them, e.g. moving from `jev-1.13` to `jev-1.14`, codes the texts again. A
moving alias (`jev-latest`) would not: pin versions.

The texts to map and the candidate names are sent to the provider; see the data
governance warning above. The API reference of `DecisionSelector` is on the
[Selectors](selectors.md) page.

## Labelling attributes: `DecisionAttributeLabeller`

An entity [attribute](../getting-started/entity_definitions.md#attributes-what-the-note-says-about-a-mention)
with a closed list of labels, such as `smoking_status.status` or the ECOG grade, is a
closed-set question. By default the extractor's own LLM call chooses the labels. To
have a decision model label them instead, or check them, give the extractor a
labeller:

```python
from aatm.extractors import DecisionAttributeLabeller
from aatm.omop import OmopExtractionPipeline

pipeline = OmopExtractionPipeline(
    ["smoking_status", "ecog_performance_status"],
    model_id="openai/gpt-5-mini",
    attribute_labeller=DecisionAttributeLabeller(
        "openrouter/typesafe/jev-1.13",
        mode="verify",
        min_probability=0.8,
    ),
)
```

`SimpleLLMExtractor(..., attribute_labeller=...)` takes the same argument. Without
it, nothing changes.

**Who labels is set in code; what kind of question each attribute is, in the
definitions.** The labeller sets the model, the mode, the attributes (`attributes=
["smoking_status", "ecog_performance_status.score"]`; all of them by default) and the
threshold. The definition file says whether an attribute's labels form a scale:
with `ordered: true` (the ECOG grades) the model answers a **score** question, which
knows that grade 2 lies between 1 and 3; otherwise a **choice** question. The
extractor ignores `ordered`, so the definition files work the same with or without a
labeller.

For each mention, the model reads the entity's name, the mention's text and its line
of the note, and answers every labelled attribute of the mention in one call. Every
question can be answered "not stated" (a `not_stated` option, or a yes/no
"is it stated?" question next to a score), since a mention need not state every
attribute: the attribute is then left out, as the extractor would.

| Mode | The extractor | The decision model | A label is written when |
|---|---|---|---|
| `verify` (default) | labels as before | labels each mention too | both give the same answer |
| `replace` | only finds mentions: the labelled attributes are left out of its prompt | labels each mention | the decision model gives it |

**`min_probability`** is a threshold on the decision model's probability. With
`min_probability=0.8`, only labels the model gives at least 80 % are written. The
default, `None`, keeps every label. Either way, the probability of each label written
is in `ExtractedEntity.attribute_probabilities` (and `EntityRecord.attribute_probabilities`).
`llm/` models give no probabilities: with a threshold, all their labels go to review.

**Review.** A label below the threshold, a disagreement (`verify`), or a refusal is
**not written**: the mention keeps its other attributes, and the label goes to the
result's `attribute_reviews` for a person to check, with the extractor's label, the
decision model's label and its probability of each option. When the attribute is
`required`, the mention is rejected (`required_attribute_under_review`), with its
children, rather than written without it. The OMOP builder collects the reviews in
`OmopTables.attribute_reviews_dataframe()`, and a run writes them to the
`attribute_reviews` folder (`OmopRun.attribute_reviews()`).

**Runs.** A run's stored results depend on the labeller as much as on the extractor:
resuming with another labeller, mode, threshold or model is refused, as with another
extractor. With `batch=True`, the labels are asked with live calls when a batch job's
results come back, before they are stored; a run that stops meanwhile collects the job
again and labels it then.

Which mode to use is a matter of accuracy and cost on your notes: see the
[attribute labelling benchmark](https://github.com/precisiondata/aatm/tree/main/benchmarks/attribute_labels).

Each mention and its line are sent to the decision provider; see the data governance
warning above.

::: aatm.extractors.attribute_labeller

::: aatm.llm.decisions

::: aatm.llm.openai_decisions

::: aatm.llm.gemini_decisions

::: aatm.llm.jev

::: aatm.llm.registry.create_decision_model

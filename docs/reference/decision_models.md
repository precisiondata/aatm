# Decision models

Decision models answer **closed-set questions** about a state (a text or a JSON
record): pick one of several options, place it on an ordered scale, or answer yes or
no. They return the answer **with a probability per option**, faster and cheaper than
a generative LLM call, and write no text.

Many steps of a mapping or extraction pipeline are decisions of this kind: choosing a
candidate concept, labelling an attribute, checking that a mention is not negated.
The first such step is [`DecisionSelector`](#selecting-concepts-decisionselector);
see [issue #77](https://github.com/precisiondata/aatm/issues/77) for the others
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

::: aatm.llm.decisions

::: aatm.llm.openai_decisions

::: aatm.llm.gemini_decisions

::: aatm.llm.jev

::: aatm.llm.registry.create_decision_model

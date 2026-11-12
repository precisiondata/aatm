"""Tests for `aatm.extractors.DecisionAttributeLabeller`, with fake models."""

import asyncio
import logging

import pytest

from aatm.data_models.extractors import (
    AttributeDefinition,
    AttributeLabel,
    EntityDefinition,
    EntityDefinitions,
    ExtractedAttribute,
    ExtractedEntityCandidate,
    ExtractedEntityCandidates,
)
from aatm.extractors import DecisionAttributeLabeller, SimpleLLMExtractor
from aatm.llm.batch import BatchResult
from aatm.llm.decisions import Answer, Decision, DecisionError, LLMDecisionModel


def labels(*values):
    return [AttributeLabel(value=v, description=f"Label {v}.") for v in values]


SMOKING = EntityDefinition(
    name="Smoking status",
    key="smoking_status",
    definition="A statement of the patient's tobacco use.",
    attributes=[
        AttributeDefinition(
            key="status",
            description="Whether the patient smokes.",
            required=True,
            labels=labels("never", "former", "current"),
        ),
        AttributeDefinition(
            key="product",
            description="What the patient smokes.",
            labels=labels("cigarettes", "cigars"),
        ),
    ],
)
QUIT_DATE = EntityDefinition(
    name="Quit date",
    key="quit_date",
    definition="When the patient stopped smoking.",
    has_parent=True,
    parent_key="smoking_status",
)
ECOG = EntityDefinition(
    name="ECOG performance status",
    key="ecog",
    definition="An ECOG grade.",
    attributes=[
        AttributeDefinition(
            key="score",
            description="The grade.",
            ordered=True,
            labels=labels("0", "1", "2", "3", "4", "5"),
        )
    ],
)
DEFINITIONS = EntityDefinitions(entities=[SMOKING, QUIT_DATE, ECOG])

NOTE = "Social history: former smoker, quit 2010.\nECOG 1 today."


def candidates(status="former", product=None, ecog=None):
    attributes = [ExtractedAttribute(key="status", value=status)] if status else []
    if product:
        attributes.append(ExtractedAttribute(key="product", value=product))
    ecog_attributes = [ExtractedAttribute(key="score", value=ecog)] if ecog else []
    return ExtractedEntityCandidates(
        entities=[
            ExtractedEntityCandidate(
                id="e1",
                text="former smoker",
                key="smoking_status",
                line="L1",
                attributes=attributes,
            ),
            ExtractedEntityCandidate(
                id="e2", text="2010", key="quit_date", line="L1", parents=["e1"]
            ),
            ExtractedEntityCandidate(
                id="e3",
                text="ECOG 1",
                key="ecog",
                line="L2",
                attributes=ecog_attributes,
            ),
        ]
    )


class FakeLLM:
    model_id = "fake/extractor"

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    async def generate(self, *, system, user, output_type):
        self.calls.append({"system": system, "user": user})
        return self.answer


class FakeDecisionModel:
    """Answers each question from `answers[name]`: ``(value, probabilities)``.

    A choice or score question without an entry answers its first option; a
    predicate without one answers yes.
    """

    def __init__(self, answers=None, model_id="fake/decider-1", error=False):
        self.answers = answers or {}
        self.model_id = model_id
        self.error = error
        self.calls = []

    async def decide(self, state, questions):
        self.calls.append((state, list(questions)))
        if self.error:
            raise DecisionError("bad answer")
        out = []
        for q in questions:
            value, probabilities = self.answers.get(q.name, (None, {}))
            if value == "refuse":
                out.append(Answer(name=q.name, kind=q.kind, value=None, refused=True))
                continue
            if q.kind == "predicate":
                p = (
                    probabilities.get("true", 0.9)
                    if value is None
                    else (0.9 if value else 0.1)
                )
                out.append(
                    Answer(
                        name=q.name,
                        kind="predicate",
                        value=p >= 0.5,
                        probabilities={"true": p, "false": 1 - p},
                    )
                )
            elif q.kind == "score":
                label = value if value is not None else q.values[0]
                out.append(
                    Answer(
                        name=q.name,
                        kind="score",
                        value=float(q.values.index(label)),
                        label=label,
                        probabilities=probabilities or {label: 0.8},
                    )
                )
            else:
                choice = value if value is not None else q.values[0]
                out.append(
                    Answer(
                        name=q.name,
                        kind="choice",
                        value=choice,
                        probabilities=probabilities or {choice: 0.9},
                    )
                )
        return Decision(model=self.model_id, answers=out)


def extract(labeller, answer, text=NOTE, **kwargs):
    llm = FakeLLM(answer)
    extractor = SimpleLLMExtractor(
        entity_definitions=DEFINITIONS, llm=llm, attribute_labeller=labeller, **kwargs
    )
    (result,) = asyncio.run(extractor([text]))
    return result, llm, extractor


def entity(result, entity_id):
    return result.get_entity(entity_id)


# --- verify mode ------------------------------------------------------------------


def test_verify_keeps_agreed_labels_with_their_probability():
    model = FakeDecisionModel({"status": ("former", {"former": 0.93})})
    labeller = DecisionAttributeLabeller(
        model, mode="verify", attributes=["smoking_status.status"]
    )
    result, llm, _ = extract(labeller, candidates())

    smoking = entity(result, "e1")
    assert smoking.get_attribute("status") == "former"
    assert smoking.attribute_probabilities == {"status": 0.93}
    assert result.attribute_reviews == []
    # verify mode leaves the extractor's prompt as it was
    assert "`status`" in llm.calls[0]["system"]
    record = result.to_records()[0]
    assert record.attribute_probabilities == {"status": 0.93}


def test_state_is_the_mention_and_its_line():
    model = FakeDecisionModel({"status": ("former", {})})
    labeller = DecisionAttributeLabeller(
        model, mode="verify", attributes=["smoking_status.status"]
    )
    extract(labeller, candidates())
    ((state, questions),) = model.calls
    assert state == {
        "entity": "Smoking status",
        "mention": "former smoker",
        "line": "Social history: former smoker, quit 2010.",
    }
    (question,) = questions
    assert question.kind == "choice"
    assert question.values == ["never", "former", "current", "not_stated"]
    assert "Smoking status" in question.instructions
    assert "Whether the patient smokes." in question.instructions


def test_one_call_per_mention_asks_all_its_attributes():
    model = FakeDecisionModel({"status": ("former", {}), "product": ("not_stated", {})})
    labeller = DecisionAttributeLabeller(
        model, mode="verify", attributes=["smoking_status"]
    )
    extract(labeller, candidates())
    ((_, questions),) = model.calls
    assert [q.name for q in questions] == ["status", "product"]


def test_disagreement_goes_to_review_and_rejects_required_attribute_with_children():
    model = FakeDecisionModel({"status": ("current", {"current": 0.7})})
    labeller = DecisionAttributeLabeller(
        model, mode="verify", attributes=["smoking_status.status"]
    )
    result, _, _ = extract(labeller, candidates())

    (review,) = result.attribute_reviews
    assert review.reason == "disagreement"
    assert review.extractor_value == "former"
    assert review.decision_value == "current"
    assert review.probability == 0.7
    assert review.text == "former smoker"
    assert review.model == "fake/decider-1"

    assert [e.id for e in result.validated_entities] == ["e3"]
    rejected = {e.id: e.reasons for e in result.unmatched_entities}
    assert rejected == {
        "e1": ["required_attribute_under_review"],
        "e2": ["parent_rejected"],
    }
    assert result.unmatched_keys == ["quit_date", "smoking_status"]


def test_disagreement_on_optional_attribute_keeps_the_mention():
    model = FakeDecisionModel(
        {"status": ("former", {}), "product": ("cigars", {"cigars": 0.6})}
    )
    labeller = DecisionAttributeLabeller(
        model, mode="verify", attributes=["smoking_status"]
    )
    result, _, _ = extract(labeller, candidates(product="cigarettes"))

    smoking = entity(result, "e1")
    assert smoking.get_attribute("status") == "former"
    assert smoking.get_attribute("product") is None
    (review,) = result.attribute_reviews
    assert (review.attribute_key, review.reason) == ("product", "disagreement")


def test_both_not_stated_agree():
    model = FakeDecisionModel(
        {"status": ("former", {}), "product": ("not_stated", {"not_stated": 0.95})}
    )
    labeller = DecisionAttributeLabeller(
        model, mode="verify", attributes=["smoking_status"]
    )
    result, _, _ = extract(labeller, candidates())
    assert entity(result, "e1").get_attribute("product") is None
    assert result.attribute_reviews == []


def test_low_probability_goes_to_review():
    model = FakeDecisionModel({"status": ("former", {"former": 0.55})})
    labeller = DecisionAttributeLabeller(
        model, mode="verify", attributes=["smoking_status.status"], min_probability=0.8
    )
    result, _, _ = extract(labeller, candidates())
    (review,) = result.attribute_reviews
    assert review.reason == "low_probability"
    assert review.decision_value == review.extractor_value == "former"
    assert entity(result, "e1") is None


def test_refusal_goes_to_review():
    model = FakeDecisionModel({"status": ("refuse", {})})
    labeller = DecisionAttributeLabeller(
        model, mode="verify", attributes=["smoking_status.status"]
    )
    result, _, _ = extract(labeller, candidates())
    (review,) = result.attribute_reviews
    assert review.reason == "refused"


def test_mentions_rejected_by_validation_are_not_labelled():
    model = FakeDecisionModel()
    labeller = DecisionAttributeLabeller(
        model, mode="verify", attributes=["smoking_status.status"]
    )
    result, _, _ = extract(labeller, candidates(status=None))
    assert model.calls == []
    assert "e1" in {e.id for e in result.unmatched_entities}


# --- replace mode -------------------------------------------------------------------


def test_replace_leaves_attributes_out_of_the_prompt_and_uses_decisions():
    model = FakeDecisionModel({"status": ("never", {"never": 0.9})})
    labeller = DecisionAttributeLabeller(
        model, mode="replace", attributes=["smoking_status.status"]
    )
    # The extractor labels anyway, and leaves the required attribute out: neither
    # matters in replace mode.
    result, llm, _ = extract(labeller, candidates(status="current", product="cigars"))

    system = llm.calls[0]["system"]
    assert "`status`" not in system
    assert "`product`" in system
    smoking = entity(result, "e1")
    assert smoking.get_attribute("status") == "never"
    assert smoking.get_attribute("product") == "cigars"
    assert [a.key for a in smoking.attributes] == ["status", "product"]
    assert result.optional_attribute_issues == []
    assert result.attribute_reviews == []


def test_replace_rejects_a_required_attribute_found_not_stated():
    model = FakeDecisionModel({"status": ("not_stated", {"not_stated": 0.9})})
    labeller = DecisionAttributeLabeller(
        model, mode="replace", attributes=["smoking_status.status"]
    )
    result, _, _ = extract(labeller, candidates(status=None))
    rejected = {e.id: e.reasons for e in result.unmatched_entities}
    assert rejected["e1"] == ["required_attribute_missing"]
    (issue,) = result.required_attribute_issues
    assert (issue.entity_id, issue.reason) == ("e1", "missing_required")


# --- ordered attributes -------------------------------------------------------------


def test_ordered_attribute_is_a_score_with_a_stated_predicate():
    model = FakeDecisionModel(
        {
            "score_stated": (None, {"true": 0.9}),
            "score": ("1", {"0": 0.1, "1": 0.8, "2": 0.1}),
        }
    )
    labeller = DecisionAttributeLabeller(model, mode="replace", attributes=["ecog"])
    result, _, _ = extract(labeller, candidates())

    ((_, questions),) = model.calls
    assert [(q.name, q.kind) for q in questions] == [
        ("score_stated", "predicate"),
        ("score", "score"),
    ]
    assert questions[1].values == ["0", "1", "2", "3", "4", "5"]
    ecog = entity(result, "e3")
    assert ecog.get_attribute("score") == "1"
    assert ecog.attribute_probabilities["score"] == pytest.approx(0.72)


def test_ordered_attribute_not_stated():
    model = FakeDecisionModel({"score_stated": (False, {})})
    labeller = DecisionAttributeLabeller(
        model, mode="replace", attributes=["ecog"], min_probability=0.8
    )
    result, _, _ = extract(labeller, candidates())
    assert entity(result, "e3").attributes == []
    assert result.attribute_reviews == []


def test_ordered_review_has_the_distribution_with_not_stated():
    model = FakeDecisionModel(
        {
            "score_stated": (None, {"true": 0.6}),
            "score": ("2", {"1": 0.5, "2": 0.5}),
        }
    )
    labeller = DecisionAttributeLabeller(
        model, mode="replace", attributes=["ecog"], min_probability=0.5
    )
    result, _, _ = extract(labeller, candidates())
    (review,) = result.attribute_reviews
    assert review.reason == "low_probability"
    assert review.probabilities == pytest.approx(
        {"1": 0.3, "2": 0.3, "not_stated": 0.4}
    )


# --- options ------------------------------------------------------------------------


def test_not_stated_option_avoids_a_label_of_the_same_name():
    definitions = EntityDefinitions(
        entities=[
            EntityDefinition(
                name="Thing",
                key="thing",
                definition="A thing.",
                attributes=[
                    AttributeDefinition(
                        key="kind",
                        description="Its kind.",
                        labels=labels("not_stated", "other"),
                    )
                ],
            )
        ]
    )
    model = FakeDecisionModel({"kind": ("not_stated_", {})})
    labeller = DecisionAttributeLabeller(model, mode="replace")
    llm = FakeLLM(
        ExtractedEntityCandidates(
            entities=[
                ExtractedEntityCandidate(id="e1", text="thing", key="thing", line="L1")
            ]
        )
    )
    extractor = SimpleLLMExtractor(
        entity_definitions=definitions, llm=llm, attribute_labeller=labeller
    )
    (result,) = asyncio.run(extractor(["a thing"]))
    ((_, (question,)),) = model.calls
    assert question.values == ["not_stated", "other", "not_stated_"]
    assert result.validated_entities[0].attributes == []


def test_attributes_default_to_all_and_are_checked():
    labeller = DecisionAttributeLabeller(FakeDecisionModel(), mode="verify")
    assert {
        k: [a.key for a in v] for k, v in labeller.targets(DEFINITIONS).items()
    } == {
        "smoking_status": ["status", "product"],
        "ecog": ["score"],
    }
    for names, message in [
        (["nope"], "Unknown entity"),
        (["quit_date"], "has no attributes"),
        (["smoking_status.nope"], "no attribute 'nope'"),
    ]:
        labeller = DecisionAttributeLabeller(
            FakeDecisionModel(), mode="replace", attributes=names
        )
        with pytest.raises(ValueError, match=message):
            # replace mode checks the names when the extractor is built
            SimpleLLMExtractor(
                entity_definitions=DEFINITIONS,
                llm=FakeLLM(candidates()),
                attribute_labeller=labeller,
            )


def test_argument_errors():
    with pytest.raises(ValueError, match="mode"):
        DecisionAttributeLabeller(FakeDecisionModel(), mode="check")
    with pytest.raises(ValueError, match="min_probability"):
        DecisionAttributeLabeller(FakeDecisionModel(), mode="verify", min_probability=2)
    with pytest.raises(TypeError, match="list of names"):
        DecisionAttributeLabeller(FakeDecisionModel(), mode="verify", attributes="ecog")
    with pytest.raises(TypeError, match="DecisionModel"):
        DecisionAttributeLabeller(object(), mode="verify")
    # The mode is never implied: the developer chooses it.
    with pytest.raises(TypeError, match="mode"):
        DecisionAttributeLabeller(FakeDecisionModel())  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        DecisionAttributeLabeller(FakeDecisionModel(), "verify")  # type: ignore[misc]
    with pytest.raises(ValueError, match="model_kwargs"):
        DecisionAttributeLabeller(
            FakeDecisionModel(), mode="verify", model_kwargs={"api_key": "x"}
        )


def test_model_without_probabilities(caplog):
    class FixedLLM:
        model_id = "fake/llm"

        async def generate(self, *, system, user, output_type):
            return output_type.model_validate({"status": "former"})

    with caplog.at_level(logging.WARNING):
        strict = DecisionAttributeLabeller(
            LLMDecisionModel(FixedLLM()),
            mode="verify",
            attributes=["smoking_status.status"],
            min_probability=0.5,
        )
    assert "gives no probabilities" in caplog.text
    result, _, _ = extract(strict, candidates())
    assert [r.reason for r in result.attribute_reviews] == ["low_probability"]

    lenient = DecisionAttributeLabeller(
        LLMDecisionModel(FixedLLM()),
        mode="verify",
        attributes=["smoking_status.status"],
    )
    result, _, _ = extract(lenient, candidates())
    smoking = entity(result, "e1")
    assert smoking.get_attribute("status") == "former"
    assert smoking.attribute_probabilities == {}


# --- errors and batch ---------------------------------------------------------------


def test_labeller_errors_follow_on_error():
    labeller = DecisionAttributeLabeller(FakeDecisionModel(error=True), mode="verify")
    with pytest.raises(DecisionError):
        extract(labeller, candidates())
    result, _, _ = extract(labeller, candidates(), on_error="record")
    assert result.error == "DecisionError: bad answer"
    assert result.validated_entities == []


class FakeBatchLLM(FakeLLM):
    async def submit_batch(self, requests, output_type):
        self.calls.extend({"system": r.system, "user": r.user} for r in requests)
        return "batch_1"

    async def poll_batch(self, batch_id, custom_ids, output_type):
        return [BatchResult(custom_id=c, output=self.answer) for c in custom_ids]


def test_poll_batch_labels_the_results():
    model = FakeDecisionModel({"status": ("current", {})})
    labeller = DecisionAttributeLabeller(
        model, mode="replace", attributes=["smoking_status.status"]
    )
    llm = FakeBatchLLM(candidates(status=None))
    extractor = SimpleLLMExtractor(
        entity_definitions=DEFINITIONS, llm=llm, attribute_labeller=labeller
    )
    items = [("a", NOTE), ("b", NOTE)]
    asyncio.run(extractor.submit_batch(items))
    assert "`status`" not in llm.calls[0]["system"]
    results = asyncio.run(extractor.poll_batch("batch_1", items))
    assert len(model.calls) == 2
    for key in "ab":
        assert results[key].get_entity("e1").get_attribute("status") == "current"


def test_poll_batch_records_labeller_errors_per_note():
    labeller = DecisionAttributeLabeller(FakeDecisionModel(error=True), mode="verify")
    extractor = SimpleLLMExtractor(
        entity_definitions=DEFINITIONS,
        llm=FakeBatchLLM(candidates()),
        attribute_labeller=labeller,
    )
    results = asyncio.run(extractor.poll_batch("batch_1", [("a", NOTE)]))
    assert results["a"].error == "DecisionError: bad answer"


# --- OMOP pipeline ------------------------------------------------------------------


def ecog_pipeline(labeller=None, definitions=None):
    from test_omop_mapping import FakeLookup, ecog, ecog_definitions

    from aatm.omop import OmopExtractionPipeline

    definitions = definitions or ecog_definitions()
    answer = ExtractedEntityCandidates.model_validate(
        {"entities": [ecog("e1", "ECOG 1", "1")]}
    )
    extractor = SimpleLLMExtractor(
        entity_definitions=definitions,
        llm=FakeLLM(answer),
        attribute_labeller=labeller,
    )
    return OmopExtractionPipeline(
        definitions, extractor=extractor, vocabulary=FakeLookup()
    )


def ecog_notes(n=2):
    return [
        {"note_id": i, "text": f"Visit {i}. ECOG 1", "note_date": "2025-01-01"}
        for i in range(1, n + 1)
    ]


def test_run_writes_attribute_reviews(tmp_path):
    from aatm.omop import OmopRun

    model = FakeDecisionModel({"score": ("2", {"2": 0.6})})
    pipeline = ecog_pipeline(DecisionAttributeLabeller(model, mode="verify"))
    summary = asyncio.run(pipeline.run(ecog_notes(), tmp_path))

    assert summary.rows == {"measurement": 2}
    output = OmopRun(tmp_path)
    measurement = output.table("measurement")
    assert measurement["value_source_value"].isna().all()

    reviews = output.attribute_reviews()
    assert sorted(reviews["note_id"]) == [1, 2]
    row = reviews.iloc[0]
    assert (row["attribute_key"], row["extractor_value"], row["decision_value"]) == (
        "score",
        "1",
        "2",
    )
    assert row["reason"] == "disagreement"
    assert row["probability"] == 0.6

    tables = asyncio.run(pipeline(["Visit. ECOG 1"]))
    df = tables.attribute_reviews_dataframe()
    assert list(df["note_index"]) == [0]


def test_resuming_needs_the_same_labeller(tmp_path):
    model = FakeDecisionModel({"score": ("1", {})})
    asyncio.run(
        ecog_pipeline(DecisionAttributeLabeller(model, mode="verify")).run(
            ecog_notes(), tmp_path
        )
    )
    # The same labeller resumes.
    asyncio.run(
        ecog_pipeline(DecisionAttributeLabeller(model, mode="verify")).run(
            ecog_notes(), tmp_path
        )
    )
    for other in (
        None,
        DecisionAttributeLabeller(model, mode="verify", min_probability=0.9),
        DecisionAttributeLabeller(model, mode="replace"),
        DecisionAttributeLabeller(
            FakeDecisionModel(model_id="fake/decider-2"), mode="verify"
        ),
    ):
        with pytest.raises(ValueError, match="another extractor"):
            asyncio.run(ecog_pipeline(other).run(ecog_notes(), tmp_path))


def test_ordered_alone_keeps_earlier_runs_resumable(tmp_path):
    from test_omop_mapping import ecog_definitions

    from aatm.omop.runner import ChunkedRun

    plain = ecog_definitions()
    ordered = ecog_definitions()
    ordered.entities[0].attributes[0].ordered = True

    def fingerprint(definitions):
        run = ChunkedRun(ecog_pipeline(definitions=definitions), tmp_path)
        return run._extraction_fingerprint()

    assert fingerprint(plain) == fingerprint(ordered)

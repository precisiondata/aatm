"""`DecisionSelector`: choosing among retrieved candidates with a decision model."""

import asyncio
import json
import logging

import pandas as pd
import pytest

from aatm.data_models.data_models import (
    EmptySelectionMetadata,
    RetrieverResults,
    SelectedExpressionMetadata,
    SelectorResults,
)
from aatm.llm import Answer, Decision, DecisionError, DecisionModel, LLMDecisionModel
from aatm.omop import OmopTableBuilder, TerminologyMapperCoder
from aatm.omop.cache import describe_component
from aatm.registries.selectors import SELECTORS_REGISTRY, load_selector
from aatm.selectors import NONE_OPTION, BaseSelector, DecisionSelector
from aatm.terminology_mapper import TerminologyMapper
from test_concept_search import FakeRetriever, expression
from test_omop_coding import UnitLookup, lab, lab_definitions, run

SERUM = "3016723"  # Creatinine [Mass/volume] in Serum or Plasma (LOINC 2160-0)
URINE = "3007760"  # Creatinine [Mass/volume] in Urine
HB = "3000963"  # Hemoglobin [Mass/volume] in Blood


class FakeDecisionModel:
    """Answers the choice question with the option whose description contains a
    given concept name, with fixed probabilities; records states and questions."""

    def __init__(
        self,
        picks=None,
        probability=0.8,
        model_id="fake/decider-1",
        probabilities=True,
        refuse=False,
        error=False,
    ):
        self.picks = picks or {}  # query text -> concept name, or None for "none"
        self.probability = probability
        self.model_id = model_id
        self.probabilities = probabilities
        self.refuse = refuse
        self.error = error
        self.calls: list[tuple] = []
        self.running = 0
        self.max_running = 0

    async def decide(self, state, questions):
        self.calls.append((state, list(questions)))
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        await asyncio.sleep(0)
        self.running -= 1
        if self.error:
            raise DecisionError("bad answer")
        (question,) = questions
        if self.refuse:
            return Decision(
                model=self.model_id,
                answers=[
                    Answer(name=question.name, kind="choice", value=None, refused=True)
                ],
            )
        name = self.picks.get(state["text"])
        value = NONE_OPTION
        if name is not None:
            value = next(o.value for o in question.options if name in o.description)
        rest = (1 - self.probability) / (len(question.options) - 1)
        probabilities = {
            o.value: self.probability if o.value == value else rest
            for o in question.options
        }
        return Decision(
            model=f"{self.model_id}-2026-01-01",
            answers=[
                Answer(
                    name=question.name,
                    kind="choice",
                    value=value,
                    probabilities=probabilities if self.probabilities else {},
                )
            ],
        )


def results(*queries):
    """Retrieved expressions per query: (query, [concept ids or (id, text)])."""
    return RetrieverResults(
        queries=[q for q, _ in queries],
        results=[
            [
                expression(c) if isinstance(c, str) else expression(c[0], text=c[1])
                for c in candidates
            ]
            for _, candidates in queries
        ],
    )


def test_fake_is_a_decision_model():
    assert isinstance(FakeDecisionModel(), DecisionModel)


def test_question_offers_the_candidates_and_none():
    model = FakeDecisionModel({"creatinine urine": "in Urine"})
    selector = DecisionSelector(model)

    (selected,) = selector.select(
        results(("creatinine urine", [SERUM, URINE, HB]))
    ).results

    state, (question,) = model.calls[0]
    assert state == {"text": "creatinine urine"}
    assert question.kind == "choice"
    assert question.values == ["c1", "c2", "c3", NONE_OPTION]
    assert question.options[0].description == (
        "Creatinine [Mass/volume] in Serum or Plasma (LOINC 2160-0, Measurement)"
    )
    assert selected.std_concept_id == URINE
    assert selected.result_list_index == 1
    # The probability is the score: the mapper's confidence_score is 1 - distance.
    assert 1 - selected.distance == pytest.approx(0.8)
    trace = json.loads(selected.trace)
    assert trace[0]["model"] == "fake/decider-1-2026-01-01"
    assert trace[0]["probabilities"]["c2"] == pytest.approx(0.8)


def test_synonyms_of_a_concept_are_one_option():
    model = FakeDecisionModel({"Hb": "Hemoglobin"})
    selector = DecisionSelector(model)

    (selected,) = selector.select(
        results(("Hb", [SERUM, (HB, "Hb Bld-mCnc"), (HB, "Haemoglobin")]))
    ).results

    (question,) = model.calls[0][1]
    assert question.values == ["c1", "c2", NONE_OPTION]
    assert question.options[1].description.endswith(
        "; matched: 'Hb Bld-mCnc'; 'Haemoglobin'"
    )
    # The best-ranked expression of the concept is returned.
    assert selected.result_list_index == 1
    assert selected.expression == "Hb Bld-mCnc"


def test_top_k_counts_distinct_concepts():
    model = FakeDecisionModel()
    selector = DecisionSelector(model, top_k=2)
    selector.select(results(("x", [SERUM, SERUM, URINE, HB])))
    (question,) = model.calls[0][1]
    assert question.values == ["c1", "c2", NONE_OPTION]


def test_none_refusals_errors_and_no_candidates_select_nothing(caplog):
    none = DecisionSelector(FakeDecisionModel()).select(results(("x", [SERUM])))
    assert isinstance(none.results[0], EmptySelectionMetadata)
    assert json.loads(none.results[0].trace)[0]["answer"] == NONE_OPTION

    refused = DecisionSelector(FakeDecisionModel(refuse=True))
    assert isinstance(
        refused.select(results(("x", [SERUM]))).results[0], EmptySelectionMetadata
    )

    with caplog.at_level(logging.WARNING):
        failed = DecisionSelector(FakeDecisionModel(error=True)).select(
            results(("x", [SERUM]))
        )
    assert failed.results[0].rationale == "No decision: bad answer"
    assert "No decision for 'x'" in caplog.text

    model = FakeDecisionModel()
    empty = DecisionSelector(model).select(results(("x", [])))
    assert isinstance(empty.results[0], EmptySelectionMetadata)
    assert model.calls == []


def test_queries_are_decided_concurrently_in_order():
    model = FakeDecisionModel({"a": "in Urine", "b": "Hemoglobin"})
    selector = DecisionSelector(model, max_concurrency=2)
    selected = selector.select(
        results(("a", [SERUM, URINE]), ("b", [HB]), ("c", [SERUM]))
    ).results
    assert [s.std_concept_id for s in selected] == [URINE, HB, None]
    assert model.max_running == 2


def test_llm_models_give_no_score(caplog):
    class FixedLLM:
        model_id = "local/model"

        async def generate(self, *, system, user, output_type):
            return output_type.model_validate({"concept": "c1"})

    with caplog.at_level(logging.WARNING):
        selector = DecisionSelector(LLMDecisionModel(FixedLLM()))
    assert not selector.has_probabilities
    assert "gives no probabilities" in caplog.text

    (selected,) = selector.select(results(("x", [SERUM]))).results
    assert selected.std_concept_id == SERUM
    assert selected.distance is None  # confidence_score None, not 1


def test_invalid_arguments():
    with pytest.raises(ValueError, match="top_k"):
        DecisionSelector(FakeDecisionModel(), top_k=255)
    with pytest.raises(ValueError, match="model_kwargs"):
        DecisionSelector(FakeDecisionModel(), model_kwargs={"api_key": "x"})
    with pytest.raises(TypeError, match="DecisionModel"):
        DecisionSelector(object())


# --- registry ------------------------------------------------------------------------


def test_registry_builds_decision_selectors(monkeypatch):
    created = []

    def fake_create(model_id, **kwargs):
        created.append((model_id, kwargs))
        return FakeDecisionModel(model_id=model_id)

    monkeypatch.setattr("aatm.llm.create_decision_model", fake_create)
    assert "decision-openrouter/typesafe/jev-1.13" in SELECTORS_REGISTRY

    selector = load_selector("decision-openrouter/typesafe/jev-1.13", top_k=5)
    assert isinstance(selector, DecisionSelector)
    assert selector.model_id == "openrouter/typesafe/jev-1.13"
    assert selector.top_k == 5

    # Any other decision model id, not only the registered ones.
    load_selector(
        "decision-jev-compatible/my-jev-2",
        model_kwargs={"base_url": "http://localhost:9000/decide"},
    )
    assert created[-1] == (
        "jev-compatible/my-jev-2",
        {"base_url": "http://localhost:9000/decide"},
    )
    with pytest.raises(ValueError, match="not found"):
        load_selector("not-a-selector")


def test_terminology_mapper_accepts_a_decision_selector_name(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "aatm.llm.create_decision_model",
        lambda model_id, **kw: FakeDecisionModel(
            {"creatinine urine": "in Urine"}, model_id=model_id
        ),
    )
    mapper = TerminologyMapper(
        retriever=FakeRetriever(default=(SERUM, URINE)),
        selector="decision-openrouter/typesafe/jev-1.13",
        output_dir=tmp_path,
    )
    df = mapper.map(expressions=["creatinine urine"], save_to_disk=False)
    assert list(df.target_concept_id) == [URINE]
    assert list(df.confidence_score) == pytest.approx([0.8])


# --- cache identity ------------------------------------------------------------------


def test_cache_identity_changes_with_the_model_and_question():
    base = DecisionSelector(FakeDecisionModel()).cache_identity()
    assert base["model_id"] == "fake/decider-1"
    assert "max_concurrency" not in base
    assert (
        DecisionSelector(FakeDecisionModel(), max_concurrency=1).cache_identity()
        == base
    )
    for other in (
        DecisionSelector(FakeDecisionModel(model_id="fake/decider-2")),
        DecisionSelector(FakeDecisionModel(), top_k=3),
        DecisionSelector(FakeDecisionModel(), instructions="Pick one."),
    ):
        assert other.cache_identity() != base


def test_mapper_cache_identity_includes_the_decision_selector(tmp_path):
    def coder(model_id):
        mapper = TerminologyMapper(
            retriever=FakeRetriever(),
            selector=DecisionSelector(FakeDecisionModel(model_id=model_id)),
            output_dir=tmp_path,
        )
        return TerminologyMapperCoder(mapper).cache_identity()

    assert coder("fake/decider-1") == coder("fake/decider-1")
    assert coder("fake/decider-1") != coder("fake/decider-2")
    selector = DecisionSelector(FakeDecisionModel(), instructions="Pick one.")
    described = describe_component(selector, _depth=1)
    assert described["identity"]["instructions"] == "Pick one."


# --- OMOP builder ---------------------------------------------------------------------


def test_min_score_thresholds_the_probability(tmp_path):
    def build(probability, min_score):
        mapper = TerminologyMapper(
            retriever=FakeRetriever(default=(SERUM, URINE)),
            selector=DecisionSelector(
                FakeDecisionModel({"creatinine": "in Serum"}, probability=probability)
            ),
            output_dir=tmp_path,
        )
        builder = OmopTableBuilder(
            lab_definitions(min_score=min_score), vocabulary=UnitLookup(), mapper=mapper
        )
        return run(builder, "creatinine 80", [lab("e1", "creatinine")])

    kept = build(0.9, 0.8)
    assert kept.rows["measurement"][0].measurement_concept_id == int(SERUM)
    df = kept.mappings_dataframe()
    score = df[df.column == "measurement_concept_id"].score.iloc[0]
    assert score == pytest.approx(0.9)

    rejected = build(0.6, 0.8)
    assert rejected.rows["measurement"][0].measurement_concept_id == 0
    assert "low_confidence" in set(rejected.issues_dataframe().reason)


# --- TerminologyMapper.amap ----------------------------------------------------------


AMAP_CSV = (
    "source_code,source_concept_id,source_vocabulary_id,source_code_description,"
    "valid_start_date,valid_end_date,invalid_reason\n"
    "X1,0,LOCAL,scored,,,\n"
    "X2,0,LOCAL,unscored,,,\n"
    "X3,0,LOCAL,unmapped,,,\n"
)


class AsyncFixedSelector(BaseSelector):
    """An async selector (as `amap` needs): scored, unscored, and no selection."""

    async def select(self, results):  # type: ignore[override]
        def chosen(distance):
            data = results.results[0][0].model_dump(exclude={"expression_id"})
            data["distance"] = distance
            return SelectedExpressionMetadata(**data, result_list_index=0)

        return SelectorResults(
            results=[chosen(0.25), chosen(None), EmptySelectionMetadata()],
            queries=results.queries,
        )


def test_amap_gives_no_confidence_without_a_distance(tmp_path):
    from aatm.aio.translators import EmptyTranslator

    (tmp_path / "in.csv").write_text(AMAP_CSV)
    mapper = TerminologyMapper(
        translator=EmptyTranslator(),
        retriever=FakeRetriever(default=(SERUM,)),
        selector=AsyncFixedSelector(),
        output_dir=tmp_path,
    )
    df = asyncio.run(mapper.amap(file_path=tmp_path / "in.csv"))
    scores = list(df.confidence_score)
    # A selection without a distance, or no selection, is not full confidence.
    assert scores[0] == pytest.approx(0.75)
    assert pd.isna(scores[1]) and pd.isna(scores[2])

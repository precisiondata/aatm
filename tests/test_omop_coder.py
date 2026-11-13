"""The coder= parameter, the coder protocols and the pipeline's extractor arguments."""

import pytest

from aatm.data_models.data_models import RetrievedExpressionMetadata, RetrieverResults
from aatm.data_models.extractors import ExtractedEntityCandidates
from aatm.extractors import DecisionAttributeLabeller
from aatm.omop import (
    CacheableCoder,
    CachedCoder,
    CodedText,
    ContextualTextCoder,
    OmopExtractionPipeline,
    OmopTableBuilder,
    TerminologyMapperCoder,
    TextCoder,
)
from aatm.omop.coding import as_coder
from aatm.retrievers import BaseRetriever
from aatm.terminology_mapper import TerminologyMapper
from test_attribute_labeller import FakeDecisionModel, FakeLLM
from test_omop_cache import ContextCoder, IdentifiedCoder
from test_omop_coding import (
    UNIT_CONCEPTS,
    FakeCoder,
    UnitLookup,
    lab,
    lab_definitions,
    run,
)
from test_omop_mapping import FakeLookup, concept, ecog_definitions

# --- coder= and the deprecated mapper= -------------------------------------------


def test_builder_takes_a_coder():
    coder = FakeCoder()
    builder = OmopTableBuilder(lab_definitions(), vocabulary=UnitLookup(), coder=coder)
    assert builder.coder is coder


def test_mapper_still_works_and_warns():
    coder = FakeCoder()
    with pytest.warns(DeprecationWarning, match="pass coder= instead") as warned:
        builder = OmopTableBuilder(
            lab_definitions(), vocabulary=UnitLookup(), mapper=coder
        )
    assert builder.coder is coder
    # The warning points at the caller, not at aatm.
    assert warned[0].filename == __file__


def test_pipeline_mapper_still_works_and_warns_once():
    coder = FakeCoder()
    with pytest.warns(DeprecationWarning) as warned:
        pipeline = OmopExtractionPipeline(
            lab_definitions(), extractor=object(), vocabulary=UnitLookup(), mapper=coder
        )
    assert pipeline.builder.coder is coder
    assert [w.filename for w in warned] == [__file__]


def test_coder_and_mapper_together_are_rejected():
    with pytest.raises(TypeError, match="not both"):
        OmopTableBuilder(
            lab_definitions(),
            vocabulary=UnitLookup(),
            coder=FakeCoder(),
            mapper=FakeCoder(),
        )


class RetrieverOnly(BaseRetriever):
    def retrieve(self, queries, *args, **kwargs):  # pragma: no cover - never called
        raise AssertionError("a retriever is not a coder")


@pytest.mark.parametrize("wrong", ["text-embedding-3-small", RetrieverOnly(), 42])
def test_a_wrong_coder_fails_when_the_builder_is_created(wrong):
    with pytest.raises(TypeError, match="must be a TerminologyMapper or a TextCoder"):
        OmopTableBuilder(lab_definitions(), vocabulary=UnitLookup(), coder=wrong)


def test_a_wrong_coder_fails_when_the_pipeline_is_created():
    with pytest.raises(TypeError, match="got RetrieverOnly"):
        OmopExtractionPipeline(
            lab_definitions(),
            extractor=object(),
            vocabulary=UnitLookup(),
            coder=RetrieverOnly(),
        )


# --- the protocols ------------------------------------------------------------------


def test_protocols_match_the_built_in_and_custom_coders(tmp_path):
    mapper_coder = TerminologyMapper(
        retriever=Retriever(), output_dir=tmp_path
    ).as_coder()
    assert isinstance(mapper_coder, TerminologyMapperCoder)
    assert isinstance(mapper_coder, ContextualTextCoder)
    assert isinstance(mapper_coder, CacheableCoder)

    plain = FakeCoder()
    assert isinstance(plain, TextCoder)
    assert not isinstance(plain, ContextualTextCoder)
    assert not isinstance(plain, CacheableCoder)

    assert isinstance(ContextCoder(), ContextualTextCoder)
    assert isinstance(IdentifiedCoder(), CacheableCoder)


def test_as_coder_wraps_a_mapper_and_keeps_a_coder(tmp_path):
    mapper = TerminologyMapper(retriever=Retriever(), output_dir=tmp_path)
    wrapped = as_coder(mapper)
    assert isinstance(wrapped, TerminologyMapperCoder)
    assert wrapped.mapper is mapper
    coder = FakeCoder()
    assert as_coder(coder) is coder


def test_contexts_are_passed_only_to_contextual_coders():
    contextual = ContextCoder()
    run(
        OmopTableBuilder(lab_definitions(), vocabulary=UnitLookup(), coder=contextual),
        "Hb 11",
        [lab("e1", "Hb")],
    )
    assert [c.text for c in contextual.calls[0]] == ["Hb"]

    class Switchable(ContextCoder):
        uses_context = False

        def code(self, texts, domain=None, vocabularies=None, contexts=None):
            assert contexts is None
            return [CodedText(text=t) for t in texts]

    run(
        OmopTableBuilder(
            lab_definitions(), vocabulary=UnitLookup(), coder=Switchable()
        ),
        "Hb 11",
        [lab("e1", "Hb")],
    )


def test_caching_needs_a_cacheable_coder(tmp_path):
    with pytest.raises(TypeError, match="CacheableCoder"):
        CachedCoder(FakeCoder(), tmp_path / "cache.db")


# --- TerminologyMapperCoder's standard-only assumption -------------------------------


class Retriever(BaseRetriever):
    """Returns one LOINC candidate, of the given concept id, per query."""

    def __init__(self, concept_id=3000963, domain="Measurement"):
        self.concept_id = concept_id
        self.domain = domain

    def retrieve(self, queries, where=None, *args, **kwargs):
        return RetrieverResults(
            queries=queries,
            results=[
                [
                    RetrievedExpressionMetadata(
                        expression=q,
                        expression_concept_id="1",
                        expression_origin="standard_concept",
                        std_concept_id=str(self.concept_id),
                        std_concept_name="Hemoglobin",
                        std_vocabulary_id="LOINC",
                        std_vocabulary_code="718-7",
                        # What the index says, which may be wrong.
                        std_domain_id=self.domain,
                        distance=0.25,
                    )
                ]
                for q in queries
            ],
        )


class LookupWithNonStandard(FakeLookup):
    def __init__(self):
        super().__init__(
            concepts=UNIT_CONCEPTS
            + [
                concept(
                    3000964,
                    "718-8",
                    "Hemoglobin, non-standard",
                    "Measurement",
                    "LOINC",
                    standard=None,
                ),
                concept(4000001, "123", "Anemia", "Condition", "SNOMED"),
            ],
            maps_to={},
        )


def build_with_retriever(tmp_path, retriever):
    mapper = TerminologyMapper(retriever=retriever, output_dir=tmp_path)
    builder = OmopTableBuilder(
        lab_definitions(), vocabulary=LookupWithNonStandard(), coder=mapper
    )
    return run(builder, "Hb 11", [lab("e1", "Hb")])


def test_a_non_standard_concept_from_a_custom_index_is_stored_as_0(tmp_path):
    tables = build_with_retriever(tmp_path, Retriever(concept_id=3000964))

    assert tables.rows["measurement"][0].measurement_concept_id == 0
    (issue,) = tables.issues
    assert (issue.column, issue.reason) == ("measurement_concept_id", "unmapped")
    assert "not a standard concept" in issue.detail
    (mapping,) = [m for m in tables.mappings if m.method == "map"]
    assert (mapping.concept_id, mapping.stored) == (3000964, 0)


def test_the_domain_of_a_coded_concept_comes_from_the_vocabulary(tmp_path):
    # The index claims Measurement; the vocabulary knows it is a Condition.
    tables = build_with_retriever(tmp_path, Retriever(concept_id=4000001))

    (issue,) = tables.issues
    assert (issue.column, issue.reason) == (
        "measurement_concept_id",
        "domain_mismatch",
    )


def test_a_standard_concept_from_the_index_is_stored(tmp_path):
    tables = build_with_retriever(tmp_path, Retriever())
    assert tables.rows["measurement"][0].measurement_concept_id == 3000963
    assert tables.issues == []


# --- extractor arguments ----------------------------------------------------------


def no_entities():
    return ExtractedEntityCandidates.model_validate({"entities": []})


def test_extractor_settings_are_forwarded():
    labeller = DecisionAttributeLabeller(FakeDecisionModel(), mode="verify")
    pipeline = OmopExtractionPipeline(
        ecog_definitions(),
        model_id="fake/extractor",
        vocabulary=FakeLookup(),
        llm=FakeLLM(no_entities()),
        attribute_labeller=labeller,
        max_concurrency=3,
        settings={"temperature": 0},
        on_error="record",
    )
    assert pipeline.extractor.attribute_labeller is labeller
    assert pipeline.extractor.max_concurrency == 3
    assert pipeline.extractor.on_error == "record"


def test_a_misspelt_extractor_argument_lists_the_valid_ones():
    with pytest.raises(TypeError) as raised:
        OmopExtractionPipeline(
            ecog_definitions(),
            model_id="fake/extractor",
            vocabulary=FakeLookup(),
            max_concurency=3,
        )
    message = str(raised.value)
    assert "'max_concurency'" in message
    assert "attribute_labeller" in message and "max_concurrency" in message
    # Set by the pipeline, so not offered.
    assert "entity_definitions" not in message.split("Valid:")[1]


@pytest.mark.parametrize(
    "kwargs", [{"max_concurrency": 3}, {"instructions": "Be brief."}]
)
def test_extractor_arguments_are_rejected_with_an_extractor(kwargs):
    with pytest.raises(TypeError, match="would be ignored with extractor="):
        OmopExtractionPipeline(
            ecog_definitions(), extractor=object(), vocabulary=FakeLookup(), **kwargs
        )

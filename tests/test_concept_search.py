"""Agentic concept search: tools, agent loop, `AgenticCoder` and `AgenticSelector`."""

import asyncio
import json
import sqlite3

import pytest

from aatm.concept_search import (
    AgentStep,
    CodingContext,
    ConceptSearchAgent,
    ConceptSearchTools,
    VocabularySearch,
)
from aatm.concept_search.selector import AgenticSelector, scope_from_where
from aatm.data_models import SourceConcept
from aatm.data_models.data_models import RetrievedExpressionMetadata, RetrieverResults
from aatm.llm import LLMResponseError
from aatm.llm.usage import record_usage, track_usage
from aatm.omop import AgenticCoder, OmopTableBuilder, TerminologyMapperCoder
from aatm.omop.coding import retriever_filter
from aatm.retrievers import BaseRetriever
from aatm.terminology_mapper import TerminologyMapper
from test_omop_coding import UnitLookup, lab, lab_definitions, part, run

# --- a small vocabulary database --------------------------------------------------

CONCEPTS = [
    # id, name, domain, vocabulary, code, standard
    (
        "3016723",
        "Creatinine [Mass/volume] in Serum or Plasma",
        "Measurement",
        "LOINC",
        "2160-0",
        "S",
    ),
    (
        "3020564",
        "Creatinine [Moles/volume] in Serum or Plasma",
        "Measurement",
        "LOINC",
        "14682-9",
        "S",
    ),
    (
        "3007760",
        "Creatinine [Mass/volume] in Urine",
        "Measurement",
        "LOINC",
        "2161-8",
        "S",
    ),
    (
        "3000963",
        "Hemoglobin [Mass/volume] in Blood",
        "Measurement",
        "LOINC",
        "718-7",
        "S",
    ),
    ("4324383", "Creatinine measurement", "Measurement", "SNOMED", "70901006", "S"),
    ("201826", "Type 2 diabetes mellitus", "Condition", "SNOMED", "44054006", "S"),
    ("45576876", "Type 2 diabetes mellitus", "Condition", "ICD10CM", "E11", None),
    ("40775801", "Creatinine", "Measurement", "LOINC", "LP14355-9", None),
    ("1033306", "Ser/Plas", "Measurement", "LOINC", "LP7567-6", None),
    ("1029650", "MCnc", "Measurement", "LOINC", "LP6827-2", None),
    ("9999999", "Old creatinine test", "Measurement", "LOINC", "0000-0", "S"),
]
SYNONYMS = [("3000963", "Hb Bld-mCnc"), ("3016723", "Creatinine SerPl-mCnc")]
RELATIONSHIPS = [
    ("3016723", "Has component", "40775801"),
    ("3016723", "Has system", "1033306"),
    ("3016723", "Has property", "1029650"),
    ("45576876", "Maps to", "201826"),
]
ANCESTORS = [  # ancestor, descendant, min levels
    ("4324383", "3016723", "1"),
    ("4324383", "3020564", "1"),
]


@pytest.fixture
def vocab_db(tmp_path):
    path = tmp_path / "omop.db"
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE concept (concept_id TEXT, concept_name TEXT, domain_id TEXT, "
        "vocabulary_id TEXT, concept_class_id TEXT, standard_concept TEXT, "
        "concept_code TEXT, valid_start_date TEXT, valid_end_date TEXT, invalid_reason TEXT)"
    )
    for cid, name, domain, vocab, code, std in CONCEPTS:
        invalid = "D" if cid == "9999999" else None
        con.execute(
            "INSERT INTO concept VALUES (?,?,?,?,?,?,?,?,?,?)",
            (cid, name, domain, vocab, "x", std, code, "", "", invalid),
        )
    con.execute(
        "CREATE TABLE concept_synonym (concept_id TEXT, concept_synonym_name TEXT, language_concept_id TEXT)"
    )
    con.executemany("INSERT INTO concept_synonym VALUES (?,?,'4180186')", SYNONYMS)
    con.execute(
        "CREATE TABLE concept_relationship (concept_id_1 TEXT, concept_id_2 TEXT, "
        "relationship_id TEXT, valid_start_date TEXT, valid_end_date TEXT, invalid_reason TEXT)"
    )
    con.executemany(
        "INSERT INTO concept_relationship VALUES (?,?,?,'','',NULL)",
        [(a, b, rel) for a, rel, b in RELATIONSHIPS],
    )
    con.execute(
        "CREATE TABLE concept_ancestor (ancestor_concept_id TEXT, descendant_concept_id TEXT, "
        "min_levels_of_separation TEXT, max_levels_of_separation TEXT)"
    )
    con.executemany(
        "INSERT INTO concept_ancestor VALUES (?,?,?,?)",
        [(a, d, n, n) for a, d, n in ANCESTORS],
    )
    con.commit()
    con.close()
    return path


def expression(cid, distance=0.3, text=None):
    row = next(c for c in CONCEPTS if c[0] == cid)
    return RetrievedExpressionMetadata(
        expression=text or row[1],
        expression_concept_id=cid,
        expression_origin="standard_concept",
        std_concept_id=cid,
        std_concept_name=row[1],
        std_vocabulary_id=row[3],
        std_vocabulary_code=row[4],
        std_domain_id=row[2],
        distance=distance,
    )


class FakeRetriever(BaseRetriever):
    """Returns fixed expressions per query (default: the urine creatinine)."""

    def __init__(self, by_query=None, default=("3007760",)):
        self.by_query = by_query or {}
        self.default = default
        self.calls: list[tuple] = []

    def retrieve(self, queries, where=None, top_k=None, *args, **kwargs):
        self.calls.append((list(queries), where, top_k))
        return RetrieverResults(
            queries=queries,
            results=[
                [expression(c) for c in self.by_query.get(q, self.default)]
                for q in queries
            ],
        )


class ScriptedLLM:
    """Answers with the given steps in order; records prompts and usage."""

    model_id = "fake/scripted"

    def __init__(self, steps):
        self.steps = list(steps)
        self.prompts: list[str] = []
        self.systems: list[str] = []

    async def generate(self, *, system, user, output_type):
        assert output_type is AgentStep
        self.prompts.append(user)
        self.systems.append(system)
        record_usage(100, 10)
        if not self.steps:
            raise LLMResponseError("no more steps")
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def step(
    action, query=None, concept_id=None, direction=None, confidence=None, reasoning="r"
):
    return AgentStep(
        reasoning=reasoning,
        action=action,
        query=query,
        concept_id=concept_id,
        direction=direction,
        confidence=confidence,
    )


def make_agent(steps, vocab_db, retriever=None, **kwargs):
    llm = ScriptedLLM(steps)
    tools = ConceptSearchTools(retriever or FakeRetriever(), vocab_db)
    return ConceptSearchAgent(llm, tools, **kwargs), llm


# --- tools --------------------------------------------------------------------------


def test_lexical_search_matches_names_and_synonyms_in_scope(vocab_db):
    vocab = VocabularySearch(vocab_db)
    found = vocab.search_names(
        "creatinine serum", domain="Measurement", vocabularies=["LOINC"]
    )
    assert [c.concept_id for c in found] == [3016723, 3020564]
    # Synonyms, standard only, invalid concepts left out.
    assert [c.concept_id for c in vocab.search_names("hb bld")] == [3000963]
    assert 9999999 not in [c.concept_id for c in vocab.search_names("creatinine")]
    assert 40775801 not in [c.concept_id for c in vocab.search_names("creatinine")]


def test_code_lookup_follows_maps_to(vocab_db):
    tools = ConceptSearchTools(None, vocab_db)
    found = tools.code_lookup("e11", domain="Condition")
    assert [(c.concept_id, c.standard) for c in found] == [
        (45576876, False),
        (201826, True),
    ]
    # The standard target is out of scope for another domain.
    assert [c.concept_id for c in tools.code_lookup("E11", domain="Measurement")] == [
        45576876
    ]


def test_concept_details_show_loinc_axes(vocab_db):
    details = ConceptSearchTools(None, vocab_db).concept_details(3016723)
    assert "LOINC axes: component=Creatinine; property=MCnc; system=Ser/Plas" in details
    assert "synonyms: Creatinine SerPl-mCnc" in details
    assert "No concept" in ConceptSearchTools(None, vocab_db).concept_details(1)


def test_hierarchy(vocab_db):
    tools = ConceptSearchTools(None, vocab_db)
    assert [c.concept_id for c in tools.hierarchy(3016723, "parents")] == [4324383]
    children = tools.hierarchy(4324383, "children", vocabularies=["LOINC"])
    assert {c.concept_id for c in children} == {3016723, 3020564}
    assert tools.hierarchy(4324383, "children", vocabularies=["SNOMED"]) == []


def test_semantic_search_filters_and_dedupes():
    retriever = FakeRetriever(default=("3016723", "3016723", "3000963"))
    tools = ConceptSearchTools(retriever)
    found = tools.semantic_search(
        "creat", domain="Measurement", vocabularies=["LOINC"], top_k=5
    )
    assert [c.concept_id for c in found] == [3016723, 3000963]
    assert retriever.calls == [
        (["creat"], retriever_filter("Measurement", ["LOINC"]), 10)
    ]
    assert tools.actions == ["semantic_search"]


# --- agent --------------------------------------------------------------------------


def test_agent_searches_then_finishes(vocab_db):
    agent, llm = make_agent(
        [
            step("lexical_search", query="creatinine serum"),
            step("concept_details", concept_id=3016723),
            step(
                "finish",
                concept_id=3016723,
                confidence="high",
                reasoning="mg/dL is mass/volume",
            ),
        ],
        vocab_db,
    )
    context = CodingContext(
        text="creatinine",
        related={"lab_unit": "mg/dl"},
        sentence="creatinine 1.4 mg/dl",
    )
    result = asyncio.run(
        agent.search(context, domain="Measurement", vocabularies=["LOINC"])
    )

    assert result.concept.concept_id == 3016723
    assert (result.confidence, result.score) == ("high", 0.9)
    assert result.rationale == "mg/dL is mass/volume"
    assert [t["action"] for t in result.trace] == [
        "semantic_search",
        "lexical_search",
        "concept_details",
        "finish",
    ]
    assert result.trace[0]["results"] == ["3007760 Creatinine [Mass/volume] in Urine"]
    assert result.tool_calls == 3
    assert (result.llm_calls, result.input_tokens, result.output_tokens) == (3, 300, 30)
    assert not result.budget_exhausted
    # The context and the scope reach the model.
    assert "lab_unit='mg/dl'" in llm.prompts[0]
    assert "Sentence: 'creatinine 1.4 mg/dl'" in llm.prompts[0]


def test_agent_rejects_concepts_outside_the_scope(vocab_db):
    agent, llm = make_agent(
        [
            step("finish", concept_id=201826, confidence="high"),  # a Condition
            step("finish", concept_id=3007760, confidence="medium"),
        ],
        vocab_db,
    )
    result = asyncio.run(
        agent.search(CodingContext(text="creatinine"), domain="Measurement")
    )
    assert result.concept.concept_id == 3007760
    assert (
        "is not a standard concept of the Measurement domain"
        in result.trace[1]["error"]
    )
    assert "Rejected" in llm.prompts[1]


def test_agent_accepts_a_valid_concept_it_was_not_shown(vocab_db):
    agent, _ = make_agent(
        [step("finish", concept_id=3020564, confidence="low")], vocab_db
    )
    result = asyncio.run(
        agent.search(CodingContext(text="creatinine"), domain="Measurement")
    )
    assert result.concept.concept_name == "Creatinine [Moles/volume] in Serum or Plasma"


def test_agent_can_conclude_there_is_no_concept(vocab_db):
    agent, _ = make_agent([step("finish", reasoning="Not a lab test")], vocab_db)
    result = asyncio.run(agent.search(CodingContext(text="xyz")))
    assert result.concept is None and result.score is None
    assert result.rationale == "Not a lab test"


def test_budget_caps_tool_calls_and_confidence(vocab_db):
    agent, llm = make_agent(
        [
            step("lexical_search", query="creatinine"),
            step("lexical_search", query="ignored, the budget is spent"),
            step("finish", concept_id=3016723, confidence="high"),
        ],
        vocab_db,
        max_tool_calls=1,
    )
    result = asyncio.run(agent.search(CodingContext(text="creatinine")))
    assert result.tool_calls == 2  # initial search + 1
    assert result.budget_exhausted
    assert result.concept is None  # the second search was not allowed
    assert "must finish now" in llm.prompts[1]

    agent, _ = make_agent(
        [
            step("lexical_search", query="creatinine"),
            step("finish", concept_id=3016723, confidence="high"),
        ],
        vocab_db,
        max_tool_calls=1,
    )
    result = asyncio.run(agent.search(CodingContext(text="creatinine")))
    assert result.concept.concept_id == 3016723
    assert (result.confidence, result.score, result.budget_exhausted) == (
        "low",
        0.3,
        True,
    )


def test_agent_reports_errors_without_raising(vocab_db):
    agent, _ = make_agent([LLMResponseError("refused")], vocab_db)
    result = asyncio.run(agent.search(CodingContext(text="creatinine")))
    assert result.concept is None
    assert result.error == "LLMResponseError: refused"


def test_unavailable_actions_and_missing_arguments_are_retried(vocab_db):
    llm = ScriptedLLM(
        [
            step("lexical_search", query="x"),  # no vocabulary: unavailable
            step("semantic_search"),  # no query
            step("finish", concept_id=3007760, confidence="medium"),
        ]
    )
    agent = ConceptSearchAgent(llm, ConceptSearchTools(FakeRetriever()))
    result = asyncio.run(agent.search(CodingContext(text="creatinine")))
    assert result.concept.concept_id == 3007760
    assert [t.get("error", "")[:20] for t in result.trace[1:3]] == [
        "lexical_search is no",
        "semantic_search need",
    ]


def test_identical_contexts_are_searched_once(vocab_db):
    agent, llm = make_agent(
        [step("finish", concept_id=3007760, confidence="medium")] * 2, vocab_db
    )
    urine = CodingContext(text="creatinine", related={"specimen": "urine"})
    results = asyncio.run(
        agent.search_many(
            [
                (urine, None),
                (CodingContext(text="creatinine", related={"specimen": "urine"}), None),
            ]
        )
    )
    assert len(llm.prompts) == 1
    assert results[0] is results[1]


def test_track_usage_nests_and_isolates_tasks():
    async def call(n):
        with track_usage() as usage:
            for _ in range(n):
                record_usage(1, 2)
                await asyncio.sleep(0)
        return usage

    async def main():
        with track_usage() as outer:
            a, b = await asyncio.gather(call(2), call(3))
        return outer, a, b

    outer, a, b = asyncio.run(main())
    assert (a.calls, b.calls, outer.calls) == (2, 3, 5)
    assert outer.output_tokens == 10


# --- OMOP tables: AgenticCoder --------------------------------------------------------


def agentic_coder(steps, vocab_db):
    coder = AgenticCoder(
        ScriptedLLM(steps), retriever=FakeRetriever(), vocabulary=vocab_db
    )
    return coder, coder.agent.llm


def test_agentic_coder_gets_the_context_of_each_mention(vocab_db):
    coder, llm = agentic_coder(
        [
            step("finish", concept_id=3016723, confidence="high", reasoning="serum"),
            step("finish", concept_id=3007760, confidence="medium", reasoning="urine"),
        ],
        vocab_db,
    )
    builder = OmopTableBuilder(lab_definitions(), vocabulary=UnitLookup(), coder=coder)
    text = "creatinine 1.4 mg/dL\nurine: creatinine 80"
    tables = run(
        builder,
        text,
        [
            lab("e1", "creatinine", line="L1"),
            part("e2", "lab_value", "1.4", "e1", line="L1"),
            part("e3", "lab_unit", "mg/dL", "e1", line="L1"),
            lab("e4", "creatinine", line="L2"),
            part("e5", "lab_value", "80", "e4", line="L2"),
        ],
    )

    # Same text, different contexts: coded twice, each row with its own concept.
    rows = tables.rows["measurement"]
    assert {r.measurement_concept_id for r in rows} == {3016723, 3007760}
    prompts = "\n".join(llm.prompts)
    assert "lab_value='1.4'; lab_unit='mg/dL'" in prompts
    assert "Sentence: 'urine: creatinine 80'" in prompts
    assert "Target: measurement.measurement_concept_id" in prompts
    # The search is limited to the column's domain and the definition's vocabularies.
    assert "of the Measurement domain from LOINC" in llm.systems[0]

    df = tables.mappings_dataframe()
    df = df[df.column == "measurement_concept_id"]
    assert set(df.rationale) == {"serum", "urine"}
    assert all(trace[-1]["action"] == "finish" for trace in df.trace)
    assert all(usage["llm_calls"] == 1 for usage in df.usage)
    assert set(df.score) == {0.9, 0.6}


def test_min_score_applies_to_the_agent_confidence(vocab_db):
    coder, _ = agentic_coder(
        [step("finish", concept_id=3007760, confidence="low")], vocab_db
    )
    builder = OmopTableBuilder(
        lab_definitions(min_score=0.5), vocabulary=UnitLookup(), coder=coder
    )
    tables = run(builder, "creatinine 80", [lab("e1", "creatinine")])
    assert tables.rows["measurement"][0].measurement_concept_id == 0
    assert "low_confidence" in set(tables.issues_dataframe().reason)


def test_plain_coders_still_code_each_text_once(vocab_db):
    from test_omop_coding import FakeCoder

    coder = FakeCoder()
    builder = OmopTableBuilder(lab_definitions(), vocabulary=UnitLookup(), coder=coder)
    run(
        builder,
        "Hb 11\nHb 12",
        [lab("e1", "Hb", line="L1"), lab("e2", "Hb", line="L2")],
    )
    assert coder.calls == [(["Hb"], "Measurement", ["LOINC"])]


# --- terminology mapping: AgenticSelector -------------------------------------------


def test_scope_from_where():
    assert scope_from_where(None) == (None, None)
    assert scope_from_where(retriever_filter("Measurement", ["LOINC"])) == (
        "Measurement",
        ["LOINC"],
    )
    assert scope_from_where({"std_domain_id": "Condition"}) == ("Condition", None)
    with pytest.raises(ValueError, match="understands filters"):
        scope_from_where({"expression_origin": "standard_concept"})


def test_agentic_selector_in_the_terminology_mapper(vocab_db, tmp_path):
    llm = ScriptedLLM(
        [
            step("code_lookup", query="E11"),
            step(
                "finish",
                concept_id=201826,
                confidence="high",
                reasoning="ICD10CM E11 maps to it",
            ),
            step("finish", concept_id=3000963, confidence="medium"),
        ]
    )
    retriever = FakeRetriever(by_query={"Hb": ("3000963",)}, default=("4324383",))
    selector = AgenticSelector(llm, vocabulary=vocab_db, max_concurrency=1)
    mapper = TerminologyMapper(
        retriever=retriever, selector=selector, output_dir=tmp_path
    )

    df = mapper.map(
        expressions=[
            SourceConcept(
                source_code="E11",
                source_vocabulary_id="ICD10CM",
                source_code_description="DM2",
            ),
            "Hb",
        ],
        save_to_disk=False,
    )

    # The selector searched with the mapper's retriever, and saw the source code.
    assert selector.agent.tools.retriever is retriever
    assert "source_code='E11'; source_vocabulary_id='ICD10CM'" in llm.prompts[0]
    assert "[retrieved candidates for 'DM2']" in llm.prompts[0]
    assert list(df.target_concept_id) == ["201826", "3000963"]
    assert list(df.confidence_score) == pytest.approx([0.9, 0.6])
    assert df.mapping_rationale[0] == "ICD10CM E11 maps to it"
    assert json.loads(df.mapping_trace[0])[1]["action"] == "code_lookup"


def test_selector_returns_the_retrieved_expression_when_it_is_chosen(vocab_db):
    llm = ScriptedLLM([step("finish", concept_id=3000963, confidence="high")])
    selector = AgenticSelector(llm, retriever=FakeRetriever(), vocabulary=vocab_db)
    results = RetrieverResults(
        queries=["Hb"],
        results=[[expression("3016723"), expression("3000963", text="Hb Bld-mCnc")]],
    )
    (selected,) = selector.select(results).results
    assert selected.result_list_index == 1
    assert selected.expression == "Hb Bld-mCnc"
    assert selected.distance == pytest.approx(0.1)
    assert selected.usage["llm_calls"] == 1


def test_selector_rejects_vocabularies_outside_the_index():
    with pytest.raises(ValueError, match="only maps to"):
        AgenticSelector(ScriptedLLM([]), vocabularies=["ICD10CM"])


def test_terminology_mapper_coder_passes_contexts_to_an_agentic_selector(
    vocab_db, tmp_path
):
    llm = ScriptedLLM(
        [step("finish", concept_id=3016723, confidence="high", reasoning="mg/dL")]
    )
    selector = AgenticSelector(llm, vocabulary=vocab_db)
    mapper = TerminologyMapper(
        retriever=FakeRetriever(), selector=selector, output_dir=tmp_path
    )
    coder = TerminologyMapperCoder(mapper)
    assert coder.uses_context

    builder = OmopTableBuilder(
        lab_definitions(), vocabulary=UnitLookup(), coder=mapper
    )
    tables = run(
        builder,
        "creatinine 1.4 mg/dL",
        [lab("e1", "creatinine"), part("e2", "lab_unit", "mg/dL", "e1")],
    )
    assert "lab_unit='mg/dL'" in llm.prompts[0]
    assert tables.rows["measurement"][0].measurement_concept_id == 3016723
    df = tables.mappings_dataframe()
    assert df[df.column == "measurement_concept_id"].rationale.iloc[0] == "mg/dL"

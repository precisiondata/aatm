"""The persistent coding cache: exact reuse, identities, limits and clearing."""

import pytest
from typer.testing import CliRunner

from aatm.concept_search import CodingContext
from aatm.omop import (
    AgenticCoder,
    CachedCoder,
    CodedText,
    CodingCache,
    OmopTableBuilder,
)
from aatm.omop.cache import as_cache
from test_concept_search import FakeRetriever, ScriptedLLM, step, vocab_db  # noqa: F401
from test_omop_coding import (
    NOTE,
    FakeCoder,
    UnitLookup,
    lab,
    lab_definitions,
    make_builder,
    part,
    run,
)


def cached_builder(coder, cache):
    return OmopTableBuilder(
        lab_definitions(), vocabulary=UnitLookup(), mapper=coder, cache=cache
    )


class IdentifiedCoder(FakeCoder):
    """`FakeCoder` that can be cached."""

    def __init__(self, known=None, model="m1"):
        super().__init__(known)
        self.model = model

    def cache_identity(self):
        return {"coder": "IdentifiedCoder", "model": self.model}


class ContextCoder:
    """Reads context; codes every text to the same concept and records the contexts."""

    uses_context = True

    def __init__(self, error_for=()):
        self.calls: list[list[CodingContext]] = []
        self.error_for = set(error_for)

    def cache_identity(self):
        return {"coder": "ContextCoder"}

    def code(self, texts, domain=None, vocabularies=None, contexts=None):
        self.calls.append(list(contexts))
        return [
            CodedText(text=t, error="boom")
            if t in self.error_for
            else CodedText(text=t, concept_id=3000963, score=0.9)
            for t in texts
        ]


def hb_entities():
    return [
        lab("e1", "Hb"),
        part("e2", "lab_value", "11,2", "e1"),
        part("e3", "lab_unit", "g/dL", "e1"),
        lab("e4", "glucose"),
    ]


# --- reuse across runs ------------------------------------------------------------


def test_cache_is_off_by_default(tmp_path):
    builder = make_builder(IdentifiedCoder())
    assert builder.cache is None
    assert not isinstance(builder.coder, CachedCoder)
    assert run(builder, NOTE, hb_entities()).cache_stats is None


@pytest.mark.parametrize("off", [None, False])
def test_as_cache_off(off):
    assert as_cache(off) is None


def test_results_are_reused_across_runs(tmp_path):
    path = tmp_path / "cache.db"
    first = IdentifiedCoder()
    tables = run(cached_builder(first, path), NOTE, hb_entities())
    assert first.calls == [(["Hb", "glucose"], "Measurement", ["LOINC"])]
    assert (tables.cache_stats.hits, tables.cache_stats.misses) == (0, 2)
    assert not tables.mappings_dataframe().cached.any()

    # A new run: a new coder and builder, the same cache file.
    second = IdentifiedCoder()
    again = run(cached_builder(second, path), NOTE, hb_entities())
    assert second.calls == []
    assert (again.cache_stats.hits, again.cache_stats.misses) == (2, 0)
    assert again.cache_stats.hit_rate == 1.0
    assert [r.model_dump() for r in again.rows["measurement"]] == [
        r.model_dump() for r in tables.rows["measurement"]
    ]
    mappings = again.mappings_dataframe()
    assert mappings[mappings.method == "map"].cached.all()
    # Concept names are still added from the vocabulary.
    assert "Hemoglobin [Mass/volume] in Blood" in set(mappings.concept_name)


def test_another_coder_identity_misses(tmp_path):
    cache = CodingCache(tmp_path / "cache.db")
    CachedCoder(IdentifiedCoder(model="m1"), cache).code(["Hb"], "Measurement")
    other = IdentifiedCoder(model="m2")
    CachedCoder(other, cache).code(["Hb"], "Measurement")
    assert other.calls == [(["Hb"], "Measurement", None)]
    assert len(cache) == 2


def test_domain_and_vocabularies_are_part_of_the_key(tmp_path):
    coder = CachedCoder(IdentifiedCoder(), tmp_path / "cache.db")
    coder.code(["Hb"], "Measurement", ["LOINC"])
    coder.code(["Hb"], "Measurement", ["LOINC"])
    coder.code(["Hb"], "Observation", ["LOINC"])
    coder.code(["Hb"], "Measurement", ["SNOMED"])
    assert coder.stats.hits == 1
    assert coder.stats.misses == 3


def test_a_coder_without_identity_needs_one(tmp_path):
    with pytest.raises(TypeError, match="cache_identity"):
        CachedCoder(FakeCoder(), tmp_path / "cache.db")
    coder = CachedCoder(FakeCoder(), tmp_path / "cache.db", identity={"v": 1})
    assert coder.code(["Hb"])[0].concept_id == 3000963


def test_already_cached_mapper_with_cache_is_an_error(tmp_path):
    cached = CachedCoder(IdentifiedCoder(), tmp_path / "cache.db")
    with pytest.raises(ValueError, match="already a CachedCoder"):
        cached_builder(cached, tmp_path / "other.db")
    # Passed on its own, it is used as is.
    assert cached_builder(cached, None).cache is cached.cache


# --- context ----------------------------------------------------------------------


def test_context_is_part_of_the_key(tmp_path):
    cache = CodingCache(tmp_path / "cache.db")
    hb_a = CodingContext(text="Hb", related={"lab_unit": "g/dL"}, sentence="Hb 11 g/dL")
    hb_b = CodingContext(text="Hb", related={"lab_unit": "g/dL"}, sentence="Hb 12 g/dL")

    first = ContextCoder()
    CachedCoder(first, cache).code(["Hb"], contexts=[hb_a])
    second = ContextCoder()
    coder = CachedCoder(second, cache)
    results = coder.code(["Hb", "Hb"], contexts=[hb_a, hb_b])

    # The same text in another sentence is coded again; the identical context is not.
    assert second.calls == [[hb_b]]
    assert [r.cached for r in results] == [True, False]
    assert (coder.stats.hits, coder.stats.misses) == (1, 1)


def test_builder_passes_contexts_through_the_cache(tmp_path):
    coder = ContextCoder()
    run(cached_builder(coder, tmp_path / "cache.db"), NOTE, hb_entities())
    (contexts,) = coder.calls
    assert {c.text for c in contexts} == {"Hb", "glucose"}
    hb = next(c for c in contexts if c.text == "Hb")
    assert hb.related["lab_unit"] == "g/dL"
    assert hb.sentence == "Hb 11,2 g/dL, glucose 99 MG/DL"


# --- what is stored ---------------------------------------------------------------


def test_errors_are_not_stored(tmp_path):
    cache = CodingCache(tmp_path / "cache.db")
    coder = CachedCoder(ContextCoder(error_for={"Hb"}), cache)
    coder.code(
        ["Hb", "glucose"],
        contexts=[CodingContext(text="Hb"), CodingContext(text="glucose")],
    )
    assert len(cache) == 1
    assert coder.stats.stored == 1


def test_each_chunk_is_stored_as_it_is_coded(tmp_path):
    class FailsSecondChunk(IdentifiedCoder):
        def code(self, texts, domain=None, vocabularies=None):
            if self.calls:
                raise RuntimeError("API down")
            return super().code(texts, domain, vocabularies)

    cache = CodingCache(tmp_path / "cache.db")
    coder = CachedCoder(FailsSecondChunk(), cache, chunk_size=2)
    with pytest.raises(RuntimeError):
        coder.code(["Hb", "glucose", "BMI"])
    assert len(cache) == 2  # the first chunk survived


def test_builder_codes_frequent_texts_first():
    coder = FakeCoder()
    builder = make_builder(coder)
    run(
        builder,
        NOTE,
        [lab("e1", "glucose"), lab("e2", "Hb"), lab("e3", "Hb", line="L2")],
    )
    assert coder.calls[0][0] == ["Hb", "glucose"]
    coder.calls.clear()
    # Alphabetically "Hb" sorts first; here "glucose" is more frequent.
    run(
        builder,
        "Hb 11 g/dL, glucose 99\nglucose 101",
        [lab("e1", "Hb"), lab("e2", "glucose"), lab("e3", "glucose", line="L2")],
    )
    assert coder.calls[0][0] == ["glucose", "Hb"]


def test_duplicates_in_one_call_are_coded_once(tmp_path):
    inner = IdentifiedCoder()
    results = CachedCoder(inner, tmp_path / "cache.db").code(["Hb", "Hb", "glucose"])
    assert inner.calls == [(["Hb", "glucose"], None, None)]
    assert [r.text for r in results] == ["Hb", "Hb", "glucose"]


# --- limits, clearing, info -------------------------------------------------------


def coded(text):
    return CodedText(text=text, concept_id=1)


def test_max_entries_evicts_the_least_recently_used(tmp_path):
    cache = CodingCache(tmp_path / "cache.db", max_entries=2)
    cache.put_many("c", [("a", coded("a")), ("b", coded("b"))])
    cache.get_many(["a"])  # "a" is now more recent than "b"
    cache.put_many("c", [("c", coded("c"))])
    assert set(cache.get_many(["a", "b", "c"])) == {"a", "c"}


def test_max_bytes_evicts_until_under_the_limit(tmp_path):
    size = len(coded("a").model_dump_json(exclude={"cached"}).encode()) + 1
    cache = CodingCache(tmp_path / "cache.db", max_bytes=size * 2)
    cache.put_many("c", [("a", coded("a")), ("b", coded("b"))])
    cache.put_many("c", [("c", coded("c"))])
    assert set(cache.get_many(["a", "b", "c"])) == {"b", "c"}
    assert cache.info().bytes <= size * 2


@pytest.mark.parametrize("limits", [{"max_entries": 0}, {"max_bytes": -1}])
def test_limits_must_be_positive(tmp_path, limits):
    with pytest.raises(ValueError):
        CodingCache(tmp_path / "cache.db", **limits)


def test_clear_all_or_one_coder(tmp_path):
    cache = CodingCache(tmp_path / "cache.db")
    one = CachedCoder(IdentifiedCoder(model="m1"), cache)
    two = CachedCoder(IdentifiedCoder(model="m2"), cache)
    one.code(["Hb", "glucose"])
    two.code(["Hb"])

    info = cache.info()
    assert info.entries == 3
    assert [(c.coder, c.entries) for c in info.coders] == [
        (one.coder_hash, 2),
        (two.coder_hash, 1),
    ]
    assert info.coders[0].identity["identity"] == {
        "coder": "IdentifiedCoder",
        "model": "m1",
    }

    assert cache.clear(one.coder_hash) == 2
    assert len(cache) == 1
    assert cache.clear() == 1
    assert len(cache) == 0


def test_cli_info_and_clear(tmp_path):
    from aatm.main import app

    path = tmp_path / "cache.db"
    CachedCoder(IdentifiedCoder(), path).code(["Hb", "glucose"])
    runner = CliRunner()

    result = runner.invoke(app, ["cache", "info", "--path", str(path)])
    assert result.exit_code == 0, result.output
    assert "2 entries" in result.output

    result = runner.invoke(app, ["cache", "clear", "--path", str(path)], input="n\n")
    assert len(CodingCache(path)) == 2

    result = runner.invoke(app, ["cache", "clear", "--path", str(path), "--yes"])
    assert result.exit_code == 0, result.output
    assert "Deleted 2 entries" in result.output
    assert len(CodingCache(path)) == 0


# --- identities of the built-in coders --------------------------------------------


def agentic(vocab_db, **kwargs):  # noqa: F811
    return AgenticCoder(
        ScriptedLLM([]), retriever=FakeRetriever(), vocabulary=vocab_db, **kwargs
    )


def test_agentic_coder_identity(tmp_path, vocab_db):  # noqa: F811
    identity = agentic(vocab_db).cache_identity()
    agent = identity["agent"]
    assert agent["llm"]["model_id"] == "fake/scripted"
    assert agent["max_tool_calls"] == 8
    assert agent["tools"]["vocabulary"]["path"].endswith(".db")
    assert len(agent["prompts"]) == 16

    path = tmp_path / "cache.db"
    base = CachedCoder(agentic(vocab_db), path).coder_hash
    # Concurrency does not change results; the tool budget does.
    assert CachedCoder(agentic(vocab_db, max_concurrency=2), path).coder_hash == base
    assert CachedCoder(agentic(vocab_db, max_tool_calls=3), path).coder_hash != base


def test_agentic_coder_errors_are_not_cached(tmp_path, vocab_db):  # noqa: F811
    cache = CodingCache(tmp_path / "cache.db")
    coder = CachedCoder(agentic(vocab_db), cache)
    (result,) = coder.code(["creatinine"], "Measurement")
    assert result.error
    assert len(cache) == 0


def test_agentic_coder_results_are_cached(tmp_path, vocab_db):  # noqa: F811
    steps = [step("finish", concept_id=3007760, confidence="high")]
    coder = AgenticCoder(
        ScriptedLLM(steps), retriever=FakeRetriever(), vocabulary=vocab_db
    )
    cache = CodingCache(tmp_path / "cache.db")
    (first,) = CachedCoder(coder, cache).code(["creatinine"], "Measurement")
    assert first.concept_id == 3007760 and not first.cached

    # The script is used up: a second search would fail.
    (again,) = CachedCoder(coder, cache).code(["creatinine"], "Measurement")
    assert again.cached
    assert again.model_dump(exclude={"cached"}) == first.model_dump(exclude={"cached"})


# --- changes to the data a coder reads --------------------------------------------


@pytest.fixture
def warnings(monkeypatch):
    import aatm.omop.cache as cache_module

    messages: list[str] = []
    monkeypatch.setattr(cache_module.logger, "warning", messages.append)
    return messages


class DataCoder(IdentifiedCoder):
    """A coder whose retriever's content is fingerprinted (or not)."""

    def __init__(self, content=None):
        super().__init__()
        self.content = content

    def cache_identity(self):
        retriever = {"class": "Index"}
        if self.content is not None:
            retriever["content"] = self.content
        return {"coder": "DataCoder", "retriever": retriever}


def test_changed_data_is_not_reused_and_is_reported(tmp_path, warnings):
    cache = CodingCache(tmp_path / "cache.db")
    old = CachedCoder(DataCoder(content={"count": 10}), cache)
    old.code(["Hb", "glucose"])
    assert warnings == []

    rebuilt = DataCoder(content={"count": 11})
    new = CachedCoder(rebuilt, cache)
    new.code(["Hb"])
    assert rebuilt.calls == [(["Hb"], None, None)]  # not reused
    (message,) = warnings
    assert "changed since 2 results were cached" in message
    assert f"aatm cache clear --coder {old.coder_hash}" in message


def test_unchecked_data_is_reported(tmp_path, warnings):
    coder = CachedCoder(DataCoder(), tmp_path / "cache.db")
    assert coder.unchecked == ["retriever (Index)"]
    (message,) = warnings
    assert "cannot be detected" in message and "aatm cache clear" in message


def make_vocabulary(path, loinc="LOINC 2.78"):
    import sqlite3

    con = sqlite3.connect(path)
    con.execute("CREATE TABLE vocabulary (vocabulary_id TEXT, vocabulary_version TEXT)")
    con.executemany(
        "INSERT INTO vocabulary VALUES (?, ?)",
        [("LOINC", loinc), ("SNOMED", "2024-08-01")],
    )
    con.commit()
    con.close()


def test_vocabulary_fingerprint_follows_the_releases(tmp_path):
    from aatm.omop.cache import vocabulary_fingerprint

    make_vocabulary(tmp_path / "a.db")
    make_vocabulary(tmp_path / "b.db")
    make_vocabulary(tmp_path / "c.db", loinc="LOINC 2.80")
    a, b, c = (vocabulary_fingerprint(tmp_path / f"{n}.db") for n in "abc")
    assert a == b  # same releases, another file
    assert a != c
    assert vocabulary_fingerprint(tmp_path / "missing.db") is None


def test_chroma_collection_fingerprint():
    from aatm.omop.cache import data_fingerprint

    class Collection:
        id = "uuid-1"

        def count(self):
            return 42

    class Retriever:
        collection = Collection()

    assert data_fingerprint(Retriever()) == {"collection_id": "uuid-1", "count": 42}


def test_agentic_coder_identity_follows_the_vocabulary(tmp_path, vocab_db):  # noqa: F811
    import sqlite3

    con = sqlite3.connect(vocab_db)
    con.execute("CREATE TABLE vocabulary (vocabulary_id TEXT, vocabulary_version TEXT)")
    con.execute("INSERT INTO vocabulary VALUES ('LOINC', 'LOINC 2.78')")
    con.commit()
    path = tmp_path / "cache.db"
    before = CachedCoder(agentic(vocab_db), path)
    assert "content" in before.identity["identity"]["agent"]["tools"]["vocabulary"]
    # The fake retriever has no collection: it is reported as unchecked.
    assert before.unchecked == [
        "agent.tools.retriever (test_concept_search.FakeRetriever)"
    ]

    con.execute("UPDATE vocabulary SET vocabulary_version = 'LOINC 2.80'")
    con.commit()
    con.close()
    assert CachedCoder(agentic(vocab_db), path).coder_hash != before.coder_hash


@pytest.mark.parametrize("as_dict", [False, True])
def test_timeout_is_not_part_of_the_identity(as_dict):
    from aatm.llm import LLMSettings
    from aatm.omop.cache import describe_component

    class LLM:
        model_id = "openai/m"

        def __init__(self, **settings):
            self.settings = settings if as_dict else LLMSettings(**settings)

    fast = describe_component(LLM(timeout=5, temperature=0))
    slow = describe_component(LLM(timeout=60, temperature=0))
    assert fast == slow
    assert "timeout" not in fast["settings"]
    assert describe_component(LLM(timeout=5, temperature=1)) != fast

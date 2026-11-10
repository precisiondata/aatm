"""Batch extraction in chunked runs (`OmopExtractionPipeline.run(batch=True)`)."""

import asyncio

import pandas as pd
import pytest

from aatm.data_models.extractors import EntityValidationResult
from aatm.omop import OmopExtractionPipeline, OmopRun

from test_omop_mapping import FakeLookup, ecog_definitions
from test_omop_runner import FakeExtractor, make_pipeline, notes, run


class FakeBatchExtractor(FakeExtractor):
    """`FakeExtractor` that can also run batch jobs, held until `ready`."""

    model_id = "fake/model"
    supports_batch = True

    def __init__(self, definitions, fail=(), lose=(), ready=True):
        super().__init__(definitions)
        self.batch_fail = set(fail)  # texts whose request fails within the job
        self.lose = set(lose)  # keys the job returns nothing for
        self.ready = ready
        self.jobs: dict[str, list[tuple[str, str]]] = {}
        self.polls = 0

    async def submit_batch(self, items):
        batch_id = f"batch-{len(self.jobs) + 1}"
        self.jobs[batch_id] = list(items)
        return batch_id

    async def poll_batch(self, batch_id, items):
        assert list(items) == self.jobs[batch_id]
        self.polls += 1
        if not self.ready:
            return None
        results = {}
        for key, text in items:
            if key in self.lose:
                continue
            if text in self.batch_fail:
                results[key] = EntityValidationResult(error="bad request")
                continue
            (results[key],) = await FakeExtractor.__call__(self, [text])
        return results

    @property
    def submitted_notes(self):
        return sum(len(items) for items in self.jobs.values())


def make_batch_pipeline(**kwargs):
    definitions = ecog_definitions()
    extractor = FakeBatchExtractor(definitions, **kwargs)
    pipeline = OmopExtractionPipeline(
        definitions, extractor=extractor, vocabulary=FakeLookup()
    )
    return pipeline, extractor


def sorted_table(out, name="measurement"):
    frame = OmopRun(out).table(name)
    return frame.sort_values(list(frame.columns)).reset_index(drop=True)


def test_batch_run_matches_live_run(tmp_path):
    pipeline, extractor = make_batch_pipeline()
    summary = run(
        pipeline, notes(5), tmp_path / "batch", chunk_size=2, batch=True, poll_every=0
    )
    assert summary.completed
    assert summary.built == summary.extracted == 5
    assert summary.batches_submitted == summary.batches_collected == 3
    assert extractor.submitted_notes == 5
    assert summary.stages["extraction"].items == 5
    assert "batch jobs: 3 submitted, 3 collected" in str(summary)

    live, _ = make_pipeline()
    run(live, notes(5), tmp_path / "live", chunk_size=2)
    pd.testing.assert_frame_equal(
        sorted_table(tmp_path / "batch"), sorted_table(tmp_path / "live")
    )

    output = OmopRun(tmp_path / "batch")
    assert output.status() == {"built": 5}
    batches = output.batches()
    assert list(batches["status"]) == ["done"] * 3
    assert list(batches["notes"]) == [2, 2, 1]
    assert set(batches["model_id"]) == {"fake/model"}

    # Nothing left to do: every note is skipped, no job submitted.
    again = run(pipeline, notes(5), tmp_path / "batch", batch=True, poll_every=0)
    assert again.skipped == 5 and again.batches_submitted == 0


def stop_while_pending(pipeline, source, out, **kwargs):
    async def go():
        await asyncio.wait_for(
            pipeline.run(source, out, batch=True, poll_every=0.01, **kwargs), 0.3
        )

    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        asyncio.run(go())


@pytest.mark.parametrize("batch", [True, False])
def test_restart_collects_pending_jobs_instead_of_resubmitting(tmp_path, batch):
    pipeline, extractor = make_batch_pipeline(ready=False)
    stop_while_pending(pipeline, notes(5), tmp_path, chunk_size=2)
    output = OmopRun(tmp_path)
    assert output.status() == {"submitted": 5}
    assert list(output.batches()["status"]) == ["pending"] * 3
    assert extractor.polls > 0

    # The jobs finish while no run is going on; the next run, in batch mode or
    # live, collects them and submits nothing.
    extractor.ready = True
    summary = run(pipeline, notes(5), tmp_path, chunk_size=2, batch=batch)
    assert len(extractor.jobs) == 3
    assert extractor.calls == 5  # one per note, by the jobs
    assert summary.batches_submitted == 0
    assert summary.batches_collected == 3
    assert summary.built == summary.extracted == 5
    assert summary.skipped == summary.duplicates == summary.failed == 0
    assert output.status() == {"built": 5}
    assert list(output.batches()["status"]) == ["done"] * 3


def test_restart_without_the_source_still_collects(tmp_path):
    pipeline, extractor = make_batch_pipeline(ready=False)
    stop_while_pending(pipeline, notes(3), tmp_path, chunk_size=3)
    extractor.ready = True
    summary = run(pipeline, [], tmp_path, batch=True, poll_every=0)
    assert summary.built == 3 and summary.batches_collected == 1
    assert len(extractor.jobs) == 1


def test_failed_requests_go_back_to_the_ledger(tmp_path):
    source = notes(5)
    failing = source[1]["text"]
    pipeline, extractor = make_batch_pipeline(fail=[failing], lose=["note-4"])
    summary = run(pipeline, source, tmp_path, chunk_size=5, batch=True, poll_every=0)
    assert summary.built == summary.extracted == 3
    assert summary.failed == 2
    assert summary.stages["extraction"].errors == 2
    failures = OmopRun(tmp_path).failures()
    assert list(failures["note_id"]) == [2, 4]
    assert set(failures["stage"]) == {"extract"}
    assert "bad request" in failures["error"][0]
    assert "no result" in failures["error"][1]
    assert list(failures["attempts"]) == [1, 1]

    # A later live run extracts them again.
    extractor.batch_fail.clear()
    calls = extractor.calls
    summary = run(pipeline, source, tmp_path, chunk_size=5)
    assert extractor.calls - calls == 2
    assert summary.built == 2 and summary.skipped == 3
    assert OmopRun(tmp_path).status() == {"built": 5}


def test_max_batches_waits_for_a_job_before_submitting(tmp_path):
    pipeline, extractor = make_batch_pipeline()
    summary = run(
        pipeline,
        notes(4),
        tmp_path,
        chunk_size=1,
        batch=True,
        poll_every=0,
        max_batches=1,
    )
    assert summary.built == 4 and summary.batches_submitted == 4


def test_repeated_note_in_a_pending_job_is_a_duplicate(tmp_path):
    pipeline, extractor = make_batch_pipeline()
    source = notes(2) + notes(1)  # note 1 again, in a later chunk
    summary = run(pipeline, source, tmp_path, chunk_size=2, batch=True, poll_every=0)
    assert extractor.submitted_notes == 2
    assert summary.built == 2
    assert summary.duplicates + summary.skipped == 1


def test_batch_needs_an_extractor_that_supports_it(tmp_path):
    pipeline, _ = make_pipeline()
    with pytest.raises(ValueError, match="batch=True needs"):
        run(pipeline, notes(1), tmp_path, batch=True)

    pipeline, _ = make_batch_pipeline()
    for kwargs in ({"max_batches": 0}, {"poll_every": -1}):
        with pytest.raises(ValueError):
            run(pipeline, notes(1), tmp_path, batch=True, **kwargs)


def test_pending_jobs_need_an_extractor_that_can_poll(tmp_path):
    pipeline, extractor = make_batch_pipeline(ready=False)
    stop_while_pending(pipeline, notes(2), tmp_path)
    # Another extractor class is refused by the extraction fingerprint already; the
    # same class without polling by the batch check.
    live, _ = make_pipeline()
    with pytest.raises(ValueError, match="other entity definitions or another"):
        run(live, notes(2), tmp_path)
    extractor.poll_batch = None
    with pytest.raises(ValueError, match="cannot poll batch jobs"):
        run(pipeline, notes(2), tmp_path)


def test_batch_usage_is_counted_in_extraction(tmp_path):
    from aatm.llm.usage import record_usage

    class Recording(FakeBatchExtractor):
        async def poll_batch(self, batch_id, items):
            for _ in items:
                record_usage(100, 10, model_id="fake/model@batch")
            return await super().poll_batch(batch_id, items)

    definitions = ecog_definitions()
    pipeline = OmopExtractionPipeline(
        definitions, extractor=Recording(definitions), vocabulary=FakeLookup()
    )
    summary = run(
        pipeline,
        notes(3),
        tmp_path,
        batch=True,
        poll_every=0,
        prices={"fake/model": (1.0, 10.0)},
    )
    usage = summary.stages["extraction"].usage
    assert usage.by_model["fake/model@batch"].calls == 3
    # Half of 300 input tokens at 1.0 and 30 output tokens at 10.0, per million.
    assert summary.stages["extraction"].cost == pytest.approx(0.5 * 600 / 1e6)

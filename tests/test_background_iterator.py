import threading

import pytest

from aatm.background_iterator import BackgroundIterator


def test_yields_all_items_in_order():
    with BackgroundIterator(iter(range(10)), maxsize=2) as items:
        assert list(items) == list(range(10))


def test_producer_runs_while_consumer_is_busy():
    second_item_produced = threading.Event()

    def produce():
        yield 1
        yield 2
        second_item_produced.set()
        yield 3

    with BackgroundIterator(produce(), maxsize=2) as items:
        iterator = iter(items)
        assert next(iterator) == 1
        # While the consumer "works" on item 1, the producer moves ahead.
        assert second_item_produced.wait(timeout=5)
        assert list(iterator) == [2, 3]


def test_producer_stays_at_most_maxsize_items_ahead():
    produced = []
    blocked = threading.Event()

    def produce():
        for i in range(100):
            produced.append(i)
            yield i
        blocked.set()

    with BackgroundIterator(produce(), maxsize=2, poll_interval=0.01) as items:
        iterator = iter(items)
        next(iterator)
        assert not blocked.wait(timeout=0.3)
        # One item consumed, two queued, one held by the blocked producer.
        assert len(produced) <= 4


def test_producer_exception_is_raised_in_consumer():
    def produce():
        yield 1
        raise ValueError("bad row")

    with BackgroundIterator(produce()) as items:
        iterator = iter(items)
        assert next(iterator) == 1
        with pytest.raises(ValueError, match="bad row"):
            next(iterator)


def test_leaving_early_stops_the_producer_and_runs_its_cleanup():
    cleaned_up = threading.Event()

    def produce():
        try:
            i = 0
            while True:
                yield i
                i += 1
        finally:
            cleaned_up.set()

    background = BackgroundIterator(produce(), maxsize=2, poll_interval=0.01)
    with pytest.raises(RuntimeError, match="consumer failed"):
        with background as items:
            for item in items:
                if item == 3:
                    raise RuntimeError("consumer failed")

    assert cleaned_up.wait(timeout=5)
    assert not background._thread.is_alive()


def test_rejects_non_positive_maxsize():
    with pytest.raises(ValueError):
        BackgroundIterator([], maxsize=0)

import pytest
import torch

from aatm import gpu_memory
from aatm.gpu_memory import (
    GpuMemoryError,
    GpuMemoryLimit,
    calibrate_batch_size,
    encode_with_oom_backoff,
)

GIB = 1024**3


class FakeGpu:
    """Simulated GPU: memory grows with batch size times the longest text.

    Speed improves with batch size up to ``fastest_batch`` and then degrades,
    like the measurements on an RTX 4090.
    """

    def __init__(self, limit_gib=10.0, gib_per_item_token=0.001, fastest_batch=64):
        self.limit = limit_gib * GIB
        self.gib_per_item_token = gib_per_item_token
        self.fastest_batch = fastest_batch
        self.peak = 0
        self.clock = 0.0
        self.calls = []

    # CudaMemoryProbe interface
    def reset_peak(self):
        self.peak = 0

    def peak_bytes(self):
        return self.peak

    def synchronize(self):
        pass

    def timer(self):
        return self.clock

    def encode(self, texts, batch_size):
        self.calls.append((len(texts), batch_size))
        longest = max(len(text) for text in texts)
        needed = 1 * GIB + batch_size * longest * self.gib_per_item_token * GIB
        if needed > self.limit:
            raise torch.OutOfMemoryError("simulated")
        self.peak = max(self.peak, needed)
        per_text = 1.0 / min(batch_size, self.fastest_batch) + batch_size / 1e4
        self.clock += len(texts) * per_text


@pytest.fixture(autouse=True)
def no_real_gpu_cleanup(monkeypatch):
    monkeypatch.setattr(gpu_memory, "free_cached_gpu_memory", lambda: None)


def calibrate(gpu, longest, typical, **kwargs):
    return calibrate_batch_size(
        gpu.encode,
        longest,
        typical,
        limit_bytes=gpu.limit,
        probe=gpu,
        timer=gpu.timer,
        **kwargs,
    )


def test_picks_fastest_batch_size_when_memory_allows():
    gpu = FakeGpu(limit_gib=100, fastest_batch=64)
    assert calibrate(gpu, ["x" * 20] * 1024, ["x" * 5] * 256) == 64


def test_worst_case_memory_limits_the_batch_size():
    # 1 GiB base + size * 100 tokens * 0.001 GiB: size 64 needs 7.4 GiB, over
    # 85% of 8 GiB, so 32 (4.2 GiB) is the largest that fits.
    gpu = FakeGpu(limit_gib=8, fastest_batch=1024)
    assert calibrate(gpu, ["x" * 100] * 1024, ["x" * 5] * 256) == 32


def test_out_of_memory_during_calibration_stops_the_search():
    gpu = FakeGpu(limit_gib=8, fastest_batch=1024)
    # With margin 1.0 the budget check never stops the search: size 128 needs
    # 7.4 GiB and fits, size 256 needs 13.8 GiB and hits the out-of-memory error.
    size = calibrate(gpu, ["x" * 50] * 1024, ["x" * 5] * 256, memory_margin=1.0)
    assert size == 128
    assert (256, 256) in gpu.calls


def test_memory_is_measured_on_the_longest_texts():
    gpu = FakeGpu(limit_gib=100)
    calibrate(gpu, ["x" * 20] * 1024, ["x" * 5] * 256)
    worst_case_calls = [c for c in gpu.calls if c[0] == c[1]]
    assert worst_case_calls  # one batch of exactly `size` longest texts


def test_few_texts_are_repeated_to_test_the_worst_case():
    gpu = FakeGpu(limit_gib=100, fastest_batch=64)
    assert calibrate(gpu, ["x" * 20] * 3, ["x" * 5] * 256) == 64
    assert (64, 64) in gpu.calls


def test_model_that_never_fits_raises():
    gpu = FakeGpu(limit_gib=1.5)
    with pytest.raises(GpuMemoryError, match="half precision"):
        calibrate(gpu, ["x" * 500] * 1024, ["x" * 5] * 256)


def test_no_data_returns_smallest_candidate():
    gpu = FakeGpu()
    assert calibrate(gpu, [], []) == 8


class FlakyEmbeddingFunction:
    def __init__(self, max_working_batch):
        self.batch_size = 64
        self.max_working_batch = max_working_batch
        self.attempts = []

    def __call__(self, texts):
        self.attempts.append(self.batch_size)
        if self.batch_size > self.max_working_batch:
            raise torch.OutOfMemoryError("simulated")
        return [[0.0]] * len(texts)


def test_oom_backoff_halves_batch_size_and_keeps_it():
    function = FlakyEmbeddingFunction(max_working_batch=16)

    assert len(encode_with_oom_backoff(function, ["a", "b"])) == 2
    assert function.attempts == [64, 32, 16]
    assert function.batch_size == 16


def test_oom_backoff_gives_up_at_batch_size_one():
    function = FlakyEmbeddingFunction(max_working_batch=0)
    with pytest.raises(GpuMemoryError, match="batch size 1"):
        encode_with_oom_backoff(function, ["a"])
    assert function.attempts == [64, 32, 16, 8, 4, 2, 1]


def test_memory_limit_is_inert_when_disabled():
    with GpuMemoryLimit(enabled=False) as limit:
        assert limit.limit_bytes == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_memory_limit_caps_and_restores_allocator():
    get_fraction = getattr(torch.cuda, "get_per_process_memory_fraction", None)
    if get_fraction is None:
        pytest.skip("this PyTorch cannot report the memory fraction")
    before = get_fraction()

    with GpuMemoryLimit() as limit:
        _, total = torch.cuda.mem_get_info()
        assert 0 < limit.limit_bytes < total
        assert get_fraction() == pytest.approx(limit.limit_bytes / total)
        with pytest.raises(torch.OutOfMemoryError):
            torch.empty(limit.limit_bytes + GIB, dtype=torch.uint8, device="cuda")

    assert get_fraction() == before

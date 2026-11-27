import threading
from typing import Any

import numpy as np
import pytest

from aatm import embedding_functions as ef
from aatm import precision


class FakeTokenizer:
    """One token per word."""

    def __call__(self, texts: Any, add_special_tokens: bool = True) -> dict:
        if isinstance(texts, str):
            return {"input_ids": texts.split()}
        return {"input_ids": [text.split() for text in texts]}


class FakeSentenceTransformer:
    """Stand-in for SentenceTransformer that records how it was used.

    Each embedding is ``[word count, 0, 0]``, so tests can tell which text an
    embedding belongs to.
    """

    instances: list["FakeSentenceTransformer"] = []

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        self.model_id = model_id
        self.init_kwargs = kwargs
        self.encode_calls: list[dict[str, Any]] = []
        self.encoded_texts: list[list[str]] = []
        self.tokenizer = FakeTokenizer()
        self.prompts = {"query": "Instruct: find"}
        self.default_prompt_name = None
        self.max_seq_length = 100
        FakeSentenceTransformer.instances.append(self)

    def eval(self) -> "FakeSentenceTransformer":
        return self

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        self.encode_calls.append(kwargs)
        self.encoded_texts.append(list(texts))
        embeddings = np.zeros((len(texts), 3), dtype=np.float32)
        embeddings[:, 0] = [len(text.split()) for text in texts]
        return embeddings


@pytest.fixture
def fake_st(monkeypatch):
    FakeSentenceTransformer.instances = []
    monkeypatch.setattr(ef, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(ef, "get_device", lambda: "cpu")
    monkeypatch.delenv("AATM_EMBEDDING_DTYPE", raising=False)
    # Checkpoints are "stored" in bfloat16 unless a test says otherwise.
    monkeypatch.setattr(precision, "checkpoint_dtype", lambda model_id: "bfloat16")
    monkeypatch.setattr(
        precision.torch.cuda,
        "is_bf16_supported",
        lambda including_emulation=True: True,
    )
    return FakeSentenceTransformer


@pytest.mark.parametrize("cls", [ef.Qwen3EmbeddingFunction, ef.GemmaEmbeddingFunction])
def test_local_models_are_marked_local_before_instantiation(cls):
    # The vector database builder checks this on the class, before loading weights.
    assert cls.is_local is True


@pytest.mark.parametrize(
    ("cls", "model_id"),
    [
        (ef.Qwen3EmbeddingFunction, ef.Qwen3EmbeddingModels.QWEN3_06B.value),
        (ef.GemmaEmbeddingFunction, ef.GemmaEmbeddingModels.EMBEDDING_GEMMA_300M.value),
    ],
)
def test_local_models_use_configured_batch_size(fake_st, cls, model_id):
    function = cls(model=model_id, batch_size=7)
    function.batch_size = 2  # the builder may overwrite it

    embeddings = function(["a", "b", "c", "d", "e"])

    assert len(embeddings) == 5
    sizes = [call["batch_size"] for call in fake_st.instances[0].encode_calls]
    assert sizes == [2, 2, 1]


def test_local_models_batch_by_token_length_and_keep_input_order(fake_st):
    function = ef.GemmaEmbeddingFunction(
        model=ef.GemmaEmbeddingModels.EMBEDDING_GEMMA_300M.value, batch_size=2
    )
    texts = ["a", "a b c d", "a b", "a b c", "a b c d e"]

    embeddings = function(texts)

    # Longest first, so texts of similar length share a batch.
    assert fake_st.instances[0].encoded_texts == [
        ["a b c d e", "a b c d"],
        ["a b c", "a b"],
        ["a"],
    ]
    assert [int(vector[0]) for vector in embeddings] == [1, 4, 2, 3, 5]


def test_local_models_fill_batches_up_to_token_budget(fake_st):
    function = ef.GemmaEmbeddingFunction(
        model=ef.GemmaEmbeddingModels.EMBEDDING_GEMMA_300M.value,
        max_batch_tokens=8,
    )
    texts = ["a b c d"] * 3 + ["a"] * 10

    function(texts)

    # A batch starting with a 4-token text holds two; one of 1-token texts, eight.
    sizes = [call["batch_size"] for call in fake_st.instances[0].encode_calls]
    assert sizes == [2, 2, 8, 1]


def test_token_lengths_include_prompt_and_respect_max_length(fake_st):
    function = ef.Qwen3EmbeddingFunction(model=ef.Qwen3EmbeddingModels.QWEN3_06B.value)
    fake_st.instances[0].max_seq_length = 5

    # The "query" prompt adds two tokens.
    assert function.token_lengths(["a", "a b", "a b c d e f"]) == [3, 4, 5]


@pytest.mark.parametrize(
    ("lengths", "batch_size", "max_batch_tokens", "expected"),
    [
        ([1, 3, 2], 2, None, [[1, 2], [0]]),
        ([1, 3, 2], 5, None, [[1, 2, 0]]),
        # A text longer than the budget still gets a batch of its own.
        ([10, 2, 2, 2], 99, 4, [[0], [1, 2], [3]]),
        ([], 4, None, []),
    ],
)
def test_plan_batches(lengths, batch_size, max_batch_tokens, expected):
    assert ef.plan_batches(lengths, batch_size, max_batch_tokens) == expected


def test_qwen3_encodes_with_query_prompt(fake_st):
    function = ef.Qwen3EmbeddingFunction(model=ef.Qwen3EmbeddingModels.QWEN3_06B.value)
    function(["a"])
    assert fake_st.instances[0].encode_calls[0]["prompt_name"] == "query"


def test_gemma_encodes_without_prompt(fake_st):
    function = ef.GemmaEmbeddingFunction(
        model=ef.GemmaEmbeddingModels.EMBEDDING_GEMMA_300M.value
    )
    function(["a"])
    assert "prompt_name" not in fake_st.instances[0].encode_calls[0]


@pytest.mark.parametrize(
    ("cls", "model_id"),
    [
        (ef.Qwen3EmbeddingFunction, ef.Qwen3EmbeddingModels.QWEN3_06B.value),
        (ef.GemmaEmbeddingFunction, ef.GemmaEmbeddingModels.EMBEDDING_GEMMA_300M.value),
    ],
)
def test_local_models_load_without_key_value_cache(fake_st, cls, model_id):
    # The cache only helps generation; for embeddings it wastes GPU memory.
    cls(model=model_id)
    assert fake_st.instances[0].init_kwargs["config_kwargs"] == {"use_cache": False}


def test_unknown_model_id_is_rejected(fake_st):
    with pytest.raises(ValueError):
        ef.Qwen3EmbeddingFunction(model="not-a-model")


def test_models_load_in_checkpoint_precision_by_default(fake_st, monkeypatch):
    monkeypatch.setattr(ef, "get_device", lambda: "cuda")
    function = ef.Qwen3EmbeddingFunction(model=ef.Qwen3EmbeddingModels.QWEN3_06B.value)
    assert function.dtype == "bfloat16"
    assert (
        fake_st.instances[0].init_kwargs["model_kwargs"]["dtype"] is ef.torch.bfloat16
    )


def test_half_precision_checkpoint_loads_in_float32_on_cpu(fake_st):
    function = ef.Qwen3EmbeddingFunction(model=ef.Qwen3EmbeddingModels.QWEN3_06B.value)
    assert function.dtype == "float32"
    assert fake_st.instances[0].init_kwargs["model_kwargs"]["dtype"] is ef.torch.float32


def test_models_load_in_requested_half_precision(fake_st, monkeypatch):
    monkeypatch.setattr(ef, "get_device", lambda: "cuda")
    function = ef.Qwen3EmbeddingFunction(
        model=ef.Qwen3EmbeddingModels.QWEN3_06B.value, dtype="float16"
    )
    assert function.dtype == "float16"
    assert fake_st.instances[0].init_kwargs["model_kwargs"]["dtype"] is ef.torch.float16


def test_gemma_rejects_float16(fake_st, monkeypatch):
    monkeypatch.setattr(ef, "get_device", lambda: "cuda")
    with pytest.raises(ef.EmbeddingPrecisionError, match="does not support float16"):
        ef.GemmaEmbeddingFunction(
            model=ef.GemmaEmbeddingModels.EMBEDDING_GEMMA_300M.value, dtype="float16"
        )
    assert fake_st.instances == []  # rejected before loading weights


# --- API models ----------------------------------------------------------------------


class StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class FakeApiModel(ef.ApiEmbeddingFunction):
    """API model stand-in: each embedding is ``[number in the text]``."""

    max_batch_size = 3

    def __init__(self, concurrency: int = 1, failures: list[Exception] | None = None):
        self.concurrency = concurrency
        self.limiter_id = "fake/embedding"
        self.requests: list[list[str]] = []
        self.failures = list(failures or [])
        self._lock = threading.Lock()

    def _embed(self, texts: list[str]) -> list[np.ndarray]:
        with self._lock:
            self.requests.append(texts)
            if self.failures:
                raise self.failures.pop(0)
        return [np.array([float(text)], dtype=np.float32) for text in texts]


@pytest.fixture
def fast_retries():
    from aatm.llm.limits import reset_rate_limits, set_rate_limits

    reset_rate_limits()
    set_rate_limits("fake/embedding", initial_backoff=0.01, max_backoff=0.01)
    yield
    reset_rate_limits()


def texts(n: int) -> list[str]:
    return [str(i) for i in range(n)]


def values(embeddings) -> list[float]:
    return [float(e[0]) for e in embeddings]


def test_api_models_are_not_local():
    assert ef.GoogleEmbeddingFunction.is_local is False
    assert ef.OpenAIEmbeddingFunction.is_local is False


def test_api_calls_are_split_within_the_providers_limit(fast_retries):
    model = FakeApiModel()

    assert values(model(texts(8))) == [float(i) for i in range(8)]
    assert [len(r) for r in model.requests] == [3, 3, 2]


def test_api_calls_are_split_into_one_request_per_slot_when_they_fit(fast_retries):
    model = FakeApiModel(concurrency=3)
    model.max_batch_size = 100

    assert values(model(texts(9))) == [float(i) for i in range(9)]
    assert sorted(len(r) for r in model.requests) == [3, 3, 3]


def test_api_requests_run_concurrently(fast_retries):
    barrier = threading.Barrier(3, timeout=5)

    class Waiting(FakeApiModel):
        def _embed(self, texts):
            barrier.wait()  # breaks unless 3 requests are in flight together
            return super()._embed(texts)

    model = Waiting(concurrency=3)

    assert values(model(texts(9))) == [float(i) for i in range(9)]


def test_rate_limit_and_server_errors_are_retried(fast_retries):
    model = FakeApiModel(failures=[StatusError(429), StatusError(503)])

    assert values(model(texts(2))) == [0.0, 1.0]
    assert len(model.requests) == 3


def test_other_errors_are_raised_at_once(fast_retries):
    model = FakeApiModel(failures=[StatusError(400)])

    with pytest.raises(StatusError):
        model(texts(2))
    assert len(model.requests) == 1


def test_api_calls_inside_an_event_loop_run_one_request_at_a_time(fast_retries):
    import asyncio

    model = FakeApiModel(concurrency=4)

    async def embed_in_loop():
        return model(texts(8))

    assert values(asyncio.run(embed_in_loop())) == [float(i) for i in range(8)]
    assert len(model.requests) == 4


def test_gemini_requests_hold_at_most_100_texts(monkeypatch, fast_retries):
    sent: list[int] = []

    class Models:
        def embed_content(self, model, contents):
            sent.append(len(contents))
            embedding = type("E", (), {"values": [1.0, 2.0]})
            return type("R", (), {"embeddings": [embedding] * len(contents)})

    client = type("Client", (), {"models": Models()})()
    monkeypatch.setattr(ef, "gemini_client", lambda api_key=None: client)

    model = ef.GoogleEmbeddingFunction(model="gemini-embedding-001")

    assert model.limiter_id == "gemini/gemini-embedding-001"
    assert len(model(texts(250))) == 250
    assert sent == [100, 100, 50]

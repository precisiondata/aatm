"""Embedding function implementations and model enums for vectorization.

This module defines embedding function adapters compatible with
ChromaDB's `EmbeddingFunction` interface. It provides implementations
for multiple embedding backends, including Google embeddings, Qwen3
embedding models served through SentenceTransformers, Gemma embedding
models served through SentenceTransformers, and OpenAI embedding
models.

The module also defines enum classes that centralize the supported model
identifiers for each provider-specific embedding family. These enums are
used to map user-facing or internal model names to the actual model
identifiers required by the corresponding client libraries.

Environment variables are loaded at import time using `dotenv` so that
provider credentials such as API keys can be resolved before embedding
clients are instantiated.

Classes:
    GoogleEmbeddingFunction:
        ChromaDB embedding function implementation backed by the Google
        GenAI embeddings API.
    Qwen3EmbeddingModels:
        Enumeration of supported Qwen3 embedding model identifiers.
    LocalSentenceTransformerEmbeddingFunction:
        Shared base for embedding models that run locally through
        SentenceTransformers.
    Qwen3EmbeddingFunction:
        ChromaDB embedding function implementation backed by a Qwen3
        embedding model loaded with SentenceTransformers.
    GemmaEmbeddingModels:
        Enumeration of supported Gemma embedding model identifiers.
    GemmaEmbeddingFunction:
        ChromaDB embedding function implementation backed by a Gemma
        embedding model loaded with SentenceTransformers.
    OpenAIEmbeddingModels:
        Enumeration of supported OpenAI embedding model identifiers.
    OpenAIEmbeddingFunction:
        ChromaDB embedding function implementation backed by the OpenAI
        embeddings API.

Notes:
    Qwen3 and Gemma embedding models are loaded onto the device returned by
    `aatm.device.get_device()`: an NVIDIA GPU (CUDA) or Apple GPU (MPS) when
    available, otherwise CPU. SentenceTransformers-based
    models are initialized with left-padding tokenizer behavior in this
    module.
"""

from enum import Enum
from typing import Any, ClassVar
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
import os
import dotenv
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import torch

from aatm.device import get_device
from aatm.precision import (
    EmbeddingDtype,
    EmbeddingPrecisionError,
    resolve_embedding_dtype,
    to_torch_dtype,
)

dotenv.load_dotenv()


class GoogleEmbeddingFunction(EmbeddingFunction):
    """Embed documents using the Google GenAI embeddings API.

    This embedding function adapts Google's embedding endpoint to the
    ChromaDB `EmbeddingFunction` interface. It creates a Google GenAI
    client using the `GOOGLE_API_KEY` environment variable and uses the
    configured model to embed incoming documents.

    Attributes:
        client: Google GenAI client used to request embeddings.
        model: Name of the embedding model to use with the Google API.
    """

    def __init__(self, model: str, *args: Any, **kwargs: Any) -> None:
        """Initialize the Google embedding function.

        Args:
            model: Name of the Google embedding model to use.
            *args: Additional positional arguments accepted for interface
                compatibility.
            **kwargs: Additional keyword arguments accepted for interface
                compatibility.

        Returns:
            None
        """
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = model

    def __call__(self, input: Documents) -> Embeddings:
        response = self.client.models.embed_content(
            model=self.model,
            contents=input,  # type: ignore[arg-type]
        )

        if response.embeddings is None:
            raise RuntimeError("Google embedding API returned no embeddings.")

        embeddings: Embeddings = []

        for emb in response.embeddings:
            if emb is None or emb.values is None:
                raise RuntimeError("Google embedding API returned an empty embedding.")

            embeddings.append(np.asarray(emb.values, dtype=np.float32))

        return embeddings


class Qwen3EmbeddingModels(Enum):
    """Enumerate the supported Qwen3 embedding models.

    This enum defines the available Qwen3 embedding model identifiers
    that can be used by `Qwen3EmbeddingFunction`.
    """

    QWEN3_06B = "Qwen/Qwen3-Embedding-0.6B"
    QWEN3_4B = "Qwen/Qwen3-Embedding-4B"
    QWEN3_8B = "Qwen/Qwen3-Embedding-8B"


class LocalSentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """Shared base for embedding models run locally with SentenceTransformers.

    Subclasses only choose which model identifiers they accept and which extra
    arguments they pass to `SentenceTransformer.encode`. Loading, device
    placement and batching live here so every local model behaves the same way
    when the vector database is built.

    The model is loaded onto the device returned by `aatm.device.get_device()`
    (CUDA, MPS or CPU), in the precision chosen by
    `aatm.precision.resolve_embedding_dtype`: ``float32`` unless half precision
    is requested.

    Attributes:
        is_local: Always ``True``. Marks the class as a local model, which lets
            the vector database builder tune the GPU batch size for it.
        model_id: Fully qualified identifier of the loaded model.
        dtype: Precision the model runs in.
        model: SentenceTransformer instance used to encode documents.
        batch_size: Number of texts encoded per forward pass.
    """

    is_local: ClassVar[bool] = True

    encode_kwargs: ClassVar[dict[str, Any]] = {}
    """Extra keyword arguments passed to every `SentenceTransformer.encode` call."""

    unsupported_dtypes: ClassVar[frozenset[str]] = frozenset()
    """Precisions known to produce broken embeddings with this model."""

    def __init__(
        self, model_id: str, batch_size: int = 32, dtype: str | None = None
    ) -> None:
        """Load the model onto the selected device.

        Args:
            model_id: Fully qualified identifier of the model to load.
            batch_size: Number of texts encoded per forward pass.
            dtype: Precision to load the model in (``float32``, ``bfloat16`` or
                ``float16``). If not provided, ``AATM_EMBEDDING_DTYPE`` or the
                default ``float32`` is used.

        Returns:
            None

        Raises:
            EmbeddingPrecisionError: If the precision is unsupported by the
                device or the model.
        """
        device = get_device()
        self.dtype: EmbeddingDtype = resolve_embedding_dtype(dtype, device)
        if self.dtype in self.unsupported_dtypes:
            raise EmbeddingPrecisionError(
                f"{model_id} does not support {self.dtype} precision."
            )

        self.model_id = model_id
        # Pass the dtype explicitly: recent transformers versions otherwise
        # load the checkpoint's own dtype, which is bfloat16 for Qwen3.
        self.model = SentenceTransformer(
            self.model_id,
            model_kwargs={"device_map": device, "dtype": to_torch_dtype(self.dtype)},
            tokenizer_kwargs={"padding_side": "left"},
        )
        self.model.eval()
        self.batch_size = batch_size

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for the provided documents.

        Args:
            input: Documents to be embedded.

        Returns:
            One float32 vector per document, already on the CPU.
        """
        with torch.inference_mode():
            embeddings = self.model.encode(
                input,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                **self.encode_kwargs,
            )
        return list(embeddings)


class Qwen3EmbeddingFunction(LocalSentenceTransformerEmbeddingFunction):
    """Embed documents using a Qwen3 SentenceTransformer model.

    Documents are encoded with the model's ``"query"`` prompt.
    """

    encode_kwargs: ClassVar[dict[str, Any]] = {"prompt_name": "query"}

    def __init__(
        self,
        model: str,
        batch_size: int = 32,
        dtype: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the Qwen3 embedding function.

        Args:
            model: Enum key corresponding to a supported Qwen3 embedding
                model.
            batch_size: Number of texts encoded per forward pass.
            dtype: Precision to load the model in. See
                `LocalSentenceTransformerEmbeddingFunction`.
            *args: Additional positional arguments accepted for interface
                compatibility.
            **kwargs: Additional keyword arguments accepted for interface
                compatibility.

        Returns:
            None

        Raises:
            ValueError: If `model` is not a valid `Qwen3EmbeddingModels`
                value.
        """
        super().__init__(
            Qwen3EmbeddingModels(model).value, batch_size=batch_size, dtype=dtype
        )


class GemmaEmbeddingModels(Enum):
    """Enumerate the supported Gemma embedding models.

    This enum defines the available Gemma embedding model identifiers
    that can be used by `GemmaEmbeddingFunction`.
    """

    EMBEDDING_GEMMA_300M = "google/embeddinggemma-300M"


class GemmaEmbeddingFunction(LocalSentenceTransformerEmbeddingFunction):
    """Embed documents using a Gemma SentenceTransformer model.

    EmbeddingGemma does not support ``float16``; use ``bfloat16`` or
    ``float32``.
    """

    unsupported_dtypes: ClassVar[frozenset[str]] = frozenset({"float16"})

    def __init__(
        self,
        model: str,
        batch_size: int = 32,
        dtype: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the Gemma embedding function.

        Args:
            model: Enum key corresponding to a supported Gemma embedding
                model.
            batch_size: Number of texts encoded per forward pass.
            dtype: Precision to load the model in. See
                `LocalSentenceTransformerEmbeddingFunction`.
            *args: Additional positional arguments accepted for interface
                compatibility.
            **kwargs: Additional keyword arguments accepted for interface
                compatibility.

        Returns:
            None

        Raises:
            ValueError: If `model` is not a valid `GemmaEmbeddingModels`
                value.
        """
        super().__init__(
            GemmaEmbeddingModels(model).value, batch_size=batch_size, dtype=dtype
        )


class OpenAIEmbeddingModels(Enum):
    """Enumerate the supported OpenAI embedding models.

    This enum defines the available OpenAI embedding model identifiers
    that can be used by `OpenAIEmbeddingFunction`.
    """

    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"


class OpenAIEmbeddingFunction(EmbeddingFunction):
    """Embed documents using the OpenAI embeddings API.

    This embedding function adapts OpenAI's embeddings endpoint to the
    ChromaDB `EmbeddingFunction` interface. It resolves the configured
    model identifier from `OpenAIEmbeddingModels` and uses an `OpenAI`
    client to generate embeddings for input documents.

    Attributes:
        model_id: Identifier of the selected OpenAI embedding model.
        client: OpenAI client used to request embeddings.
    """

    def __init__(self, model: str, *args: Any, **kwargs: Any) -> None:
        """Initialize the OpenAI embedding function.

        Args:
            model: Enum key corresponding to a supported OpenAI embedding
                model.
            *args: Additional positional arguments accepted for interface
                compatibility.
            **kwargs: Additional keyword arguments accepted for interface
                compatibility.

        Returns:
            None

        Raises:
            ValueError: If `model` is not a valid `OpenAIEmbeddingModels`
                value.
        """
        self.model_id = OpenAIEmbeddingModels(model).value
        self.client = OpenAI()

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for the provided documents.

        Args:
            input: Documents to be embedded.

        Returns:
            Embeddings returned by the OpenAI embeddings API in the format
            expected by ChromaDB.
        """
        response = self.client.embeddings.create(input=input, model=self.model_id)

        embeddings: Embeddings = []

        if response.data is None:
            raise RuntimeError("OpenAI embedding API returned no embeddings.")

        for emb in response.data:
            if emb is None or emb.embedding is None:
                raise RuntimeError("OpenAI embedding API returned an empty embedding.")
            embeddings.append(np.asarray(emb.embedding, dtype=np.float32))

        return embeddings

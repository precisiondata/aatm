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
    Qwen3 and Gemma embedding models are loaded onto CUDA when a GPU is
    available and otherwise fall back to CPU. SentenceTransformers-based
    models are initialized with left-padding tokenizer behavior in this
    module.
"""

from enum import Enum
from typing import Any
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
import os
import dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import torch

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
        """Generate embeddings for the provided documents.

        Args:
            input: Documents to be embedded.

        Returns:
            Embeddings returned by the Google embedding API, converted to the
            format expected by ChromaDB.
        """
        response = self.client.models.embed_content(
            model=self.model,
            contents=input,
        )
        return [emb.values for emb in response.embeddings]


class Qwen3EmbeddingModels(Enum):
    """Enumerate the supported Qwen3 embedding models.

    This enum defines the available Qwen3 embedding model identifiers
    that can be used by `Qwen3EmbeddingFunction`.
    """

    QWEN3_06B = "Qwen/Qwen3-Embedding-0.6B"
    QWEN3_4B = "Qwen/Qwen3-Embedding-4B"
    QWEN3_8B = "Qwen/Qwen3-Embedding-8B"


class Qwen3EmbeddingFunction(EmbeddingFunction):
    """Embed documents using a Qwen3 SentenceTransformer model.

    This embedding function adapts Qwen3 embedding models loaded through
    SentenceTransformers to the ChromaDB `EmbeddingFunction` interface.
    The selected model is loaded onto CUDA when available and otherwise
    onto CPU.

    Attributes:
        model_id: Fully qualified identifier of the selected Qwen3
            embedding model.
        model: SentenceTransformer instance used to encode documents.
    """

    def __init__(self, model: str, *args: Any, **kwargs: Any) -> None:
        """Initialize the Qwen3 embedding function.

        Args:
            model: Enum key corresponding to a supported Qwen3 embedding
                model.
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
        self.model_id = Qwen3EmbeddingModels(model).value
        self.model = SentenceTransformer(
            self.model_id,
            model_kwargs={"device_map": "cuda" if torch.cuda.is_available() else "cpu"},
            tokenizer_kwargs={"padding_side": "left"},
        )

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for the provided documents.

        This method encodes the input documents using the SentenceTransformer
        model with the `"query"` prompt configuration.

        Args:
            input: Documents to be embedded.

        Returns:
            Embeddings for the provided documents as a list of vectors.
        """
        embeddings = self.model.encode(input, prompt_name="query")
        return embeddings.tolist()


class GemmaEmbeddingModels(Enum):
    """Enumerate the supported Gemma embedding models.

    This enum defines the available Gemma embedding model identifiers
    that can be used by `GemmaEmbeddingFunction`.
    """

    EMBEDDING_GEMMA_300M = "google/embeddinggemma-300M"


class GemmaEmbeddingFunction(EmbeddingFunction):
    """Embed documents using a Gemma SentenceTransformer model.

    This embedding function adapts Gemma embedding models loaded through
    SentenceTransformers to the ChromaDB `EmbeddingFunction` interface.
    The selected model is loaded onto CUDA when available and otherwise
    onto CPU.

    Attributes:
        model_id: Fully qualified identifier of the selected Gemma
            embedding model.
        model: SentenceTransformer instance used to encode documents.
    """

    def __init__(self, model: str, *args: Any, **kwargs: Any) -> None:
        """Initialize the Gemma embedding function.

        Args:
            model: Enum key corresponding to a supported Gemma embedding
                model.
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
        self.model_id = GemmaEmbeddingModels(model).value
        self.model = SentenceTransformer(
            self.model_id,
            model_kwargs={"device_map": "cuda" if torch.cuda.is_available() else "cpu"},
            tokenizer_kwargs={"padding_side": "left"},
        )

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for the provided documents.

        Args:
            input: Documents to be embedded.

        Returns:
            Embeddings for the provided documents as a list of vectors.
        """
        embeddings = self.model.encode(input)
        return embeddings.tolist()


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
        return [emb.embedding for emb in response.data]

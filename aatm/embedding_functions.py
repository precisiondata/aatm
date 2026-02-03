from enum import Enum
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
import os
import dotenv
from sentence_transformers import SentenceTransformer
import torch

dotenv.load_dotenv()


class GoogleEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model: str, *args, **kwargs):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = model

    def __call__(self, input: Documents) -> Embeddings:
        response = self.client.models.embed_content(
            model=self.model,
            contents=input,
        )
        return [emb.values for emb in response.embeddings]


class Qwen3Models(Enum):
    QWEN3_06B = "Qwen/Qwen3-Embedding-0.6B"
    QWEN3_4B = "Qwen/Qwen3-Embedding-4B"
    QWEN3_8B = "Qwen/Qwen3-Embedding-8B"


class Qwen3EmbeddingFunction(EmbeddingFunction):
    def __init__(self, model: str, *args, **kwargs):
        self.model_id = Qwen3Models(model).value
        self.model = SentenceTransformer(
            self.model_id,
            model_kwargs={"device_map": "cuda" if torch.cuda.is_available() else "cpu"},
            tokenizer_kwargs={"padding_side": "left"},
        )

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(input, prompt_name="query")
        return embeddings.tolist()


class GemmaEmbeddingModels(Enum):
    EMBEDDING_GEMMA_300M = "google/embeddinggemma-300M"


class GemmaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model: str, *args, **kwargs):
        self.model_id = GemmaEmbeddingModels(model).value
        self.model = SentenceTransformer(
            self.model_id,
            model_kwargs={"device_map": "cuda" if torch.cuda.is_available() else "cpu"},
            tokenizer_kwargs={"padding_side": "left"},
        )

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(input)
        return embeddings.tolist()

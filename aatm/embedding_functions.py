from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
import os
import dotenv

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

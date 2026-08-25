"""OpenAI embedding client wrapper."""
from __future__ import annotations

from openai import OpenAI

from .config import settings


class EmbeddingClient:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or settings.embedding_model
        key = api_key or settings.openai_api_key
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file or export it "
                "before running ingestion or the MCP server."
            )
        self._client = OpenAI(api_key=key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. OpenAI accepts up to 2048 inputs per call;
        callers batch upstream (see ingest/pipeline.py) to stay well under that."""
        if not texts:
            return []
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

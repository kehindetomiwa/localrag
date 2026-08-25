"""LangChain OpenAI embeddings factory."""
from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from .config import settings


def get_embeddings(model: str | None = None, api_key: str | None = None) -> OpenAIEmbeddings:
    key = api_key or settings.openai_api_key
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file or export it "
            "before running ingestion or the MCP server."
        )
    return OpenAIEmbeddings(model=model or settings.embedding_model, api_key=key)

"""Embedding provider factory."""
from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from .config import settings


def get_embeddings(
    model: str | None = None,
    api_key: str | None = None,
):
    provider = settings.embedding_provider.lower()
    selected_model = model or settings.embedding_model

    if provider == "huggingface":
        return HuggingFaceEmbeddings(model_name=selected_model)

    if provider == "openai":
        key = api_key or settings.openai_api_key
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )
        return OpenAIEmbeddings(model=selected_model, api_key=key)

    raise ValueError(
        f"Unsupported embedding provider: {settings.embedding_provider}"
    )
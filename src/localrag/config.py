"""Central configuration for LocalRAG.

Resolution order (highest priority first):
1. Explicit function/CLI arguments passed by the caller
2. Environment variables (loaded from a .env file if present)
3. Hard-coded defaults below

Two values are intentionally NEVER hard-coded here: the ingestion source
path and the target collection ("DB") name. Both are required arguments
on the CLI/tooling that use them, so the same codebase can ingest and
serve any number of projects.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY") or None

    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "1536"))

    chunk_max_tokens: int = int(os.getenv("CHUNK_MAX_TOKENS", "500"))
    chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))


settings = Settings()

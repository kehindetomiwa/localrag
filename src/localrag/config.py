"""Central configuration for LocalRAG.(GENERAL SETTINGS)

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
#This is used to skip all the boring __init__, __str__ and other decorators stuff making the class easier to write
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()
#Qdrant is a stronger vector database(like chroma but Stronger)

@dataclass
class Settings:
    #creating variables and assigning them to the env variables alongside their default
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY") or None #variable can be a string or None

    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    # embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    #This specifies the numbers per embedding
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "384")) # Use 1536 when using the API model
    embedding_provider: str= os.getenv("EMBEDDING_PROVIDER", "huggingface")
    embedding_model: str = os.getenv("EMBEDDING_MODEL","sentence-transformers/all-MiniLM-L6-v2",)


    chunk_max_tokens: int = int(os.getenv("CHUNK_MAX_TOKENS", "500"))
    chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))


settings = Settings()

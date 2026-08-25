"""Ingestion orchestration: load -> chunk -> embed -> upsert.

Both `docs_path` (the folder to ingest) and `collection_name` (the target
Qdrant "DB") are required arguments here, never hard-coded, so this one
pipeline can serve any number of projects/doc sets.
"""
from __future__ import annotations

import hashlib
import logging

from ..embeddings import EmbeddingClient
from ..vectorstore import VectorStore
from .chunker import chunk_document
from .loader import load_documents

logger = logging.getLogger(__name__)

BATCH_SIZE = 64


def _stable_id(doc_name: str, chunk_index: int) -> int:
    """Deterministic point ID so re-ingesting the same doc overwrites its
    old chunks cleanly instead of duplicating them."""
    digest = hashlib.sha256(f"{doc_name}::{chunk_index}".encode()).hexdigest()
    return int(digest[:16], 16)


def run_ingestion(
    docs_path: str,
    collection_name: str,
    recreate_collection: bool = False,
) -> dict:
    """Ingest every supported file under `docs_path` into `collection_name`.

    Returns a summary dict: {"documents": int, "chunks": int, "collection": str}
    """
    documents = load_documents(docs_path)
    if not documents:
        logger.warning("No supported documents found under %s", docs_path)
        return {"documents": 0, "chunks": 0, "collection": collection_name}

    embedder = EmbeddingClient()
    store = VectorStore(collection_name=collection_name)
    store.ensure_collection(recreate=recreate_collection)

    total_chunks = 0
    for doc in documents:
        chunks = chunk_document(doc.content)
        if not chunks:
            continue

        texts = [c.content for c in chunks]
        points = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[start : start + BATCH_SIZE]
            batch_chunks = chunks[start : start + BATCH_SIZE]
            vectors = embedder.embed(batch_texts)
            for chunk, vector in zip(batch_chunks, vectors):
                points.append(
                    {
                        "id": _stable_id(doc.doc_name, chunk.chunk_index),
                        "vector": vector,
                        "payload": {
                            "doc_name": doc.doc_name,
                            "doc_group": doc.doc_group,
                            "doc_type": doc.doc_type,
                            "source_path": doc.source_path,
                            "section_title": chunk.section_title,
                            "chunk_index": chunk.chunk_index,
                            "content": chunk.content,
                        },
                    }
                )
        store.upsert(points)
        total_chunks += len(points)
        logger.info("Ingested %s (%d chunks)", doc.doc_name, len(points))

    return {"documents": len(documents), "chunks": total_chunks, "collection": collection_name}

"""Lightweight lexical rerank on top of vector search results.

Blends the vector similarity score with a BM25 score computed over the
candidate set, so exact keyword matches (function names, config keys)
that the embedding model might under-weight still surface near the top.
This avoids standing up a second, persistent sparse-vector index just for
a hybrid signal -- LangChain's `BM25Retriever` is built fresh, in memory,
over just the candidate documents returned by the vector search.
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

VECTOR_WEIGHT = 0.7
BM25_WEIGHT = 0.3


@dataclass
class RankedChunk:
    doc_name: str
    doc_group: str
    section_title: str
    content: str
    source_path: str
    score: float


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _normalize(value: float, lo: float, hi: float) -> float:
    return (value - lo) / (hi - lo) if hi > lo else 0.0


def rerank(
    query: str, candidates: list[tuple[Document, float]], top_n: int = 8
) -> list[RankedChunk]:
    """candidates: [(Document, vector_score), ...] from VectorStore.search"""
    if not candidates:
        return []

    docs = [doc for doc, _ in candidates]
    bm25 = BM25Retriever.from_documents(docs, preprocess_func=_tokenize)
    bm25_scores = bm25.vectorizer.get_scores(_tokenize(query))

    vector_scores = [score for _, score in candidates]
    v_lo, v_hi = min(vector_scores), max(vector_scores)
    b_lo, b_hi = min(bm25_scores), max(bm25_scores)

    blended: list[RankedChunk] = []
    for (doc, vector_score), bm25_score in zip(candidates, bm25_scores):
        v_norm = _normalize(vector_score, v_lo, v_hi)
        b_norm = _normalize(bm25_score, b_lo, b_hi)
        final_score = VECTOR_WEIGHT * v_norm + BM25_WEIGHT * b_norm

        blended.append(
            RankedChunk(
                doc_name=doc.metadata["doc_name"],
                doc_group=doc.metadata["doc_group"],
                section_title=doc.metadata["section_title"],
                content=doc.page_content,
                source_path=doc.metadata["source_path"],
                score=final_score,
            )
        )

    blended.sort(key=lambda c: c.score, reverse=True)
    return blended[:top_n]

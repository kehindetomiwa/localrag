"""Lightweight lexical rerank on top of vector search results.

Blends the vector similarity score with a BM25 score computed over the
candidate set, so exact keyword matches (function names, config keys)
that the embedding model might under-weight still surface near the top.
This avoids standing up a second, sparse-vector index just for a hybrid
signal.
"""
from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

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


def rerank(query: str, candidates: list[dict], top_n: int = 8) -> list[RankedChunk]:
    """candidates: [{"score": float, "payload": {...}}, ...] from VectorStore.search"""
    if not candidates:
        return []

    corpus = [_tokenize(c["payload"]["content"]) for c in candidates]
    bm25 = BM25Okapi(corpus)
    bm25_scores = bm25.get_scores(_tokenize(query))

    vector_scores = [c["score"] for c in candidates]
    v_lo, v_hi = min(vector_scores), max(vector_scores)
    b_lo, b_hi = min(bm25_scores), max(bm25_scores)

    blended: list[RankedChunk] = []
    for candidate, bm25_score in zip(candidates, bm25_scores):
        v_norm = _normalize(candidate["score"], v_lo, v_hi)
        b_norm = _normalize(bm25_score, b_lo, b_hi)
        final_score = VECTOR_WEIGHT * v_norm + BM25_WEIGHT * b_norm

        payload = candidate["payload"]
        blended.append(
            RankedChunk(
                doc_name=payload["doc_name"],
                doc_group=payload["doc_group"],
                section_title=payload["section_title"],
                content=payload["content"],
                source_path=payload["source_path"],
                score=final_score,
            )
        )

    blended.sort(key=lambda c: c.score, reverse=True)
    return blended[:top_n]

"""Lightweight query analysis.

Detects whether a user query references a specific document name or group
so retrieval can be scoped instead of searching the whole collection. This
is the main defense against confusing two structurally-similar documents.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QueryIntent:
    doc_name: str | None = None
    doc_group: str | None = None


def analyze_query(
    query: str,
    known_doc_names: list[str],
    known_doc_groups: list[str],
) -> QueryIntent:
    """Naive substring match against known tags. Good enough as a first
    pass; swap in an LLM-based classifier later if precision matters more
    (e.g. fuzzy matches, synonyms, typos)."""
    lowered = query.lower()

    doc_name = next((n for n in known_doc_names if n.lower() in lowered), None)
    doc_group = next((g for g in known_doc_groups if g.lower() in lowered), None)

    return QueryIntent(doc_name=doc_name, doc_group=doc_group)

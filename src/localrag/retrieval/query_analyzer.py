"""Lightweight query analysis.

Detects whether a user query references a specific document name or group
so retrieval can be scoped instead of searching the whole collection. This
is the main defense against confusing two structurally-similar documents.
Its used to make the users query more specific and reduce the number of
chunks returned.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QueryIntent:
    doc_name: str | None = None
    doc_group: str | None = None


def analyze_query(
    query: str, #The users questions
    known_doc_names: list[str], #the list of documents ingested into the Qdrant
    known_doc_groups: list[str],#all the categories of documents ingested into the Qdrant for a better search
) -> QueryIntent:
    """Naive substring match against known tags. Good enough as a first
    pass; swap in an LLM-based classifier later if precision matters more
    (e.g. fuzzy matches, synonyms, typos)."""
    lowered = query.lower()

    #Python generators expression, next is used to find the first match
    #Checking if the user query matches any of the docs name at all and returns the first match or return None
    doc_name = next((n for n in known_doc_names if n.lower() in lowered), None)
    doc_group = next((g for g in known_doc_groups if g.lower() in lowered), None)

    return QueryIntent(doc_name=doc_name, doc_group=doc_group)

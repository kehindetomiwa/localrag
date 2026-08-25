"""Structure-aware chunking.

Splits documents along markdown header boundaries so chunks map onto
logical sections (e.g. "Overview", "Setup", "API"). Because sibling
documents in the same doc_group share a template, this keeps same-named
sections comparable across documents, while the doc_name/doc_group tags
added during ingestion keep them distinguishable at retrieval time.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

from ..config import settings

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
_encoding = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    section_title: str
    content: str
    chunk_index: int
    token_count: int


def _token_len(text: str) -> int:
    return len(_encoding.encode(text))


def _split_by_headers(content: str) -> list[tuple[str, str]]:
    """Returns [(section_title, section_text), ...].
    Falls back to a single 'Document' section if no headers are found
    (e.g. plain .txt files)."""
    matches = list(_HEADER_RE.finditer(content))
    if not matches:
        return [("Document", content)]

    sections: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        title = match.group(2).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        sections.append((title, content[start:end].strip()))
    return sections


def _split_oversized(
    title: str, text: str, max_tokens: int, overlap_tokens: int
) -> list[tuple[str, str]]:
    """Token-window split for sections that exceed max_tokens, with overlap
    so context isn't lost at the boundary."""
    tokens = _encoding.encode(text)
    if len(tokens) <= max_tokens:
        return [(title, text)]

    parts: list[tuple[str, str]] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        parts.append((title, _encoding.decode(tokens[start:end])))
        if end == len(tokens):
            break
        start = end - overlap_tokens
    return parts


def chunk_document(
    content: str,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[Chunk]:
    max_tokens = max_tokens or settings.chunk_max_tokens
    overlap_tokens = overlap_tokens or settings.chunk_overlap_tokens

    chunks: list[Chunk] = []
    for title, section_text in _split_by_headers(content):
        for sub_title, sub_text in _split_oversized(title, section_text, max_tokens, overlap_tokens):
            if not sub_text.strip():
                continue
            chunks.append(
                Chunk(
                    section_title=sub_title,
                    content=sub_text,
                    chunk_index=len(chunks),
                    token_count=_token_len(sub_text),
                )
            )
    return chunks

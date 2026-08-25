"""Structure-aware chunking via LangChain text splitters.

Splits documents along markdown header boundaries first (`MarkdownHeaderTextSplitter`)
so chunks map onto logical sections (e.g. "Overview", "Setup", "API"), then
runs a token-window split with overlap (`TokenTextSplitter`) over any section
that's still too large. Because sibling documents in the same doc_group share
a template, this keeps same-named sections comparable across documents, while
the doc_name/doc_group tags added during ingestion keep them distinguishable
at retrieval time.
"""
from __future__ import annotations

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, TokenTextSplitter

from ..config import settings

_HEADERS_TO_SPLIT_ON = [("#" * n, f"h{n}") for n in range(1, 7)]
_encoding = tiktoken.get_encoding("cl100k_base")


def _section_title(section: Document) -> str:
    for n in range(6, 0, -1):
        title = section.metadata.get(f"h{n}")
        if title:
            return title
    return "Document"


def chunk_document(
    content: str,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[Document]:
    max_tokens = max_tokens or settings.chunk_max_tokens
    overlap_tokens = overlap_tokens or settings.chunk_overlap_tokens

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT_ON, strip_headers=False
    )
    token_splitter = TokenTextSplitter(
        encoding_name="cl100k_base", chunk_size=max_tokens, chunk_overlap=overlap_tokens
    )

    chunks: list[Document] = []
    for section in header_splitter.split_text(content):
        title = _section_title(section)
        for sub_text in token_splitter.split_text(section.page_content):
            if not sub_text.strip():
                continue
            chunks.append(
                Document(
                    page_content=sub_text,
                    metadata={
                        "section_title": title,
                        "chunk_index": len(chunks),
                        "token_count": len(_encoding.encode(sub_text)),
                    },
                )
            )
    return chunks

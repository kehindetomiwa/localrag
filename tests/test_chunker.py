"""Tests for localrag.ingest.chunker.

Note: tiktoken downloads its BPE encoding file on first use if it isn't
already cached locally, so these tests need network access the first
time they run in a fresh environment.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localrag.ingest.chunker import chunk_document  # noqa: E402


def test_splits_on_headers():
    content = "# Overview\nintro text\n\n## Setup\nsetup text\n"
    chunks = chunk_document(content, max_tokens=500, overlap_tokens=50)

    titles = [c.metadata["section_title"] for c in chunks]
    assert titles == ["Overview", "Setup"]


def test_no_headers_falls_back_to_single_chunk():
    content = "plain text with no headers at all"
    chunks = chunk_document(content, max_tokens=500, overlap_tokens=50)

    assert len(chunks) == 1
    assert chunks[0].metadata["section_title"] == "Document"


def test_oversized_section_is_split_with_overlap():
    content = "# Big\n" + ("word " * 2000)
    chunks = chunk_document(content, max_tokens=100, overlap_tokens=10)

    assert len(chunks) > 1
    assert all(c.metadata["section_title"] == "Big" for c in chunks)
    assert all(c.metadata["token_count"] <= 100 for c in chunks)

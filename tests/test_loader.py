"""Tests for localrag.ingest.loader. Pure filesystem logic, no network/API needed."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localrag.ingest.loader import load_documents  # noqa: E402


def test_doc_name_and_group_are_derived_from_path(tmp_path):
    (tmp_path / "services").mkdir()
    (tmp_path / "services" / "auth.md").write_text("# Auth\ncontent")
    (tmp_path / "services" / "billing.md").write_text("# Billing\ncontent")

    docs = load_documents(tmp_path)

    names = {d.metadata["doc_name"] for d in docs}
    groups = {d.metadata["doc_group"] for d in docs}

    assert names == {"services/auth", "services/billing"}
    assert groups == {"services"}


def test_traverses_nested_subfolders():
    """Regression test: a docs folder whose subfolders contain their own
    subfolders (e.g. docs/architecture/decisions/*.md) must be walked all
    the way down, not just one level deep."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "architecture" / "decisions").mkdir(parents=True)
        (root / "architecture" / "overview.md").write_text("# Overview\ntop level")
        (root / "architecture" / "decisions" / "adr-001.md").write_text("# ADR 1\nnested")

        docs = load_documents(root)
        names = {d.metadata["doc_name"] for d in docs}
        groups = {d.metadata["doc_group"] for d in docs}

        assert names == {"architecture/overview", "architecture/decisions/adr-001"}
        assert groups == {"architecture", "decisions"}


def test_unsupported_extensions_are_skipped(tmp_path):
    (tmp_path / "notes.md").write_text("# Notes\nkept")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")

    docs = load_documents(tmp_path)

    assert len(docs) == 1
    assert docs[0].metadata["doc_name"] == "notes"


def test_missing_path_raises(tmp_path):
    missing = tmp_path / "does-not-exist"
    try:
        load_documents(missing)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass

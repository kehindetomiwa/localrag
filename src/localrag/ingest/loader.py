"""Load files from a directory tree and attach document identity metadata.

doc_name  -- unique identifier for the document (relative path, no extension)
doc_group -- the parent directory name; documents in the same folder are
             assumed to follow the same template/structure, which is the
             whole reason grouping matters for retrieval.
doc_type  -- file extension, kept for filtering/debugging.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".md", ".mdx", ".txt", ".rst"}


@dataclass
class LoadedDocument:
    doc_name: str
    doc_group: str
    doc_type: str
    source_path: str
    content: str


def load_documents(root_path: str | Path) -> list[LoadedDocument]:
    """Recursively load every supported file under `root_path`.

    `root_path` is a required argument, not a default/constant, so the same
    loader can be pointed at any docs folder for any project.
    """
    root = Path(root_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Ingestion path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Ingestion path is not a directory: {root}")

    documents: list[LoadedDocument] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        relative = path.relative_to(root)
        doc_name = str(relative.with_suffix(""))
        doc_group = relative.parent.name if str(relative.parent) != "." else root.name

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # skip binary/non-utf8 files rather than fail the whole run

        if not content.strip():
            continue

        documents.append(
            LoadedDocument(
                doc_name=doc_name,
                doc_group=doc_group,
                doc_type=path.suffix.lstrip(".").lower(),
                source_path=str(path),
                content=content,
            )
        )
    return documents

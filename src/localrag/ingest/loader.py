"""Load files from a directory tree and attach document identity metadata.

Returns LangChain `Document` objects with metadata:

doc_name  -- unique identifier for the document (relative path, no extension)
doc_group -- the parent directory name; documents in the same folder are
             assumed to follow the same template/structure, which is the
             whole reason grouping matters for retrieval.
doc_type  -- file extension, kept for filtering/debugging.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

#Supported file extensions
SUPPORTED_EXTENSIONS = {".md", ".mdx", ".txt", ".rst"}

#Loading all the supported files under a directory/inside a folder
def load_documents(root_path: str | Path) -> list[Document]:
    """Recursively load every supported file under `root_path`, at any
    nesting depth -- not just the top level.

    `Path.rglob("*")` walks the entire directory tree, so files inside
    subfolders of subfolders (e.g. `docs/architecture/decisions/*.md`) are
    picked up along with files sitting directly under `root_path`.

    `root_path` is a required argument, not a default/constant, so the same
    loader can be pointed at any docs folder for any project.
    """
    # .expanduser- replaces ~ with the user's actual home directory
    # .resolve()- makes the path absolute by resolving all relevant part like . or .. 
    root = Path(root_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Ingestion path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Ingestion path is not a directory: {root}")

    documents: list[Document] = []
    #loop through all the files in the directory and sort them in the alphabetical order and the way they are nested
    for path in sorted(root.rglob("*")):
        #Checking if the file is valid
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        #Getting the relative path to the root directory instead of the jargon of absolute path e.g docs/pi.md
        relative = path.relative_to(root)
        #Getting the name of the file without the extension e.g docs/pi
        doc_name = str(relative.with_suffix(""))
        #extracts the folder e,g docs/pi.md
        doc_group = relative.parent.name if str(relative.parent) != "." else root.name

        try:
            #Reading the file and converting it to string
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # skip binary/non-utf8 files rather than fail the whole run

        if not content.strip():
            continue

        # Add the document to the list
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "doc_name": doc_name,
                    "doc_group": doc_group,
                    #Removes whitespace or extra things from the left only 
                    "doc_type": path.suffix.lstrip(".").lower(),
                    "source_path": str(path),
                },
            )
        )
    return documents

#!/usr/bin/env python
"""CLI entrypoint for ingestion.

Both the source folder and the target collection ("DB") name are required
arguments -- neither is hard-coded -- so this same script can ingest any
docs folder into any collection:

    python scripts/ingest.py --path /path/to/docs --collection edu365_docs
    python scripts/ingest.py --path /path/to/docs --collection edu365_docs --recreate
"""

#Used to improve compatability across different python versions
from __future__ import annotations

# This is used to read commands from the command line
import argparse
#This is used to print nice logs like ingesting
import logging

#Used to work with Files and directory
import sys
from pathlib import Path


#Find my project root by going up two folders from where I am, then look in the src folder, and add that folder to Python's search list.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

#The actual function that does the ingestion i.e the chunking and saving to the Qdrant
from localrag.ingest.pipeline import run_ingestion  # noqa: E402

#this will show us whats happening in the console
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    #
    parser = argparse.ArgumentParser(description="Ingest a documents folder into LocalRAG")
    parser.add_argument(
        "--path",
        required=True,
        help="Path to the folder of documents to ingest (dynamic, e.g. any project's docs/ folder)",
    )
    parser.add_argument(
        "--collection",
        required=True,
        help="Target Qdrant collection ('DB') name (dynamic, e.g. 'edu365_docs')",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the collection before ingesting (use for a clean re-index)",
    )
    args = parser.parse_args()

    summary = run_ingestion(
        docs_path=args.path,
        collection_name=args.collection,
        recreate_collection=args.recreate,
    )
    print(
        f"Ingested {summary['documents']} documents -> {summary['chunks']} chunks "
        f"into collection '{summary['collection']}'"
    )


if __name__ == "__main__":
    main()

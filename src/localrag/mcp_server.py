"""LocalRAG MCP server -- retrieval-only tools for Claude Code.

Runs locally over stdio. Claude Code spawns this as a subprocess, so there
is no network exposure or auth to manage. The collection ("DB") name is
fixed at startup via --collection / LOCALRAG_COLLECTION: one server
instance always talks to exactly one Qdrant collection, so point Claude
Code's MCP config at a different --collection per project.

This server does retrieval only -- it returns labeled, ranked chunks and
lets Claude Code's own model do the reasoning/generation over them.
"""
from __future__ import annotations

import argparse
import os

from mcp.server.fastmcp import FastMCP

from .retrieval.retriever import search
from .vectorstore import VectorStore


def build_server(collection_name: str) -> FastMCP:
    mcp = FastMCP("localrag")

    @mcp.tool()
    def search_codebase(
        query: str,
        doc_name: str | None = None,
        doc_group: str | None = None,
        top_n: int = 8,
    ) -> list[dict]:
        """Search the ingested documentation for chunks relevant to `query`.

        Pass doc_name or doc_group to scope the search to one document or
        one group of similarly-structured documents, avoiding confusion
        between them. If omitted, the query text is checked against known
        document names/groups first.
        """
        results = search(
            query=query,
            collection_name=collection_name,
            doc_name=doc_name,
            doc_group=doc_group,
            top_n=top_n,
        )
        return [
            {
                "doc_name": r.doc_name,
                "doc_group": r.doc_group,
                "section": r.section_title,
                "content": r.content,
                "source_path": r.source_path,
                "score": round(r.score, 4),
            }
            for r in results
        ]

    @mcp.tool()
    def list_documents() -> list[str]:
        """List every doc_name currently indexed in this collection."""
        store = VectorStore(collection_name=collection_name)
        return store.list_document_names()

    @mcp.tool()
    def list_document_groups() -> list[str]:
        """List every doc_group currently indexed in this collection."""
        store = VectorStore(collection_name=collection_name)
        return store.list_document_groups()

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="LocalRAG MCP server (stdio)")
    parser.add_argument(
        "--collection",
        default=os.getenv("LOCALRAG_COLLECTION"),
        help="Qdrant collection ('DB') name to serve. Also settable via LOCALRAG_COLLECTION.",
    )
    args = parser.parse_args()

    if not args.collection:
        parser.error("--collection is required (or set LOCALRAG_COLLECTION in the environment)")

    server = build_server(args.collection)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()

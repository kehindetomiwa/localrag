"""Retrieval orchestration: analyze query -> route -> vector search -> rerank."""
from __future__ import annotations

from ..vectorstore import VectorStore
from .query_analyzer import analyze_query
from .reranker import RankedChunk, rerank


def search(
    query: str,
    collection_name: str,
    doc_name: str | None = None,
    doc_group: str | None = None,
    top_k: int = 20,
    top_n: int = 8,
) -> list[RankedChunk]:
    """Search `collection_name` for chunks relevant to `query`.

    If `doc_name`/`doc_group` are not passed explicitly, they are inferred
    from the query text against the collection's known tags, so a caller
    (e.g. the MCP tool) doesn't have to do that work itself.
    """
    # TODO: support multiple collections
    #its used to connect to the db from the vector db
    store = VectorStore(collection_name=collection_name)

    if doc_name is None and doc_group is None:
        known_names = store.list_document_names() #Listing all the document names available in the vector db
        known_groups = store.list_document_groups()#Listing all the document groups available in the vector db
        #Narrowing down the search to the most relevant document name and group
        intent = analyze_query(query, known_doc_names=known_names, known_doc_groups=known_groups)
        doc_name = intent.doc_name
        doc_group = intent.doc_group

    #We then search the vector db for the chunks that match the query ad returns the optimized/best results
    hits = store.search(query, top_k=top_k, doc_name=doc_name, doc_group=doc_group)
    return rerank(query, hits, top_n=top_n)

"""
MAIN RAG PIPELINE
Ingestion orchestration: load -> chunk -> embed -> upsert.

Both `docs_path` (the folder to ingest) and `collection_name` (the target
Qdrant "DB") are required arguments here, never hard-coded, so this one
pipeline can serve any number of projects/doc sets.
"""




from __future__ import annotations
import hashlib
import logging
from langchain_core.documents import Document
from ..vectorstore import VectorStore
from .chunker import chunk_document
from .loader import load_documents


logger = logging.getLogger(__name__)

#Creatign a unique_id for a document chunk (Preventing duplicates)
#e.g "report.pdf, 0"
def _stable_id(doc_name: str, chunk_index: int) -> int:
    """Deterministic point ID so re-ingesting the same doc overwrites its
    old chunks cleanly instead of duplicating them."""
    #This whole line generates a 64 character hexadecimal
    #.encode converts the python unicode into an encoded bytes because this hash algorithm operate strictly on binary bytes
    # .sha256- Creates an internal SHA-256 hash object(encryption)
    # .hexdigits converts the oject into readable string of hex letters and numbers
    digest = hashlib.sha256(f"{doc_name}::{chunk_index}".encode()).hexdigest()
    #Takes the first 16 of the encrypted Unique ID
    return int(digest[:16], 16)


#Ingestion covers the entire cleaning, splitting, chunking, and vector embedding
def run_ingestion(
    docs_path: str,
    collection_name: str,
    recreate_collection: bool = False,
) -> dict:
    # docs_path -> path to the doc we want to ingest, collection_name -> where the data will be stored
    #Recreate_collection -> If its true it will reset the existing collection and ingest the data again, if false it will just overwrite existing data
    """Ingest every supported file under `docs_path` (recursing into every
    nested subfolder) into `collection_name`.

    Returns a summary dict: {"documents": int, "chunks": int, "collection": str}
    """
    #Our custom loader 
    documents = load_documents(docs_path)
    if not documents:
        logger.warning("No supported documents found under %s", docs_path)
        return {"documents": 0, "chunks": 0, "collection": collection_name}

    #Creating our vector store
    store = VectorStore(collection_name=collection_name)
    store.ensure_collection(recreate=recreate_collection)

    total_chunks = 0
    #Loop
    for doc in documents:
        #Chunking the extracted texts(.page_content) from the Document object
        chunks = chunk_document(doc.page_content)
        if not chunks:
            continue

        #Reconstructiong the document metadata by creating a list of new document object for each chunk
        to_index = [
            Document(
                page_content=chunk.page_content,
                metadata={
                    "doc_name": doc.metadata["doc_name"],
                    "doc_group": doc.metadata["doc_group"],
                    "doc_type": doc.metadata["doc_type"],
                    "source_path": doc.metadata["source_path"],
                    "section_title": chunk.metadata["section_title"],
                    "chunk_index": chunk.metadata["chunk_index"],
                },
            )
            for chunk in chunks #... Does this for each chunk
        ]
        #Creating a list of unique ids for each chunk
        ids = [
            _stable_id(doc.metadata["doc_name"], chunk.metadata["chunk_index"])
            for chunk in chunks
        ]

        store.add_documents(to_index, ids=ids)
        total_chunks += len(to_index)
        logger.info("Ingested %s (%d chunks)", doc.metadata["doc_name"], len(to_index))

    return {"documents": len(documents), "chunks": total_chunks, "collection": collection_name}

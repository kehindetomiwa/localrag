"""Qdrant vector store wrapper, backed by LangChain's QdrantVectorStore.

Every chunk is stored with doc_name/doc_group/doc_type as indexed payload
fields (nested under "metadata", which is where langchain-qdrant puts
Document metadata), so a search can be hard-filtered to a single document
or group instead of relying on embedding similarity alone to tell apart
chunks from structurally-similar documents.

Collection lifecycle (create/recreate/payload indexes) and the admin
list_* scans stay on the raw qdrant-client, since langchain-qdrant only
covers add/search, not collection administration.
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .config import settings
from .embeddings import get_embeddings

_FILTERABLE_FIELDS = ("metadata.doc_name", "metadata.doc_group", "metadata.doc_type")


class VectorStore:
    def __init__(
        self,
        collection_name: str,
        vector_size: int | None = None,
        url: str | None = None,
        api_key: str | None = None,
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size or settings.embedding_dim
        self.client = QdrantClient(
            url=url or settings.qdrant_url,
            api_key=api_key or settings.qdrant_api_key,
        )
        self._store: QdrantVectorStore | None = None

    def ensure_collection(self, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection_name)
        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.vector_size, distance=qmodels.Distance.COSINE
                ),
            )
            for field in _FILTERABLE_FIELDS:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )

    @property
    def store(self) -> QdrantVectorStore:
        """LangChain vector store bound to this collection, for add/search."""
        if self._store is None:
            self._store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=get_embeddings(),
            )
        return self._store

    def add_documents(self, documents: list[Document], ids: list[int]) -> None:
        self.store.add_documents(documents, ids=ids)

    def search(
        self,
        query: str,
        top_k: int = 20,
        doc_name: str | None = None,
        doc_group: str | None = None,
    ) -> list[tuple[Document, float]]:
        must: list[qmodels.FieldCondition] = []
        if doc_name:
            must.append(
                qmodels.FieldCondition(
                    key="metadata.doc_name", match=qmodels.MatchValue(value=doc_name)
                )
            )
        if doc_group:
            must.append(
                qmodels.FieldCondition(
                    key="metadata.doc_group", match=qmodels.MatchValue(value=doc_group)
                )
            )
        query_filter = qmodels.Filter(must=must) if must else None

        return self.store.similarity_search_with_score(query, k=top_k, filter=query_filter)

    def list_document_names(self) -> list[str]:
        """Distinct doc_name values currently indexed, via scroll (no separate
        metadata table needed for a project of this size)."""
        seen: set[str] = set()
        next_offset = None
        while True:
            records, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                limit=256,
                offset=next_offset,
            )
            for record in records:
                metadata = (record.payload or {}).get("metadata") or {}
                name = metadata.get("doc_name")
                if name:
                    seen.add(name)
            if next_offset is None:
                break
        return sorted(seen)

    def list_document_groups(self) -> list[str]:
        seen: set[str] = set()
        next_offset = None
        while True:
            records, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                limit=256,
                offset=next_offset,
            )
            for record in records:
                metadata = (record.payload or {}).get("metadata") or {}
                group = metadata.get("doc_group")
                if group:
                    seen.add(group)
            if next_offset is None:
                break
        return sorted(seen)

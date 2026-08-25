"""Qdrant vector store wrapper.

Every chunk is stored with doc_name/doc_group/doc_type as indexed payload
fields, so a search can be hard-filtered to a single document or group
instead of relying on embedding similarity alone to tell apart chunks
from structurally-similar documents.
"""
from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .config import settings

_FILTERABLE_FIELDS = ("doc_name", "doc_group", "doc_type")


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

    def upsert(self, points: list[dict[str, Any]]) -> None:
        """points: [{"id": int, "vector": [...], "payload": {...}}, ...]"""
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                qmodels.PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
                for p in points
            ],
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        doc_name: str | None = None,
        doc_group: str | None = None,
    ) -> list[qmodels.ScoredPoint]:
        must: list[qmodels.FieldCondition] = []
        if doc_name:
            must.append(
                qmodels.FieldCondition(key="doc_name", match=qmodels.MatchValue(value=doc_name))
            )
        if doc_group:
            must.append(
                qmodels.FieldCondition(key="doc_group", match=qmodels.MatchValue(value=doc_group))
            )
        query_filter = qmodels.Filter(must=must) if must else None

        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return result.points

    def list_document_names(self) -> list[str]:
        """Distinct doc_name values currently indexed, via scroll (no separate
        metadata table needed for a project of this size)."""
        seen: set[str] = set()
        next_offset = None
        while True:
            records, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=["doc_name"],
                limit=256,
                offset=next_offset,
            )
            for record in records:
                name = record.payload.get("doc_name") if record.payload else None
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
                with_payload=["doc_group"],
                limit=256,
                offset=next_offset,
            )
            for record in records:
                group = record.payload.get("doc_group") if record.payload else None
                if group:
                    seen.add(group)
            if next_offset is None:
                break
        return sorted(seen)

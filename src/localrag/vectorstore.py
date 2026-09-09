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

#Unlike Chromadb which helps us do all the vectordb under the hood quadrant gives us more control over the vector db
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .config import settings
from .embeddings import get_embeddings

_FILTERABLE_FIELDS = ("metadata.doc_name", "metadata.doc_group", "metadata.doc_type")


class VectorStore:
    def __init__(
        self,
        collection_name: str, #The name of our database inside Qdrant (the collection)
        vector_size: int | None = None,#Total number of embeddings per document
        url: str | None = None, # Where Qrant is located
        api_key: str | None = None,
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size or settings.embedding_dim
        #This connects us to the Qdranr server, Api key is required only if you want to access it remotely
        self.client = QdrantClient(
            url=url or settings.qdrant_url,
            api_key=api_key or settings.qdrant_api_key,
        )
        #Lazy loading is implemented here... the vector db isnt created yet until we actually need it 
        self._store: QdrantVectorStore | None = None

    #Handles creating and resetting the table inside the Qdrant where your chunk live
    def ensure_collection(self, recreate: bool = False) -> None:
        #Checks if a particular collection exists
        exists = self.client.collection_exists(self.collection_name)
        #If it exits and we want to recreate it, delete it
        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            exists = False
        if not exists: 
            #Create both the vector db(if it hasnt been created yet) and collection in the vector db
            self.client.create_collection(
                collection_name=self.collection_name,
                #This is used to describe how we want to use the blueprint to be structured and how vectors will be stored here
                vectors_config=qmodels.VectorParams(
                    #distance tells Qdrant to use cosine similarity to compate vectors durting search queries
                    size=self.vector_size, distance=qmodels.Distance.COSINE
                ),
            )
            #Ths is done so that the LLM can search easily like doc_name/doc_group/doc_type making the search faster
            for field in _FILTERABLE_FIELDS:
                #Tells the Qdrant to build a payload index(this is used to increase the speed of search queries)
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    #This is used to tell Qdrant what data type to use for the index and how to index them
                    field_schema=qmodels.PayloadSchemaType.KEYWORD, #This means exact matches only(Nt full text search)
                )

    #The store property is lazy loaded, so we dont have to worry about creating the vector db until we need it
    @property
    def store(self) -> QdrantVectorStore: #It must return a QdrantVectorStore
        """LangChain vector store bound to this collection, for add/search."""
        #Creating the db
        if self._store is None:
            self._store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=get_embeddings(),
            )
        return self._store

    #Overiding the add_documents method so we can assign each document a unique id
    def add_documents(self, documents: list[Document], ids: list[int]) -> None:
        """Addin chinks into our Qdrant vector db alongside their unique ID"""
        self.store.add_documents(documents, ids=ids)

    #This does the actual query
    def search(
        self,
        query: str,
        top_k: int = 20,
        #This are optional and are mainly used to narrow down the search to make the query more accurate
        doc_name: str | None = None,
        doc_group: str | None = None,
    ) -> list[tuple[Document, float]]:
        #creating a list that will store all possible filtering conditions
        must: list[qmodels.FieldCondition] = []
        #if there is a doc_name, we add it to the list of conditions the query will follow
        if doc_name:
            must.append(
                qmodels.FieldCondition(
                    #This tells the Qdrant that the doc_name(from our query) must match the docname field in the db
                    key="metadata.doc_name", match=qmodels.MatchValue(value=doc_name)
                )
            )
        if doc_group:
            must.append(
                qmodels.FieldCondition(
                    key="metadata.doc_group", match=qmodels.MatchValue(value=doc_group)
                )
            )
        #Creating a filter that will be used to narrow down the search
        query_filter = qmodels.Filter(must=must) if must else None

        #Returns the relevant chunks with the filter we created alongside the returned chunks from the highest score to the least
        return self.store.similarity_search_with_score(query, k=top_k, filter=query_filter)


#This scans through the Qdrant to pull all the document name 
    def list_document_names(self) -> list[str]:
        """Distinct doc_name values currently indexed, via scroll (no separate
        metadata table needed for a project of this size)."""
        seen: set[str] = set() #Used to prevent duplicates
        next_offset = None #This is used to paginate through the results so qdrant will know where to start and go next along the pages
        while True:
            #The .scroll calls the Qdrant api used to scroll down a feed and fetch records from a database page by page
            #Records holds the list of vector points that match the query while next_offset is the ID marker for the next page
            records, next_offset = self.client.scroll( 
                #Here we define what we want the Qdrant to collect
                collection_name=self.collection_name,
                with_payload=True, #The Payload is used to store the metadata of the document to provide more context(useable information) to the LLM
                limit=256, #Fetches up to 256 records
                offset=next_offset, #our pagination
            )
            #Looping through all the records we retrieved earlier
            for record in records:
                #Safely extracts the dictionary from the payload(under the metadata key)
                metadata = (record.payload or {}).get("metadata") or {}
                name = metadata.get("doc_name")
                #Checking if the name is valind and non-empty
                if name:
                    seen.add(name) # We then add it to the set of seen names
            if next_offset is None: # if were dne going through all the records we stop the while loop
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

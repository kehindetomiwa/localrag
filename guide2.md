I’ll anchor this on the open `PROJECT_GUIDE.md`, then trace only the code paths it points to so I can give you a practical learning order instead of an overwhelming file dump.

Read 

Read , lines 1 to 240

Read , lines 1 to 220

Read 

The guide confirms this is a small, coherent pipeline: ingestion and retrieval are the two main stories, with MCP as the boundary to Claude Code. I’m now reading those modules and the tests to separate “must understand line by line” from library plumbing you can learn by contract.

Read 

Read 

Read 

Read 

Read 

One small path detail surfaced: retrieval lives under `retrieval`, while the guide’s diagram omits that prefix. The core ingestion code is straightforward; next I’m checking the vector-store adapter, retrieval scoring, CLI, and tests so the study order follows actual execution.

Read 

Read 

Read 

Read 

Read 

Read 

Read 

Read 

Yes. You do **not** need to understand every dependency line by line. Learn this project in execution order.

**1. Start with the tests**

Read:

- `test_loader.py`
- `test_chunker.py`

These show the expected behavior more clearly than the implementation.

**2. Learn document loading**

Read `loader.py` line by line.

Understand:

- `Path`
- recursive file discovery with `rglob`
- file extension filtering
- relative paths
- metadata
- LangChain `Document`
- exceptions and skipped files

**3. Learn chunking**

Read `chunker.py`.

Understand:

- Markdown heading splitting
- token-based splitting
- overlap
- why metadata such as `section_title` and `chunk_index` is added
- list comprehensions and nested loops

**4. Learn the ingestion pipeline**

Read `pipeline.py`.

This is the central path:

```text
load documents
→ split into chunks
→ copy metadata
→ create stable IDs
→ save to Qdrant
```

Pay special attention to `_stable_id()`. It makes repeated ingestion update existing chunks instead of creating duplicates.

**5. Learn configuration and embeddings**

Read:

- `config.py`
- `embeddings.py`

Understand:

- environment variables
- default values
- dataclasses
- optional values such as API keys
- why embeddings convert text into vectors

You do not need to understand OpenAI’s embedding implementation internally. Understand its input and output contract.

**6. Learn the Qdrant wrapper**

Read `vectorstore.py`.

Understand:

- collections
- vector dimensions
- cosine similarity
- payload metadata
- filters for `doc_name` and `doc_group`
- `add_documents`
- similarity search
- pagination through `scroll`

This file connects your Python code to the database.

**7. Learn retrieval**

Read these in this order:

1. `query_analyzer.py`
2. `reranker.py`
3. `retriever.py`

The retrieval flow is:

```text
user query
→ detect document/group names
→ apply metadata filters
→ vector similarity search
→ BM25 keyword scoring
→ combine scores
→ return ranked chunks
```

The most important concepts are:

- vector similarity finds related meaning
- BM25 finds exact words
- the reranker combines both scores
- `70% vector score + 30% BM25 score`

**8. Learn the MCP boundary last**

Read `mcp_server.py`.

Understand:

- `FastMCP`
- decorators such as `@mcp.tool()`
- how Claude Code calls `search_codebase`
- how results are converted into dictionaries
- why this server retrieves information but does not generate answers

**9. Read the command-line entry point**

Finally read `ingest.py`.

It mainly teaches:

- `argparse`
- command-line arguments
- calling `run_ingestion`
- the `if __name__ == "__main__"` pattern
- Python import paths

A good learning route is:

```text
tests/test_loader.py
→ loader.py
→ tests/test_chunker.py
→ chunker.py
→ pipeline.py
→ vectorstore.py
→ query_analyzer.py
→ reranker.py
→ retriever.py
→ mcp_server.py
→ scripts/ingest.py
```

You can initially skip understanding the internals of LangChain, Qdrant, OpenAI, BM25, Docker, and MCP. Learn how this project uses each library first; study the libraries themselves only when you want to modify that part.
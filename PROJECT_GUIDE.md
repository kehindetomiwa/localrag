# LocalRAG Project Guide

## What this project does

LocalRAG makes a folder of documentation searchable by Claude Code.

It does not generate final answers with an LLM. Instead, it:

1. Loads documentation files.
2. Splits them into smaller chunks.
3. Converts each chunk into an embedding using OpenAI.
4. Stores the embeddings and metadata in Qdrant.
5. Searches and ranks relevant chunks.
6. Exposes the search through an MCP server.

Claude Code receives the chunks and uses its own model to reason about them.

## Big-picture architecture

```text
Document folder
    |
    v
scripts/ingest.py
    |
    v
localrag.ingest.pipeline
    |
    +--> loader.py       Find files and assign metadata
    |
    +--> chunker.py      Split files into searchable sections
    |
    +--> embeddings.py   Create OpenAI embeddings
    |
    +--> vectorstore.py  Save chunks in Qdrant

Claude Code
    |
    v
mcp_server.py
    |
    v
retrieval/retriever.py
    |
    +--> query_analyzer.py  Detect document/group hints
    +--> vectorstore.py      Run filtered vector search
    +--> reranker.py          Combine vector and keyword scores
    |
    v
Ranked chunks returned to Claude Code
```

## Important concepts

### Documents

A document is one supported file under the ingestion directory:

- `.md`
- `.mdx`
- `.txt`
- `.rst`

The loader walks all nested folders.

For example:

```text
docs/
  services/
    auth.md
    billing.md
  architecture/
    decisions/
      adr-001.md
```

The metadata becomes:

| File | `doc_name` | `doc_group` |
|---|---|---|
| `services/auth.md` | `services/auth` | `services` |
| `architecture/decisions/adr-001.md` | `architecture/decisions/adr-001` | `decisions` |

`doc_name` identifies one document. `doc_group` identifies its immediate parent folder.

These values are important because many documents may have similarly named sections such as `Setup`, `Config`, or `API`.

### Chunks

A large document is divided into chunks before indexing.

The chunker uses two stages:

1. Split Markdown on headings from `#` through `######`.
2. Split any oversized section into token windows.

The defaults are:

```text
Maximum chunk size: 500 tokens
Overlap:            50 tokens
```

The section title is stored with every chunk so search results can tell Claude which part of a document they came from.

### Embeddings and Qdrant

An embedding is a numeric representation of text. Similar questions and documentation receive similar vectors.

Qdrant stores:

- The vector embedding.
- The chunk text.
- Metadata such as document name, group, section, and source path.

The default embedding model is `text-embedding-3-small` with a dimension of `1536`.

The Qdrant collection name is supplied at runtime. This allows separate projects to use separate collections.

## Ingestion path

Start here when you want to understand or modify indexing.

### `scripts/ingest.py`

This is the command-line entry point:

```powershell
python scripts/ingest.py --path docs --collection my_docs
```

Use `--recreate` for a clean rebuild:

```powershell
python scripts/ingest.py --path docs --collection my_docs --recreate
```

It passes the arguments to `run_ingestion()`.

### `ingest/loader.py`

This module:

- Validates the input directory.
- Recursively finds supported files.
- Skips unsupported, empty, and non-UTF-8 files.
- Creates LangChain `Document` objects.
- Adds `doc_name`, `doc_group`, `doc_type`, and `source_path` metadata.

Change this file when you want to:

- Support another extension.
- Ignore specific directories.
- Change how document names or groups are calculated.
- Extract additional metadata.

### `ingest/chunker.py`

This module creates chunks from a document's text.

Change it when you want to:

- Use different heading rules.
- Change how code blocks are handled.
- Preserve more context around sections.
- Use a different chunking strategy.

The configuration values are in `config.py`:

```text
CHUNK_MAX_TOKENS
CHUNK_OVERLAP_TOKENS
```

### `ingest/pipeline.py`

This is the ingestion coordinator:

```text
load documents
    -> chunk each document
    -> copy document and chunk metadata
    -> create stable IDs
    -> send chunks to Qdrant
```

The stable ID is based on:

```text
(doc_name, chunk_index)
```

This lets repeated ingestion overwrite the same points instead of duplicating them.

When changing chunking or metadata behavior, use `--recreate`. Otherwise, old chunks can remain in Qdrant if the new version creates fewer chunks than the old version.

## Retrieval path

Start here when search results are poor or incorrectly scoped.

### `retrieval/retriever.py`

This coordinates a search:

1. Create a `VectorStore` for the requested collection.
2. Infer document/group filters if the caller did not provide them.
3. Run vector similarity search.
4. Rerank the candidates.
5. Return the best results.

### `retrieval/query_analyzer.py`

This looks for known document names and groups inside the query.

It currently uses simple case-insensitive substring matching. It does not handle:

- Typos.
- Synonyms.
- Fuzzy matches.
- Natural-language intent beyond literal names.

Improve this file if automatic routing needs to become smarter.

### `retrieval/reranker.py`

The reranker combines two signals:

```text
70% vector similarity
30% BM25 keyword similarity
```

Vector search is useful for meaning. BM25 is useful for exact terms such as:

- Function names.
- Configuration keys.
- Class names.
- Error messages.

Adjust `VECTOR_WEIGHT` and `BM25_WEIGHT` when search needs to favor one type of matching.

The BM25 index is temporary. It is created only over the candidates returned by Qdrant; it is not stored as a separate database.

## Qdrant layer

### `vectorstore.py`

This wraps Qdrant and LangChain.

It handles:

- Connecting to Qdrant.
- Creating collections.
- Recreating collections.
- Creating metadata indexes.
- Adding documents.
- Running similarity searches.
- Filtering by document name or group.
- Listing indexed documents and groups.

Current searchable metadata fields are:

```text
metadata.doc_name
metadata.doc_group
metadata.doc_type
```

Add a field here if you introduce another metadata filter. The loader and ingestion pipeline must also populate that metadata first.

## MCP server

### `mcp_server.py`

Claude Code starts this server as a subprocess using standard input/output.

The server exposes three tools:

### `search_codebase`

Searches the collection and returns ranked chunks with:

- Document name.
- Document group.
- Section title.
- Content.
- Source path.
- Score.

It accepts optional `doc_name` and `doc_group` filters.

### `list_documents`

Returns every distinct indexed document name.

### `list_document_groups`

Returns every distinct indexed group.

Add or modify Claude-facing functionality in this file.

## Configuration

`config.py` loads values from environment variables, usually from `.env`.

Important settings:

```text
QDRANT_URL
QDRANT_API_KEY
OPENAI_API_KEY
EMBEDDING_MODEL
EMBEDDING_DIM
CHUNK_MAX_TOKENS
CHUNK_OVERLAP_TOKENS
```

Explicit function or command-line arguments take priority over environment values. Environment values take priority over hard-coded defaults.

## Running the system

### Start Qdrant

```powershell
docker compose up -d
```

Qdrant is normally available at:

```text
http://localhost:6333
```

### Ingest documents

```powershell
python scripts/ingest.py --path path\to\docs --collection my_docs --recreate
```

### Run the MCP server manually

```powershell
python -m localrag.mcp_server --collection my_docs
```

Normally Claude Code starts this process from its MCP configuration.

## Where to tweak first

| Desired change | Primary file |
|---|---|
| Add supported file types | `src/localrag/ingest/loader.py` |
| Ignore files or folders | `src/localrag/ingest/loader.py` |
| Change chunk size | `src/localrag/config.py` |
| Change Markdown section handling | `src/localrag/ingest/chunker.py` |
| Add metadata | `loader.py`, then `pipeline.py` |
| Add metadata filtering | `vectorstore.py` |
| Improve document/group detection | `retrieval/query_analyzer.py` |
| Change ranking behavior | `retrieval/reranker.py` |
| Add an MCP tool | `src/localrag/mcp_server.py` |
| Change the embedding model | `src/localrag/config.py` |

## Tests

The current tests cover:

- File loading.
- Recursive directory traversal.
- Supported extensions.
- Missing directories.
- Markdown heading splitting.
- Oversized section splitting.

They do not currently cover:

- Real OpenAI embedding calls.
- Qdrant integration.
- Reranking quality.
- MCP tool behavior.
- Collection recreation.

The repository's checked-in `env` directory does not currently contain `env\Scripts\python.exe`, so tests cannot be run from that environment until it is rebuilt or replaced with a working virtual environment.

## A practical workflow for changes

1. Decide whether the change affects ingestion or retrieval.
2. Edit the owning module listed above.
3. Add or update a focused test.
4. Re-ingest with `--recreate` if indexing behavior changed.
5. Test a few real searches through the MCP server.
6. Inspect the returned `doc_name`, `doc_group`, section, and source path before judging ranking quality.

The most important rule is that metadata, chunking, embeddings, and retrieval filters must agree. If you change how a document is named or grouped, re-index the collection before evaluating search behavior.

# LocalRAG

A codebase/docs RAG pipeline that tags every chunk with its **document name**
and **document group** (its parent folder) at ingestion time, and filters on
those tags at retrieval time. This is what keeps structurally-similar
documents (e.g. every service's `README.md`, every module's `config.md`)
from getting confused with each other.

Retrieval is exposed to **Claude Code** as a local MCP server (stdio
transport) with retrieval-only tools -- this project does not call an LLM
to generate answers; Claude Code's own model reasons over the returned
chunks.

Built on **LangChain**: documents/chunks are `langchain_core.documents.Document`
objects, chunking uses `langchain-text-splitters` (`MarkdownHeaderTextSplitter`
+ `TokenTextSplitter`), embeddings go through `langchain-openai`
(`OpenAIEmbeddings`), the vector store is `langchain-qdrant`
(`QdrantVectorStore`), and the lexical half of the hybrid rerank uses
`langchain-community`'s `BM25Retriever`.

## Project layout

```
localrag/
├── pyproject.toml
├── docker-compose.yml          # local Qdrant
├── .env.example
├── claude_code_mcp.example.json
├── src/localrag/
│   ├── config.py                # env-driven settings
│   ├── embeddings.py             # langchain-openai OpenAIEmbeddings factory
│   ├── vectorstore.py            # langchain-qdrant QdrantVectorStore wrapper (filterable by doc_name/doc_group)
│   ├── mcp_server.py              # MCP server -- entrypoint for Claude Code
│   ├── ingest/
│   │   ├── loader.py              # recursively walks a folder (any nesting depth), derives doc_name/doc_group, returns Documents
│   │   ├── chunker.py             # MarkdownHeaderTextSplitter + TokenTextSplitter
│   │   └── pipeline.py            # load -> chunk -> embed -> upsert
│   └── retrieval/
│       ├── query_analyzer.py      # detects doc_name/doc_group hints in a query
│       ├── reranker.py            # blends vector score + langchain BM25Retriever score
│       └── retriever.py           # ties the above together
├── scripts/ingest.py              # CLI: python scripts/ingest.py --path ... --collection ...
└── tests/
```

## LocalRAG Setup and Claude Code MCP Configuration

This guide explains how to run LocalRAG from a clean terminal session and
configure it as an MCP server in Claude Code.

### 1. Navigate to the project

```bash
cd localrag
```

### 2. Activate the virtual environment

A `.venv` virtual environment using Python 3.14, with all dependencies
already installed, exists in this project from an earlier setup.

Activate it:

```bash
source .venv/bin/activate
```

Confirm that it is using the expected Python version:

```bash
python --version
```

The command should print `Python 3.14.7`.

If `.venv` is missing or broken, rebuild it:

```bash
/usr/local/bin/python3.14 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 3. Set the OpenAI API key

A `.env` file already exists and was created from `.env.example`. Open it:

```bash
open -e .env
```

Make sure `OPENAI_API_KEY` contains your real API key rather than the
`sk-...` placeholder.

Leave the other defaults -- such as `QDRANT_URL`, `EMBEDDING_MODEL`, and
the chunk-size settings -- unchanged unless you have a reason to modify
them.

### 4. Start Qdrant

Start the Qdrant vector database in the background:

```bash
docker compose up -d
```

Verify that it is running:

```bash
docker compose ps
curl -s http://localhost:6333/collections
```

On the first run, the response should resemble:

```json
{"result":{"collections":[]},"status":"ok"}
```

### 5. Run the test suite (optional)

```bash
pytest -v
```

The tests do not require Qdrant or an API key. They exercise only the
loader and chunker logic. `test_chunker.py` uses `tiktoken`, which
downloads its encoding file on first use if it isn't already cached --
that first run needs network access.

### 6. Ingest a documentation folder

```bash
python scripts/ingest.py \
  --path docs \
  --collection edu365_docs
```

- `--path` accepts any folder and recursively processes its nested
  subfolders.
- `--collection` sets the Qdrant collection name. Use a distinct name for
  each project.
- Add `--recreate` to drop and rebuild the collection from scratch instead
  of performing an incremental upsert.

Re-running ingestion on the same folder/collection is safe: chunk IDs are
derived deterministically from `doc_name` + `chunk_index`, so re-ingesting
a changed file overwrites its old chunks instead of duplicating them.

### 7. Check that the MCP server starts

```bash
python -m localrag.mcp_server --collection edu365_docs
```

The process should remain active and listen on standard input/output
without reporting errors. This indicates that the server is working.

Press `Ctrl+C` to stop it. Claude Code will start and manage the server
after it is configured.

### 8. Configure Claude Code

Add one MCP server entry for each collection that you want to make
searchable. Merge the following entry into your existing configuration
rather than overwriting the entire file:

```json
{
  "mcpServers": {
    "localrag-edu365": {
      "command": ".venv/bin/python",
      "args": [
        "-m",
        "localrag.mcp_server",
        "--collection",
        "edu365_docs"
      ]
    }
  }
}
```

The repository's `claude_code_mcp.example.json` file contains the same
template.

### 9. Restart Claude Code

After restarting Claude Code, the following tools should be available for
the configured collection:

- `search_codebase(query, doc_name?, doc_group?, top_n?)` -- ranked, labeled chunks
- `list_documents()` -- every `doc_name` currently indexed
- `list_document_groups()` -- every `doc_group` currently indexed

### Adding another project later

To ingest another project, repeat steps 6 and 8 with a different `--path`
and `--collection` value. The virtual environment, Qdrant instance, and
`.env` file remain shared infrastructure.

### Where Claude Code stores its MCP configuration

Claude Code does not use only one dedicated MCP configuration file. MCP
server entries can be stored using several scopes.

**Option A: Add the server with the CLI (recommended)**

This avoids manually editing a large JSON file:

```bash
cd localrag

claude mcp add localrag-edu365 -- \
  localrag/.venv/bin/python \
  -m localrag.mcp_server \
  --collection edu365_docs
```

Run the command from the project directory whose Claude Code session
should have access to this server -- that's what determines which project
the entry gets scoped to.

Confirm the configured MCP servers afterward:

```bash
claude mcp list
```

**Option B: Edit `~/.claude.json` manually**

Project-scoped MCP servers are stored under:

```
~/.claude.json
  -> "projects"
  -> "/absolute/path/to/the/project"
  -> "mcpServers"
```

Add the LocalRAG server entry to that `mcpServers` object.

The same file may also contain a top-level `mcpServers` object for
globally available servers. A global entry is available in every project,
so project scope is usually a better fit when a server is tied to a
specific collection.

**Caution:** `~/.claude.json` may contain account information and settings
for other projects. Back it up before editing, and make sure the
resulting file remains valid JSON.

**Option C: Create a project-level `.mcp.json`**

You can place a `.mcp.json` file directly in a project's root, e.g.:

```
localrag/.mcp.json
```

Claude Code can detect this file and offer to enable its MCP
configuration for the project. This approach is useful when the
configuration should be shared through version control.

You can create it from this repository's `claude_code_mcp.example.json`
template.

## Extending later

- Swap the naive substring `query_analyzer` for an LLM-based classifier if you need fuzzy matching (typos, synonyms) between a query and a `doc_name`.
- Swap the BM25 rerank in `reranker.py` for a hosted cross-encoder (e.g. Cohere Rerank) if lexical blending isn't precise enough.
- Add a `tree-sitter`-based chunker alongside the markdown one if you start ingesting source code files directly, not just docs.

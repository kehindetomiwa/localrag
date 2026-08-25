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

## Project layout

```
localrag/
├── pyproject.toml
├── docker-compose.yml          # local Qdrant
├── .env.example
├── claude_code_mcp.example.json
├── src/localrag/
│   ├── config.py                # env-driven settings
│   ├── embeddings.py             # OpenAI embedding client
│   ├── vectorstore.py            # Qdrant wrapper (filterable by doc_name/doc_group)
│   ├── mcp_server.py              # MCP server -- entrypoint for Claude Code
│   ├── ingest/
│   │   ├── loader.py              # walks a folder, derives doc_name/doc_group
│   │   ├── chunker.py             # header-aware chunking + token windows
│   │   └── pipeline.py            # load -> chunk -> embed -> upsert
│   └── retrieval/
│       ├── query_analyzer.py      # detects doc_name/doc_group hints in a query
│       ├── reranker.py            # blends vector + BM25 score
│       └── retriever.py           # ties the above together
├── scripts/ingest.py              # CLI: python scripts/ingest.py --path ... --collection ...
└── tests/
```

## Setup

```bash
cd localrag
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# edit .env and set OPENAI_API_KEY

docker compose up -d   # starts Qdrant on localhost:6333
```

## Ingest a docs folder

Both the source path and the collection ("DB") name are **required, dynamic
arguments** -- nothing is hard-coded, so you can point this at any project:

```bash
python scripts/ingest.py \
  --path /Users/kehindetomiwa/Documents/skttechnologies/projects/edu365/edu365/docs \
  --collection edu365_docs
```

Add `--recreate` to drop and rebuild the collection from scratch (useful
after changing the chunking strategy or re-running on updated docs).

Re-running ingestion on the same folder/collection is safe: chunk IDs are
derived deterministically from `doc_name` + `chunk_index`, so re-ingesting
a changed file overwrites its old chunks instead of duplicating them.

## Run the MCP server for Claude Code

The collection name is also a required, dynamic argument here -- one server
instance always serves exactly one collection:

```bash
python -m localrag.mcp_server --collection edu365_docs
```

To wire this into Claude Code, add an entry like `claude_code_mcp.example.json`
to your Claude Code MCP config, pointing `command` at this project's venv
Python. Run one server (with its own `--collection`) per project you want
Claude Code to search.

Exposed tools:
- `search_codebase(query, doc_name?, doc_group?, top_n?)` -- ranked, labeled chunks
- `list_documents()` -- every `doc_name` currently indexed
- `list_document_groups()` -- every `doc_group` currently indexed

## Tests

```bash
pytest
```

`test_chunker.py` uses `tiktoken`, which downloads its encoding file on
first use if it isn't already cached -- that first run needs network
access.

## Extending later

- Swap the naive substring `query_analyzer` for an LLM-based classifier if you need fuzzy matching (typos, synonyms) between a query and a `doc_name`.
- Swap the BM25 rerank in `reranker.py` for a hosted cross-encoder (e.g. Cohere Rerank) if lexical blending isn't precise enough.
- Add a `tree-sitter`-based chunker alongside the markdown one if you start ingesting source code files directly, not just docs.

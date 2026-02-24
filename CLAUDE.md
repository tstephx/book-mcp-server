# CLAUDE.md — book-mcp-server
<!-- project-name: book-mcp-server -->

**DO NOT scan directories on startup.** This project is well-documented below.

## What This Is
Two MCP servers + a CLI pipeline in one repo:
- **Book library** — read-only search/read/learning tools for Claude Desktop
- **Agentic pipeline** — book processing, approval, autonomy management
- **CLI** — human operator commands (`agentic-pipeline`)

## Canonical Entry Points (do not guess)

| Component | Entry point | What it does |
|-----------|-------------|-------------|
| **Book library MCP server** | `server.py` → `src/server.py` | FastMCP stdio server for Claude Desktop. Search, read, learning tools. |
| **Agentic pipeline MCP server** | `agentic_mcp_server.py` → `agentic_pipeline/mcp_server.py` | Pipeline approval, health, autonomy tools. |
| **CLI (humans)** | `agentic-pipeline` → `agentic_pipeline/cli.py` | Click CLI: init, health, worker, approve, escape-hatch. |
| **MCP tool definitions** | `agentic_pipeline/mcp_server.py` (pipeline), `src/tools/*.py` (library) | Where to add/edit tools. |

## Environment Variables

| Component | DB env var | Books dir env var | Notes |
|-----------|-----------|-------------------|-------|
| Pipeline (agentic-pipeline) | `AGENTIC_PIPELINE_DB` | `WATCH_DIR`, `PROCESSED_DIR` | Worker + state machine |
| MCP server (Claude Desktop) | `BOOK_DB_PATH` | `BOOKS_DIR` | Read/search tools |
| Both | — | — | Shared DB: `~/_Projects/book-ingestion-python/data/library.db` |
| Embeddings | `OPENAI_API_KEY` | — | Required for semantic search + pipeline embedding |

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Run book-library MCP server (stdio, for Claude Desktop)
python server.py

# Run agentic-pipeline MCP server
python agentic_mcp_server.py

# CLI: initialize database
agentic-pipeline init

# CLI: check health
agentic-pipeline health

# CLI: run worker with directory watching + auto-archive
agentic-pipeline worker --watch-dir /path/to/books/ --processed-dir /path/to/books/processed

# Tests (use python -m pytest, not bare pytest)
python -m pytest tests/ -v
```

## Key Concepts

### Pipeline States
Books flow through: `DETECTED` → `HASHING` → `CLASSIFYING` → `SELECTING_STRATEGY` → `PROCESSING` → `VALIDATING` → `PENDING_APPROVAL` → `APPROVED` → `EMBEDDING` → `COMPLETE`

High-confidence books (≥0.7, no review needed) skip `PENDING_APPROVAL` and auto-approve. Approval runs embedding inline — no separate worker needed.

### Autonomy Modes
- **supervised** — All books require human approval (default)
- **partial** — Auto-approve high-confidence known types
- **confident** — Per-type calibrated thresholds

### File Watcher
```bash
agentic-pipeline worker --watch-dir /path/to/books/
```
Scans run as lowest-priority step in the poll loop. Deduplication via content hash — dropping the same file twice is a no-op.

### Auto-Archive
```bash
agentic-pipeline worker --watch-dir /path/to/books/ --processed-dir /path/to/books/processed
```
Files in `processed_dir` are excluded from watch scans. Name collisions handled with counter suffixes. Archive failures logged but don't affect pipeline state.

### Escape Hatch
```bash
agentic-pipeline escape-hatch "reason"
```
One command reverts to fully supervised mode.

## Tuning Knobs

| Knob | File | Symbol | Env override | Default |
|------|------|--------|-------------|---------|
| Auto-approve threshold | `agentic_pipeline/config.py` | `confidence_threshold` | `CONFIDENCE_THRESHOLD` | 0.7 |
| Processing timeout | `agentic_pipeline/config.py` | `processing_timeout` | `PROCESSING_TIMEOUT_SECONDS` | 600s |
| Embedding timeout | `agentic_pipeline/config.py` | `embedding_timeout` | `EMBEDDING_TIMEOUT_SECONDS` | 300s |
| Worker poll interval | `agentic_pipeline/config.py` | `worker_poll_interval` | `WORKER_POLL_INTERVAL_SECONDS` | 5s |
| Max retries | `agentic_pipeline/config.py` | `max_retry_attempts` | `MAX_RETRY_ATTEMPTS` | 3 |

## If Something Breaks

| Symptom | Check |
|---------|-------|
| Pipeline stuck | `agentic-pipeline health` + stuck detector output |
| Watcher not picking files | Confirm `WATCH_DIR` + file extension (.epub/.pdf) + `PROCESSED_DIR` exclusion |
| Claude Desktop not seeing new books | Confirm embeddings generated, server restarted, `BOOK_DB_PATH` matches |
| Embedding failures | Check `OPENAI_API_KEY` is set in environment |
| DB locked | Only one writer at a time; check for zombie worker processes |

## Common Tasks

### Adding a New Feature
1. Write tests first in `tests/`
2. Implement in appropriate module
3. Add CLI command if user-facing
4. Add MCP tool if Claude should use it
5. Run `python -m pytest tests/ -v`

### Database Changes
1. Add migration to `agentic_pipeline/db/migrations.py` in `MIGRATIONS` list
2. Write test in `tests/test_*_migrations.py`
3. Existing DBs auto-migrate on `run_migrations()`

### Adding CLI Commands
Commands in `agentic_pipeline/cli.py` using Click.

### Adding MCP Tools
Pipeline tools in `agentic_pipeline/mcp_server.py`. Library tools in `src/tools/*.py`.

## Testing

```bash
python -m pytest tests/ -v                          # All tests
python -m pytest tests/test_phase5*.py -v            # Specific phase
python -m pytest tests/ --cov=agentic_pipeline       # With coverage
```

## Embeddings

**Model:** OpenAI `text-embedding-3-small` (1536 dims). Requires `OPENAI_API_KEY`.

**How embeddings are generated:**
- **Pipeline path** — `approve_book()` runs inline: `APPROVED → EMBEDDING → COMPLETE`. No separate worker.
- **Manual refresh** — Ask Claude `"refresh embeddings"` (calls `refresh_embeddings` MCP tool), or run `python -m pytest tests/test_openai_embeddings.py` to verify.
- **Summary embeddings** — separate from chapter embeddings; call `generate_summary_embeddings` MCP tool.

**Where they live:** `chapters.embedding` column (BLOB, numpy float32 array). Indexed via cosine similarity at query time — no separate vector store.

**Troubleshooting:**
- Semantic search returns nothing → check `OPENAI_API_KEY` is set; run `refresh_embeddings`
- New book not searchable → confirm pipeline reached `COMPLETE` state; embeddings generated at approval time

## Architecture Decisions
1. **SQLite + WAL mode** — all `agentic_pipeline/` connections via `get_pipeline_db()` (timeout=10, row_factory=sqlite3.Row)
2. **Inline embedding** — `approve_book()` runs full APPROVED → EMBEDDING → COMPLETE flow
3. **ProcessingAdapter** — wraps `book-ingestion` as library (lazy-imported)
4. **Hybrid Search** — RRF combines FTS5 keyword + semantic vector; optional MMR for diversity

## Book Library Tools (Claude Desktop)

| Category | Tools |
|----------|-------|
| **Search** | `semantic_search`, `text_search`, `hybrid_search`, `search_all_books` |
| **Discovery** | `get_topic_coverage`, `find_related_content`, `extract_code_examples` |
| **Reading** | `get_chapter`, `get_book_info`, `list_books` |
| **Learning** | `teach_concept`, `generate_learning_path`, `create_study_guide` |
| **Planning** | `generate_project_learning_path`, `create_implementation_plan` |
| **Progress** | `mark_as_read`, `add_bookmark`, `get_reading_progress`, `get_bookmarks` |
| **Export** | `export_chapter_to_markdown` |

Example: "Search my books for Kubernetes content". See `docs/USER-GUIDE.md` for full usage.

## MCP Client Config

Add to `.mcp.json` in any project that needs the book library:

```json
{
  "mcpServers": {
    "book-library": {
      "command": "/Users/taylorstephens/_Projects/book-mcp-server/.venv/bin/python",
      "args": ["/Users/taylorstephens/_Projects/book-mcp-server/server.py"],
      "env": {
        "BOOK_DB_PATH": "/Users/taylorstephens/_Projects/book-ingestion-python/data/library.db",
        "BOOKS_DIR": "/Users/taylorstephens/_Projects/book-ingestion-python/data/books",
        "OPENAI_API_KEY": "${OPENAI_API_KEY}"
      }
    }
  }
}
```

**Critical:** Command must use the **venv python** (has mcp, fastmcp, openai deps). Env vars are `BOOK_DB_PATH` and `BOOKS_DIR` (NOT `BOOK_LIBRARY_DB`/`LIBRARY_PATH`). Do NOT use `uv`.

---

*Last updated: 2026-02-23*

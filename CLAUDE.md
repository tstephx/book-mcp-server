---
status: active
tags: [project/book-mcp-server, format/readme]
type: project
created: '2026-03-05'
modified: '2026-03-05'
related: ["[[Claude-Config/mcp-servers/book-library]]", "[[Claude-Config/mcp-servers/agentic-pipeline]]"]
---

# CLAUDE.md — book-mcp-server
<!-- project-name: book-mcp-server -->

**DO NOT scan directories on startup.** This project is well-documented below.

## Reference Documentation (`ref/`)

| File | Contents |
|------|----------|
| [`ref/pipeline-architecture.md`](ref/pipeline-architecture.md) | State machine, orchestrator API, approval flow, autonomy modes, config knobs |
| [`ref/mcp-tools.md`](ref/mcp-tools.md) | All MCP tools for both servers with signatures and descriptions |
| [`ref/db-schema.md`](ref/db-schema.md) | All pipeline DB tables and columns |
| [`ref/module-map.md`](ref/module-map.md) | "Which file handles X?" — responsibility of every module in `agentic_pipeline/` |
| [`ref/cli-commands.md`](ref/cli-commands.md) | All 31 CLI commands grouped by category with args and descriptions |

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
| **CLI (humans)** | `agentic-pipeline` → `agentic_pipeline/cli.py` | Click CLI: 31 commands for pipeline management, approval, autonomy, library maintenance (incl. `rechunk`). See `ref/cli-commands.md`. |
| **MCP tool definitions** | `agentic_pipeline/mcp_server.py` (pipeline), `src/tools/*.py` (library) | Where to add/edit tools. |

## Environment Variables

| Component | DB env var | Books dir env var | Notes |
|-----------|-----------|-------------------|-------|
| Pipeline (agentic-pipeline) | `AGENTIC_PIPELINE_DB` | `WATCH_DIR`, `PROCESSED_DIR` | Worker + state machine |
| MCP server (Claude Desktop) | `BOOK_DB_PATH` | `BOOKS_DIR` | Read/search tools |
| Both | — | — | Shared DB: `~/Library/Application Support/book-library/library.db` |
| Embeddings | `OPENAI_API_KEY` | — | Required for semantic search + pipeline embedding (`text-embedding-3-large`) |
| Classifier | `CLASSIFIER_PROVIDER` | — | `claude-code` (default primary, `claude -p`) or `openai`; set to force single-provider mode |

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
agentic-pipeline worker --watch-dir ~/Documents/_ebooks/agentic-book-pipeline --processed-dir ~/Documents/_ebooks/agentic-book-pipeline/processed

# CLI: full command reference — see ref/cli-commands.md

# Tests — use Makefile targets (venv must be activated)
source .venv/bin/activate
make test          # unit tests (default)
make test-fast     # parallel, ~2x faster
make test-cov      # with HTML coverage report → htmlcov/index.html
make test-integration  # requires real DB + OPENAI_API_KEY
# Or directly: python -m pytest tests/ -v
```

## Key Concepts

### Pipeline States
Pipeline states and auto-approval guardrails — see `ref/pipeline-architecture.md`.

### Classifier Providers
Provider chain and fallback — see `ref/module-map.md:48`. The launchd worker needs `claude` on its PATH — `scripts/run-worker.sh` exports `~/.local/bin`.

### Chunking & Re-chunking
Chunking, re-chunking, and the eval gate — see `ref/module-map.md`, `ref/cli-commands.md` (`rechunk`), and `ref/db-schema.md` (`chunks` table).

### Autonomy Modes
Autonomy modes, thresholds, and guardrails — see `ref/pipeline-architecture.md`.

### Watcher & Auto-Archive
Watcher and auto-archive behavior — see `ref/pipeline-architecture.md`.

### Book ID Resolution
Book ID resolution (partial-slug matching) — see `ref/mcp-tools.md`.

### Escape Hatch
```bash
agentic-pipeline escape-hatch "reason"
```
One command reverts to fully supervised mode.

## Tuning Knobs

All tunable knobs (file, symbol, env override, default) — see `ref/pipeline-architecture.md`'s Configuration table.

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

### Database Changes, CLI Commands, MCP Tools
Adding a migration/CLI command/MCP tool — use the `new-migration`/`new-cli-command`/`new-mcp-tool` skills.

### Architecture Decisions
New architecture decision or design doc — use the `new-adr`/`design-doc` skills.

## Testing

```bash
python -m pytest tests/ -v                          # All tests
python -m pytest tests/test_phase5*.py -v            # Specific phase
python -m pytest tests/ --cov=agentic_pipeline       # With coverage
```

Manual QA checklist — see `docs/MANUAL-TEST-PLAN.md`.

**Pre-push gate.** Hosted CI (`.github/workflows/ci.yml`) is dormant:
GitHub Actions is billing-blocked account-wide (taylor-dev-core issue
#81) and the owner decided not to restore it. The workflow file stays in
place and resumes automatically if billing is ever fixed. Local
replacement: a tracked `scripts/pre-push-verify.sh` git hook, installed
per clone (`.git/hooks` is untracked) via
`ln -sf ../../scripts/pre-push-verify.sh .git/hooks/pre-push` from the
repository root, runs `uv sync --locked` then `make lint && make test`
before any `git push` and blocks it on failure. Emergency bypass:
`git push --no-verify`.

## Embeddings

**Model:** OpenAI `text-embedding-3-large` (3072 dims). Requires `OPENAI_API_KEY`.

**Manual refresh** — Ask Claude `"refresh embeddings"` (calls `refresh_embeddings` MCP tool).

Storage and cache-invalidation detail — see `ref/db-schema.md`'s Embedding Table section.

## Architecture Decisions
1. **SQLite + WAL mode** — all `agentic_pipeline/` connections via `get_pipeline_db()` (timeout=10, row_factory=sqlite3.Row). Both this connection and the MCP-tool-side `src/database.py`'s `get_db_connection()` set `PRAGMA foreign_keys = ON` (PR #10, 2026-08-14) — `ON DELETE CASCADE` now fires consistently on either path. See `ref/db-schema.md` / the `db-schema` skill before touching delete/cascade behavior.
2. **Inline embedding** — `approve_book()` runs full APPROVED → EMBEDDING → COMPLETE flow
3. **ProcessingAdapter** — wraps `book-ingestion` as library (lazy-imported)
4. **Hybrid Search** — RRF combines FTS5 keyword + semantic vector; optional MMR for diversity
5. **Provider abstraction** — classifier default is `claude-code` (subscription billing) with OpenAI fallback; `CLASSIFIER_PROVIDER` forces single-provider mode
6. **Hierarchical chunker** — paragraph packing with sentence-window fallback for wall-of-text; re-chunk is staged + retrieval-eval-gated (`rechunk`/`--swap`), `library_meta.data_version` drives cache coherence
7. **Extraction validation** — `count_source_words` (EPUB via zip, PDF via PyMuPDF) feeds Check 8 source-coverage; scanned PDFs (<100 words) skip the check

## Book Library Tools (Claude Desktop)

Example: "Search my books for Kubernetes content". See `ref/mcp-tools.md` for the full tool index by category, `docs/USER-GUIDE.md` for usage, and `docs/PROJECT-PLANNING-QUICKREF.md`, `docs/PROJECT-PLANNING-TOOLS.md`, `docs/PROJECT-LEARNING-TOOLS.md` for Planning/Learning tool usage guides.

## MCP Client Config

First-time setup and Claude Desktop / other-project `.mcp.json` configuration for the book library — see `docs/QUICKSTART.md`.

### `agentic-pipeline` is a shared MCP provider

Before touching `pyproject.toml`'s `[tool.uv] constraint-dependencies`, or when a new project wants to consume `agentic-pipeline` as an MCP server — read `docs/shared-mcp-provider.md`.

---

*Last updated: 2026-08-14*

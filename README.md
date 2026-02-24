# book-mcp-server

Two MCP servers and a CLI pipeline for building and using a personal AI-searchable book library.

---

## What's In This Repo

### 1. Book Library MCP Server (`server.py`)
A read-only MCP server for Claude Desktop. Drop it in your Claude config and ask Claude to search, read, and learn from your book library.

```
"Search my books for Kubernetes content"
"Create a learning path for distributed systems"
"Teach me the concept of eventual consistency from my library"
```

### 2. Agentic Processing Pipeline (`agentic_mcp_server.py` + CLI)
An AI-powered pipeline that processes new EPUBs/PDFs into the library automatically:
1. **Detects** new files and deduplicates by content hash
2. **Classifies** the book type (tutorial, textbook, magazine, etc.) using AI
3. **Selects** the optimal processing strategy for that book type
4. **Processes** the book through the ingestion pipeline
5. **Validates** extraction quality
6. **Queues** for approval (or auto-approves high-confidence books)
7. **Embeds** approved books for semantic search

---

## Quick Start

### Book Library (Claude Desktop)

Add to your Claude Desktop MCP config:

```json
{
  "mcpServers": {
    "book-library": {
      "command": "/path/to/book-mcp-server/.venv/bin/python",
      "args": ["/path/to/book-mcp-server/server.py"],
      "env": {
        "BOOK_DB_PATH": "/path/to/library.db",
        "BOOKS_DIR": "/path/to/books",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

**Critical:** Use the `.venv/bin/python` path — it has all required dependencies. `OPENAI_API_KEY` is required for semantic search.

### Agentic Pipeline (CLI)

```bash
# Install
pip install -e .

# Initialize database
agentic-pipeline init

# Start the file watcher
agentic-pipeline worker --watch-dir /path/to/ebooks --processed-dir /path/to/ebooks/processed

# Check health
agentic-pipeline health

# Review pending books
agentic-pipeline pending

# Approve a book
agentic-pipeline approve <pipeline-id>
```

---

## Pipeline States

Books flow through:

```
DETECTED → HASHING → CLASSIFYING → SELECTING_STRATEGY → PROCESSING
       → VALIDATING → PENDING_APPROVAL → APPROVED → EMBEDDING → COMPLETE
```

High-confidence books (≥0.7, no issues) skip `PENDING_APPROVAL` and auto-approve.

---

## Autonomy Modes

| Mode | Behavior |
|------|----------|
| **supervised** | All books require human approval (default) |
| **partial** | Auto-approve high-confidence known types |
| **confident** | Per-type calibrated thresholds |

```bash
# Instantly revert to fully supervised mode
agentic-pipeline escape-hatch "Something seems off"
```

---

## Common CLI Commands

```bash
agentic-pipeline pending                              # Books awaiting review
agentic-pipeline approve <id>                         # Approve a book
agentic-pipeline reject <id> --reason "Poor quality"  # Reject with reason
agentic-pipeline batch-approve --min-confidence 0.9 --execute
agentic-pipeline health                               # Pipeline health
agentic-pipeline stuck                                # Find stuck pipelines
agentic-pipeline audit --last 20                      # Audit trail
```

---

## Project Structure

```
book-mcp-server/
├── server.py                    # Book library MCP server (entry point)
├── agentic_mcp_server.py        # Pipeline MCP server (entry point)
├── src/                         # Book library source
│   ├── server.py                # FastMCP server implementation
│   ├── tools/                   # MCP tool definitions
│   └── utils/                   # Search, embeddings, caching
├── agentic_pipeline/            # Pipeline source
│   ├── cli.py                   # CLI commands
│   ├── mcp_server.py            # Pipeline MCP tools
│   ├── agents/                  # AI classifier + providers
│   ├── pipeline/                # State machine
│   ├── approval/                # Approval queue
│   ├── autonomy/                # Trust calibration
│   ├── db/                      # Database + migrations
│   └── health/                  # Health monitor + stuck detection
├── config/strategies/           # Processing configs per book type
├── tests/                       # Test suite
└── docs/                        # Documentation
```

---

## Environment Variables

| Variable | Component | Purpose |
|----------|-----------|---------|
| `BOOK_DB_PATH` | Library server | Path to SQLite library database |
| `BOOKS_DIR` | Library server | Path to book files directory |
| `OPENAI_API_KEY` | Both | Required for semantic search + pipeline embedding |
| `AGENTIC_PIPELINE_DB` | Pipeline | Override pipeline database path |
| `WATCH_DIR` | Pipeline worker | Directory to watch for new books |
| `PROCESSED_DIR` | Pipeline worker | Archive dir for processed books |

---

## Documentation

- [CLAUDE.md](./CLAUDE.md) — Full reference for contributors (entry points, commands, architecture)
- [docs/USER-GUIDE.md](./docs/USER-GUIDE.md) — Book library tool reference
- [docs/MANUAL-TEST-PLAN.md](./docs/MANUAL-TEST-PLAN.md) — Manual test procedures
- [DESIGN.md](./DESIGN.md) — Architecture decisions and rationale

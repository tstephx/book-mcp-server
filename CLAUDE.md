# Claude Project Context
<!-- project-name: book-mcp-server -->

**DO NOT scan directories on startup.** This project is well-documented below.

---

This is the **Agentic Book Processing Pipeline** - an AI-powered system that automatically processes, classifies, and ingests books into a searchable knowledge library.

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Run tests (use python -m pytest, not bare pytest)
python -m pytest tests/ -v

# Initialize database
agentic-pipeline init

# Check pipeline health
agentic-pipeline health

# Check autonomy status
agentic-pipeline autonomy status

# Run worker with directory watching
agentic-pipeline worker --watch-dir /path/to/books/
```

## Project Structure

```
agentic_pipeline/
├── agents/                 # AI-powered components
│   ├── classifier.py       # Book type classification
│   ├── validator.py        # Quality validation
│   └── providers/          # LLM providers (OpenAI, Anthropic)
├── adapters/               # External system adapters
│   ├── processing_adapter.py  # Wraps book-ingestion library
│   └── llm_fallback_adapter.py # LLM fallback for low confidence
├── pipeline/               # State machine & orchestration
│   ├── states.py           # Pipeline states enum
│   ├── strategy.py         # Strategy selection
│   └── transitions.py      # State transitions
├── approval/               # Approval queue & actions
│   ├── queue.py            # ApprovalQueue
│   └── actions.py          # approve/reject/rollback + inline embedding
├── autonomy/               # Phase 5: Graduated trust
│   ├── config.py           # AutonomyConfig (modes, escape hatch)
│   ├── metrics.py          # MetricsCollector
│   ├── calibration.py      # CalibrationEngine (thresholds)
│   └── spot_check.py       # SpotCheckManager
├── health/                 # Phase 4: Production monitoring
│   ├── monitor.py          # HealthMonitor
│   └── stuck_detector.py   # StuckDetector
├── batch/                  # Phase 4: Bulk operations
│   ├── filters.py          # BatchFilter
│   └── operations.py       # BatchOperations
├── audit/                  # Phase 4: Audit trail
│   └── trail.py            # AuditTrail
├── library/                # Library status dashboard
│   └── status.py           # LibraryStatus
├── db/                     # Database layer
│   ├── migrations.py       # Schema definitions (WAL mode enabled)
│   ├── pipelines.py        # PipelineRepository
│   └── config.py           # DB path configuration
├── orchestrator/           # Main orchestrator (package)
│   └── orchestrator.py     # Orchestrator class
├── cli.py                  # Click CLI commands
├── mcp_server.py           # MCP tools for Claude
└── config.py               # OrchestratorConfig
```

## Key Concepts

### Pipeline States
Books flow through: `DETECTED` → `HASHING` → `CLASSIFYING` → `SELECTING_STRATEGY` → `PROCESSING` → `VALIDATING` → `PENDING_APPROVAL` → `APPROVED` → `EMBEDDING` → `COMPLETE`

High-confidence books (≥0.7, no review needed) skip `PENDING_APPROVAL` and auto-approve. Approval (single or batch) runs embedding inline — no separate worker needed.

### Autonomy Modes
- **supervised** - All books require human approval (default)
- **partial** - Auto-approve high-confidence known types
- **confident** - Per-type calibrated thresholds

### File Watcher
The worker can watch a directory for new `.epub`/`.pdf` files and automatically queue them:
```bash
agentic-pipeline worker --watch-dir /path/to/books/
# Or via env var: WATCH_DIR=/path/to/books agentic-pipeline worker
```
Scans run as the lowest-priority step in the poll loop. Deduplication via content hash — dropping the same file twice is a no-op.

### Escape Hatch
One command reverts to fully supervised mode:
```bash
agentic-pipeline escape-hatch "reason"
```

## Common Tasks

### Adding a New Feature
1. Write tests first in `tests/`
2. Implement in appropriate module
3. Add CLI command if user-facing
4. Add MCP tool if Claude should use it
5. Run `pytest tests/ -v` to verify

### Database Changes
1. Add migration to `agentic_pipeline/db/migrations.py` in `MIGRATIONS` list
2. Write test in `tests/test_*_migrations.py`
3. Existing DBs auto-migrate on `run_migrations()`

### CLI Commands
Commands are in `agentic_pipeline/cli.py` using Click:
```python
@main.command()
@click.option("--flag", is_flag=True)
def my_command(flag: bool):
    """Command description."""
    pass
```

### MCP Tools
Tools are in `agentic_pipeline/mcp_server.py`:
```python
def my_tool(arg: str) -> dict:
    """Tool description for Claude."""
    return {"result": "value"}
```

## Testing

```bash
# All tests (use python -m pytest, not bare pytest)
python -m pytest tests/ -v

# Specific phase
python -m pytest tests/test_phase5*.py -v

# Single file
python -m pytest tests/test_autonomy_config.py -v

# With coverage
python -m pytest tests/ --cov=agentic_pipeline
```

## Environment

- Python 3.12+
- Virtual env at `.venv/`
- Database at `~/_Projects/book-ingestion-python/data/library.db` (shared by pipeline + MCP server)
- Override with `AGENTIC_PIPELINE_DB` env var

## Architecture Decisions

1. **SQLite + WAL mode** - Single-file database with concurrent read/write support; all connections use `timeout=10`
2. **Inline embedding** - `approve_book()` runs the full APPROVED → EMBEDDING → COMPLETE flow; no separate worker needed
3. **ProcessingAdapter** - Wraps `book-ingestion` as a library (not subprocess); lazy-imported in approval to avoid hard dependency
4. **Click CLI** - Standard Python CLI framework
5. **Rich** - Terminal formatting and tables
6. **TDD** - Tests written before implementation
7. **Immutable Audit** - All decisions logged permanently

## Using the Book Library

Once books are processed, they're available via MCP tools in Claude Desktop.

### Key Capabilities

| Category | Tools |
|----------|-------|
| **Search** | `semantic_search`, `text_search`, `search_all_books` |
| **Discovery** | `topic_coverage`, `cross_book_comparison`, `find_related_content` |
| **Reading** | `get_chapter`, `get_book_info`, `list_books` |
| **Learning** | `teach_concept`, `learning_path`, `create_study_guide` |
| **Planning** | `generate_project_learning_path`, `create_implementation_plan` |
| **Progress** | `mark_as_read`, `add_bookmark`, `get_reading_progress` |
| **Export** | `export_chapter`, `export_book`, `generate_flashcards` |

### MCP Client Config

Add to `.mcp.json` in any project that needs the book library:

```json
{
  "mcpServers": {
    "book-library": {
      "command": "/Users/taylorstephens/_Projects/book-mcp-server/venv/bin/python",
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

**Critical notes:**
- Command must use the **venv python** (has mcp, fastmcp, openai deps)
- Env vars are `BOOK_DB_PATH` and `BOOKS_DIR` (see `src/config.py`)
- Do NOT use `uv` (not installed) or `BOOK_LIBRARY_DB`/`LIBRARY_PATH` (wrong names)

### Example Queries to Claude

- "Search my books for Kubernetes content"
- "Create a learning path for DevOps"
- "Compare how my books explain microservices"
- "Generate flashcards from the Docker chapter"

See `docs/USER-GUIDE.md` for comprehensive usage documentation.

## Documentation

- `README.md` - User-facing overview
- `DESIGN.md` - Technical architecture
- `docs/USER-GUIDE.md` - **Comprehensive usage guide**
- `docs/PHASE4-PRODUCTION-HARDENING-COMPLETE.md` - Phase 4 features
- `docs/PHASE5-CONFIDENT-AUTONOMY-COMPLETE.md` - Phase 5 features
- `docs/plans/` - Design documents and implementation plans

---

## Memory Integration

This project uses claude-innit for persistent context:
- `get_context(project="book-mcp-server")` - Load project memory
- `remember(content, category="project", project="book-mcp-server")` - Save decisions
- `save_session(summary, project="book-mcp-server")` - End-of-session summary

---

*Last updated: 2026-02-11*

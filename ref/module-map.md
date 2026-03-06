# Module Map — agentic_pipeline

Quick reference: "which file handles X?"

---

## Top-Level

| File | Responsibility |
|------|---------------|
| `agentic_pipeline/config.py` | `OrchestratorConfig` dataclass — all tunable knobs, reads env vars |
| `agentic_pipeline/cli.py` | Click CLI entry point — 29 human-facing commands (see `ref/cli-commands.md`) |
| `agentic_pipeline/mcp_server.py` | MCP tool implementations — thin wrappers around domain modules |

---

## `orchestrator/`

| File | Responsibility |
|------|---------------|
| `orchestrator.py` | `Orchestrator` — main state machine driver. `process_one()`, `reprocess_existing()`, `run_worker()`, `retry_failed()`. All state transitions live here. |
| `errors.py` | Custom exceptions: `OrchestratorError`, `ProcessingError` |
| `logging.py` | `PipelineLogger` — structured JSON logging for pipeline events |

---

## `pipeline/`

| File | Responsibility |
|------|---------------|
| `states.py` | `PipelineState` enum, `TRANSITIONS` dict, `TERMINAL_STATES`, `can_transition()`, `is_terminal_state()` |
| `strategy.py` | Strategy selection — maps book types to processing configurations |

---

## `agents/`

| File | Responsibility |
|------|---------------|
| `classifier.py` | `ClassifierAgent` — LLM-based book classification. Delegates to `LLMProvider` (OpenAI default, Anthropic alternative) |
| `classifier_types.py` | `BookType` enum and classifier data classes |

### `agents/providers/`

| File | Responsibility |
|------|---------------|
| `base.py` | `LLMProvider` ABC — defines `classify(text, metadata) -> BookProfile` contract and `name` property |
| `openai_provider.py` | `OpenAIProvider(LLMProvider)` — calls `gpt-4.1-mini` via OpenAI SDK; normalizes unicode before sending |
| `anthropic_provider.py` | `AnthropicProvider(LLMProvider)` — calls `claude-haiku-4-5-20251001` via Anthropic SDK; reads `ANTHROPIC_API_KEY` from env |

### `agents/prompts/`

| File | Responsibility |
|------|---------------|
| `__init__.py` | `load_prompt(name) -> str` — loads prompt templates from `prompts/` directory by filename (`{name}.txt`); raises `ValueError` if not found |
| `classify.txt` | Classification prompt template used by both providers |

---

## `adapters/`

| File | Responsibility |
|------|---------------|
| `processing_adapter.py` | Wraps `book-ingestion` library for use by the orchestrator. Calls `BookIngestionApp`, handles chunking and embedding injection. |
| `llm_fallback_adapter.py` | `LLMFallbackAdapter` — implements `LLMFallbackPort` protocol from `book-ingestion`. Provides LLM assistance when chapter detection confidence is low. |

**Dependency:** Both adapters lazy-import `book_ingestion`. If not installed, pipeline falls back to `llm_fallback_adapter`.

---

## `approval/`

| File | Responsibility |
|------|---------------|
| `actions.py` | `approve_book()`, `reject_book()`, `rollback_book()`. `approve_book()` is non-blocking — spawns daemon thread for embedding. `_run_embedding_background()` and `_complete_approved()` also here. |
| `queue.py` | `ApprovalQueue` — `get_pending(sort_by)` query; formats books for human review |

---

## `audit/`

| File | Responsibility |
|------|---------------|
| `trail.py` | `AuditTrail` — append-only log. `log()` writes to `approval_audit`. `query()` filters by book_id/actor/action/date. |

---

## `autonomy/`

| File | Responsibility |
|------|---------------|
| `config.py` | `AutonomyConfig` — get/set mode, escape hatch activation, reads `autonomy_config` table |
| `metrics.py` | `MetricsCollector` — aggregates processing outcomes into `autonomy_metrics` |
| `calibration.py` | `CalibrationEngine` — calculates per-type thresholds from feedback, updates `autonomy_thresholds` |
| `spot_check.py` | `SpotCheckManager` — 10% random sample of auto-approvals for ongoing accuracy verification |

---

## `batch/`

| File | Responsibility |
|------|---------------|
| `filters.py` | `BatchFilter` dataclass — `min_confidence`, `max_confidence`, `book_type`, `max_count` |
| `operations.py` | `BatchOperations` — `approve(filter, actor, execute)`, `reject(filter, reason, actor, execute)`. `execute=False` → preview only |

---

## `db/`

| File | Responsibility |
|------|---------------|
| `connection.py` | `get_pipeline_db(path)` — canonical connection factory (WAL mode, timeout=10, row_factory=Row). **Always use this, never open sqlite3 directly.** |
| `config.py` | `get_db_path()` — resolves DB path from env var or default |
| `migrations.py` | `MIGRATIONS` list + `run_migrations(path)` — auto-applied on startup; append-only |
| `pipelines.py` | `PipelineRepository` — CRUD for `processing_pipelines`. `create()`, `get()`, `update_state()`, `update_book_profile()`, `prepare_reingest()`, `list_pending()` |

---

## `health/`

| File | Responsibility |
|------|---------------|
| `monitor.py` | `HealthMonitor` — `get_health()` returns active/queued/error counts and alerts |
| `stuck_detector.py` | `StuckDetector` — `detect()` finds pipelines exceeding `p95_seconds` per state |

---

## `backfill/`

| File | Responsibility |
|------|---------------|
| `manager.py` | `BackfillManager` — finds library books with no pipeline record, creates them. `find_untracked()`, `run(dry_run)` |
| `validator.py` | `LibraryValidator` — checks all books for quality issues: missing chapters, missing embeddings, low word count |

---

## `library/`

| File | Responsibility |
|------|---------------|
| `chapter_reader.py` | `read_chapter_content(file_path, books_dir)` — shared utility for reading chapter markdown from disk, handles split chapters |
| `migration.py` | Helpers for chunking and re-embedding the full book library (one-time migrations) |
| `status.py` | `LibraryStatus` — unified dashboard combining books, chapters, and pipeline state |

---

## `converters/`

| File | Responsibility |
|------|---------------|
| `enhanced_epub_parser.py` | EPUB text extraction with structure preservation (used by orchestrator for classification sample) |

---

## `validation/`

| File | Responsibility |
|------|---------------|
| `extraction_validator.py` | `ExtractionValidator` — validates ingested book quality: chapter count, word count, structure. `validate()` returns pass/fail with details. |

---

## Dependency Flow

```
cli.py / mcp_server.py
    └── Orchestrator (orchestrator/)
          ├── ClassifierAgent (agents/) → LLMProvider (agents/providers/)
          ├── ProcessingAdapter (adapters/) → book-ingestion library
          ├── ExtractionValidator (validation/)
          ├── PipelineRepository (db/)
          └── approval/actions.py
                ├── AuditTrail (audit/)
                └── _complete_approved() → runs embedding inline
```

**Key rule:** `db/connection.py:get_pipeline_db()` is the only correct way to open the pipeline DB. All other modules call it — never open `sqlite3.connect()` directly.

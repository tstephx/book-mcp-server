# Database Schema Reference

All pipeline state lives in a single SQLite file (WAL mode).
- **Location:** `AGENTIC_PIPELINE_DB` env var, or `~/_Projects/book-ingestion-python/data/library.db`
- **Connection:** always via `get_pipeline_db()` in `agentic_pipeline/db/connection.py` (timeout=10, row_factory=sqlite3.Row)
- **Migrations:** `agentic_pipeline/db/migrations.py` — `MIGRATIONS` list, auto-applied by `run_migrations()`

---

## Core Pipeline Tables

### `processing_pipelines`
Primary pipeline state record for each book.

| Column | Type | Notes |
|--------|------|-------|
| `id` | TEXT PK | UUID |
| `source_path` | TEXT | Original file path |
| `content_hash` | TEXT | SHA256 — dedup key |
| `state` | TEXT | `PipelineState` enum value |
| `book_profile` | JSON | Classification result: `{book_type, confidence, suggested_tags, ...}` |
| `strategy_config` | JSON | Processing strategy selected |
| `validation_result` | JSON | Validation output |
| `processing_result` | JSON | Ingestion result |
| `approved_by` | TEXT | Actor who approved (`human:X` or `auto:X`) |
| `approved_at` | TIMESTAMP | Approval time |
| `rejected_by` | TEXT | Actor who rejected |
| `rejected_at` | TIMESTAMP | Rejection time |
| `rejection_reason` | TEXT | Human-readable rejection reason |
| `retry_count` | INTEGER | Times retried, default 0 |
| `created_at` | TIMESTAMP | When record created |
| `updated_at` | TIMESTAMP | Last state change |

### `pipeline_state_history`
Append-only audit trail of every state transition.

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PK | Auto-increment |
| `pipeline_id` | TEXT | FK → processing_pipelines.id |
| `from_state` | TEXT | Previous state (null on initial) |
| `to_state` | TEXT | New state |
| `duration_ms` | INTEGER | Time spent in previous state |
| `agent_output` | JSON | LLM/processor output at this step |
| `error_details` | JSON | Error info if transition was triggered by failure |
| `created_at` | TIMESTAMP | When transition occurred |

### `processing_strategies`
Named processing configurations by book type.

| Column | Type |
|--------|------|
| `name` | TEXT PK |
| `book_type` | TEXT |
| `config` | JSON |
| `version` | INTEGER |
| `created_at` | TIMESTAMP |
| `is_active` | BOOLEAN |

### `pipeline_config`
Key-value store for runtime config.

| Column | Type |
|--------|------|
| `key` | TEXT PK |
| `value` | JSON |
| `updated_at` | TIMESTAMP |

---

## Approval & Audit Tables

### `approval_audit`
Every approve/reject/rollback action.

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PK | Auto-increment |
| `book_id` | TEXT | Pipeline/book ID |
| `pipeline_id` | TEXT | May differ if reingested |
| `action` | TEXT | `approved`/`rejected`/`rollback` |
| `actor` | TEXT | e.g. `human:taylor`, `auto:confident` |
| `reason` | TEXT | For rejections |
| `before_state` | JSON | State snapshot before action |
| `after_state` | JSON | State snapshot after action |
| `created_at` | TIMESTAMP | |
| `adjustments` | JSON | Any field adjustments made at approval |

### `audit_retention`
Cleanup schedule per audit type.

| Column | Type |
|--------|------|
| `audit_type` | TEXT PK |
| `retain_days` | INTEGER |
| `last_cleanup` | TIMESTAMP |

---

## Autonomy Tables

### `autonomy_config`
Singleton (id=1). Current autonomy mode and thresholds.

| Column | Type | Default |
|--------|------|---------|
| `current_mode` | TEXT | `supervised` |
| `auto_approve_threshold` | REAL | `0.95` |
| `escape_hatch_active` | BOOLEAN | `false` |
| `escape_hatch_reason` | TEXT | — |
| `escape_hatch_at` | TIMESTAMP | — |

### `autonomy_metrics`
Periodic rollup of processing outcomes.

| Column | Type |
|--------|------|
| `period_start/end` | DATE |
| `total_processed` | INTEGER |
| `auto_approved` | INTEGER |
| `human_approved` | INTEGER |
| `human_rejected` | INTEGER |
| `human_adjusted` | INTEGER |

### `autonomy_thresholds`
Per-book-type calibrated thresholds (set by `CalibrationEngine`).

| Column | Type |
|--------|------|
| `book_type` | TEXT PK |
| `auto_approve_threshold` | REAL |
| `sample_count` | INTEGER |
| `measured_accuracy` | REAL |
| `manual_override` | REAL |
| `override_reason` | TEXT |

### `autonomy_feedback`
Human override records used for calibration.

| Column | Type |
|--------|------|
| `book_id` | TEXT |
| `original_decision` | TEXT |
| `original_confidence` | REAL |
| `original_book_type` | TEXT |
| `human_decision` | TEXT |
| `human_adjustments` | JSON |

### `spot_checks`
Random audit samples of auto-approved books.

| Column | Type |
|--------|------|
| `book_id` | TEXT |
| `original_classification` | TEXT |
| `original_confidence` | REAL |
| `classification_correct` | BOOLEAN |
| `quality_acceptable` | BOOLEAN |

---

## Health Tables

### `health_metrics`
Singleton (id=1). Current pipeline health snapshot.

| Column | Type |
|--------|------|
| `active_count` | INTEGER |
| `queued_count` | INTEGER |
| `stuck_count` | INTEGER |
| `error_rate` | REAL |
| `last_updated` | TIMESTAMP |

### `state_duration_stats`
Expected duration per state (for stuck detection).

| Column | Type |
|--------|------|
| `state` | TEXT PK |
| `sample_count` | INTEGER |
| `median_seconds` | REAL |
| `p95_seconds` | REAL |
| `max_seconds` | REAL |

---

## Embedding Table

### `chunks` (in book-library DB)
Chapter/section chunks with embeddings for semantic search.

| Column | Type | Notes |
|--------|------|-------|
| `id` | TEXT PK | UUID |
| `chapter_id` | TEXT | FK → chapters |
| `book_id` | TEXT | FK → books |
| `chunk_index` | INTEGER | Position within chapter |
| `content` | TEXT | Chunk text |
| `word_count` | INTEGER | |
| `embedding` | BLOB | numpy float32 array (3072 dims, text-embedding-3-large) |
| `embedding_model` | TEXT | Model used |

**Note:** Embeddings are generated inline during `approve_book()` → EMBEDDING → COMPLETE. No separate embedding worker.

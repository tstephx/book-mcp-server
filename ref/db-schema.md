---
status: active
tags: [project/book-mcp-server, format/reference]
type: project
created: '2026-03-05'
modified: '2026-03-05'
---

# Database Schema Reference

All pipeline state lives in a single SQLite file (WAL mode).
- **Location:** `AGENTIC_PIPELINE_DB` env var, or `~/Library/Application Support/book-library/library.db`
- **Connection:** always via `get_pipeline_db()` in `agentic_pipeline/db/connection.py` (timeout=10, row_factory=sqlite3.Row)
- **Migrations:** `agentic_pipeline/db/migrations.py` — `MIGRATIONS` list, auto-applied by `run_migrations()`

---

## Core Pipeline Tables

### `processing_pipelines`
Primary pipeline state record for each book.

| Column | Type | Notes |
|--------|------|-------|
| `id` | TEXT PK | UUID |
| `source_path` | TEXT NOT NULL | Original file path |
| `content_hash` | TEXT NOT NULL | SHA256 — dedup key; enforced unique via table constraint |
| `state` | TEXT NOT NULL | `PipelineState` enum value |
| `book_profile` | JSON | Classification result: `{book_type, confidence, suggested_tags, ...}` |
| `strategy_config` | JSON | Processing strategy selected |
| `validation_result` | JSON | Validation output |
| `processing_result` | JSON | Ingestion result |
| `retry_count` | INTEGER | Default 0 |
| `max_retries` | INTEGER | Default 2 |
| `error_log` | JSON | Error details accumulated across retries |
| `created_at` | TIMESTAMP | When record created |
| `updated_at` | TIMESTAMP | Last state change |
| `completed_at` | TIMESTAMP | When pipeline reached terminal state |
| `timeout_at` | TIMESTAMP | Timeout deadline for current state |
| `last_heartbeat` | TIMESTAMP | Last worker heartbeat |
| `priority` | INTEGER | Default 5 (lower = higher priority) |
| `approved_by` | TEXT | Actor who approved (`human:X` or `auto:X`) |
| `approval_confidence` | REAL | Confidence score at time of approval |

### `pipeline_state_history`
Append-only audit trail of every state transition.

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PK | Auto-increment |
| `pipeline_id` | TEXT | FK → processing_pipelines.id |
| `from_state` | TEXT | Previous state (null on initial) |
| `to_state` | TEXT NOT NULL | New state |
| `duration_ms` | INTEGER | Time spent in previous state |
| `agent_output` | JSON | LLM/processor output at this step |
| `error_details` | JSON | Error info if transition was triggered by failure |
| `created_at` | TIMESTAMP | When transition occurred |

### `processing_strategies`
Named processing configurations by book type.

| Column | Type |
|--------|------|
| `name` | TEXT PK |
| `book_type` | TEXT NOT NULL |
| `config` | JSON NOT NULL |
| `version` | INTEGER (default 1) |
| `created_at` | TIMESTAMP |
| `is_active` | BOOLEAN (default TRUE) |

### `pipeline_config`
Key-value store for runtime config.

| Column | Type |
|--------|------|
| `key` | TEXT PK |
| `value` | JSON NOT NULL |
| `updated_at` | TIMESTAMP |

---

## Approval & Audit Tables

### `approval_audit`
Every approve/reject/rollback action.

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PK | Auto-increment |
| `book_id` | TEXT NOT NULL | Pipeline/book ID |
| `pipeline_id` | TEXT | May differ if reingested |
| `action` | TEXT NOT NULL | `approved`/`rejected`/`rollback` |
| `actor` | TEXT NOT NULL | e.g. `human:taylor`, `auto:confident` |
| `reason` | TEXT | For rejections |
| `before_state` | JSON | State snapshot before action |
| `after_state` | JSON | State snapshot after action |
| `adjustments` | JSON | Any field adjustments made at approval |
| `filter_used` | JSON | Batch filter criteria (for batch operations) |
| `confidence_at_decision` | REAL | Book confidence score at decision time |
| `autonomy_mode` | TEXT | Mode active when decision was made |
| `session_id` | TEXT | Session identifier for grouping actions |
| `performed_at` | TIMESTAMP | When action occurred (default CURRENT_TIMESTAMP) |

### `audit_retention`
Cleanup schedule per audit type.

| Column | Type |
|--------|------|
| `audit_type` | TEXT PK |
| `retain_days` | INTEGER NOT NULL |
| `last_cleanup` | TIMESTAMP |

---

## Autonomy Tables

### `autonomy_config`
Singleton (id=1). Current autonomy mode and guard-rail settings.

| Column | Type | Default |
|--------|------|---------|
| `id` | INTEGER PK | CHECK (id = 1) |
| `current_mode` | TEXT | `supervised` |
| `auto_approve_threshold` | REAL | `0.95` |
| `auto_retry_threshold` | REAL | `0.70` |
| `require_known_book_type` | BOOLEAN | `TRUE` |
| `require_zero_issues` | BOOLEAN | `TRUE` |
| `max_auto_approvals_per_day` | INTEGER | `50` |
| `spot_check_percentage` | REAL | `0.10` |
| `escape_hatch_active` | BOOLEAN | `FALSE` |
| `escape_hatch_activated_at` | TIMESTAMP | — |
| `escape_hatch_reason` | TEXT | — |
| `updated_at` | TIMESTAMP | CURRENT_TIMESTAMP |

### `autonomy_metrics`
Periodic rollup of processing outcomes.

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PK | Auto-increment |
| `period_start` | DATE NOT NULL | UNIQUE(period_start, period_end) composite constraint |
| `period_end` | DATE NOT NULL | |
| `total_processed` | INTEGER | Default 0 |
| `auto_approved` | INTEGER | Default 0 |
| `human_approved` | INTEGER | Default 0 |
| `human_rejected` | INTEGER | Default 0 |
| `human_adjusted` | INTEGER | Default 0 |
| `avg_confidence_auto_approved` | REAL | |
| `avg_confidence_human_approved` | REAL | |
| `avg_confidence_human_rejected` | REAL | |
| `auto_approved_later_rolled_back` | INTEGER | Default 0 |
| `human_approved_later_rolled_back` | INTEGER | Default 0 |
| `metrics_by_type` | JSON | Per-book-type breakdown |
| `confidence_buckets` | JSON | Distribution of confidence scores |
| `created_at` | TIMESTAMP | |

### `autonomy_thresholds`
Per-book-type calibrated thresholds (set by `CalibrationEngine`).

| Column | Type |
|--------|------|
| `book_type` | TEXT PK |
| `auto_approve_threshold` | REAL |
| `sample_count` | INTEGER (default 0) |
| `measured_accuracy` | REAL |
| `last_calculated` | TIMESTAMP |
| `calibration_data` | JSON |
| `manual_override` | REAL |
| `override_reason` | TEXT |

### `autonomy_feedback`
Human override records used for calibration.

| Column | Type |
|--------|------|
| `id` | INTEGER PK (auto-increment) |
| `book_id` | TEXT NOT NULL |
| `pipeline_id` | TEXT |
| `original_decision` | TEXT NOT NULL |
| `original_confidence` | REAL |
| `original_book_type` | TEXT |
| `human_decision` | TEXT NOT NULL |
| `human_adjustments` | JSON |
| `feedback_category` | TEXT |
| `feedback_notes` | TEXT |
| `created_at` | TIMESTAMP |

### `spot_checks`
Random audit samples of auto-approved books.

| Column | Type |
|--------|------|
| `id` | INTEGER PK (auto-increment) |
| `book_id` | TEXT NOT NULL |
| `pipeline_id` | TEXT |
| `original_classification` | TEXT |
| `original_confidence` | REAL |
| `auto_approved_at` | TIMESTAMP |
| `classification_correct` | BOOLEAN |
| `quality_acceptable` | BOOLEAN |
| `reviewer` | TEXT |
| `notes` | TEXT |
| `checked_at` | TIMESTAMP (default CURRENT_TIMESTAMP) |

---

## Health Tables

### `health_metrics`
Singleton (id=1). Current pipeline health snapshot.

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PK | CHECK (id = 1) |
| `active_count` | INTEGER NOT NULL | Default 0 |
| `queued_count` | INTEGER NOT NULL | Default 0 |
| `stuck_count` | INTEGER NOT NULL | Default 0 |
| `completed_24h` | INTEGER NOT NULL | Default 0 |
| `failed_count` | INTEGER NOT NULL | Default 0 |
| `avg_processing_seconds` | REAL | |
| `queue_by_priority` | JSON | Counts per priority level |
| `stuck_pipelines` | JSON | Details of stuck pipelines |
| `alerts` | JSON | Active alerts |
| `updated_at` | TIMESTAMP NOT NULL | Default CURRENT_TIMESTAMP |

### `state_duration_stats`
Expected duration per state (for stuck detection).

| Column | Type |
|--------|------|
| `state` | TEXT PK |
| `sample_count` | INTEGER NOT NULL (default 0) |
| `median_seconds` | REAL NOT NULL (default 0) |
| `p95_seconds` | REAL NOT NULL (default 0) |
| `max_seconds` | REAL NOT NULL (default 0) |
| `updated_at` | TIMESTAMP NOT NULL (default CURRENT_TIMESTAMP) |

---

## Embedding Table

### `chunks`
Chapter/section chunks with embeddings for semantic search. Created by pipeline migrations (`run_migrations()`); lives in the shared library DB.

| Column | Type | Notes |
|--------|------|-------|
| `id` | TEXT PK | UUID |
| `chapter_id` | TEXT NOT NULL | FK → chapters |
| `book_id` | TEXT NOT NULL | FK → books |
| `chunk_index` | INTEGER NOT NULL | Position within chapter |
| `content` | TEXT NOT NULL | Chunk text |
| `word_count` | INTEGER NOT NULL | |
| `embedding` | BLOB | numpy float32 array (3072 dims, text-embedding-3-large) |
| `embedding_model` | TEXT | Model used |
| `content_hash` | TEXT | Content dedup hash |
| `created_at` | TIMESTAMP | Default CURRENT_TIMESTAMP |

**Note:** Embeddings are generated inline during `approve_book()` → EMBEDDING → COMPLETE. No separate embedding worker.

---

## Infrastructure Tables

### `schema_migrations`
Tracks which versioned ALTER TABLE migrations have been applied.

| Column | Type |
|--------|------|
| `name` | TEXT PK |
| `applied_at` | TIMESTAMP (default CURRENT_TIMESTAMP) |

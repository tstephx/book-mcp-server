---
status: active
tags: [project/book-mcp-server, format/reference]
type: project
created: '2026-03-04'
modified: '2026-03-04'
---

# Pipeline Architecture Reference

## State Machine

Books flow through these states in order:

```
DETECTED → HASHING → CLASSIFYING → SELECTING_STRATEGY → PROCESSING
         → VALIDATING → PENDING_APPROVAL → APPROVED → EMBEDDING → COMPLETE
```

**Shortcut:** High-confidence books (≥0.7, `auto_review=False`) skip `PENDING_APPROVAL` and auto-approve.

**Terminal states:** `COMPLETE`, `REJECTED`, `ARCHIVED`, `DUPLICATE`, `FAILED`

**Error states:** `NEEDS_RETRY` (retryable), `FAILED` (permanent, not archivable)

### Full State List (`agentic_pipeline/pipeline/states.py`)

| State | Value | Notes |
|-------|-------|-------|
| `DETECTED` | `detected` | File found by watcher |
| `HASHING` | `hashing` | Computing content hash |
| `DUPLICATE` | `duplicate` | Terminal — same hash exists |
| `CLASSIFYING` | `classifying` | LLM classification |
| `SELECTING_STRATEGY` | `selecting_strategy` | Choosing processing strategy |
| `PROCESSING` | `processing` | Running book ingestion |
| `VALIDATING` | `validating` | Quality validation |
| `PENDING_APPROVAL` | `pending_approval` | Awaiting human review |
| `NEEDS_RETRY` | `needs_retry` | Transient failure, will retry |
| `APPROVED` | `approved` | Human/auto approved |
| `EMBEDDING` | `embedding` | Generating embeddings |
| `COMPLETE` | `complete` | Terminal — success |
| `REJECTED` | `rejected` | Human rejected or validation failed |
| `ARCHIVED` | `archived` | Terminal — soft-deleted |
| `FAILED` | `failed` | Terminal — permanent error |

---

## Orchestrator (`agentic_pipeline/orchestrator/orchestrator.py`)

### Public Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `process_one` | `(book_path: str) -> dict` | Full pipeline: hash → classify → process → validate → approve/pending |
| `reprocess_existing` | `(pipeline_id, book_path, content_hash) -> dict` | Re-drive existing record through pipeline (used by `reingest`) |
| `run_worker` | `() -> None` | Long-running worker loop with watcher, retries, health |
| `retry_failed` | `() -> list[dict]` | Retry all NEEDS_RETRY pipelines |

### Private State Machine Steps

| Method | Transition(s) |
|--------|---------------|
| `_compute_hash` | — |
| `_check_idempotency` | Detects DUPLICATE |
| `_run_classifier` | CLASSIFYING → SELECTING_STRATEGY |
| `_run_processing` | PROCESSING → VALIDATING |
| `_run_embedding` | EMBEDDING → COMPLETE |
| `_process_book` | Drives full CLASSIFYING→PENDING_APPROVAL flow |
| `_complete_approved` | APPROVED → EMBEDDING → COMPLETE (called by background thread) |

### Worker Loop (`run_worker`)

Poll cycle (every `WORKER_POLL_INTERVAL_SECONDS`, default 5s):
1. Retry NEEDS_RETRY pipelines
2. Process any PROCESSING/VALIDATING stuck books
3. Scan watch directory for new files (lowest priority)

---

## Approval Flow (`agentic_pipeline/approval/`)

### `approve_book(db_path, pipeline_id, actor, adjustments)` → `dict`

```
1. Validate pipeline is in PENDING_APPROVAL
2. Transition → APPROVED
3. Write approval_audit record
4. Spawn daemon thread: _run_embedding_background(db_path, pipeline_id, pipeline)
5. Return immediately: {"success": True, "state": "approved", "embedding": "queued"}
```

Background thread calls `_complete_approved()` → EMBEDDING → COMPLETE (non-blocking).

### `reject_book(db_path, pipeline_id, reason, actor, retry)` → `dict`

Transitions to REJECTED (or NEEDS_RETRY if `retry=True`). Writes audit record.

### `rollback_book(db_path, pipeline_id, reason, actor)` → `dict`

Reverts COMPLETE/APPROVED book: removes chapters from library, transitions to ARCHIVED.

---

## Autonomy Modes (`agentic_pipeline/autonomy/`)

| Mode | Behavior |
|------|----------|
| `supervised` | All books require human approval (default) |
| `partial` | Auto-approve high-confidence known book types |
| `confident` | Per-type calibrated thresholds from `autonomy_thresholds` |

**Escape hatch:** `agentic-pipeline escape-hatch "reason"` → immediately reverts to `supervised`.

### Thresholds
- `CONFIDENCE_THRESHOLD` env var (default `0.7`) — minimum confidence for auto-approve
- Per-type thresholds stored in `autonomy_thresholds` table, updated by `CalibrationEngine`
- Readiness gates: partial requires 100 processed + <15% override rate; confident requires 500 + <5%

---

## File Watcher

```bash
agentic-pipeline worker --watch-dir ~/Documents/_ebooks/agentic-book-pipeline \
                         --processed-dir ~/Documents/_ebooks/agentic-book-pipeline/processed
```

- Accepts `.epub` and `.pdf` only
- Deduplication via content hash — dropping the same file twice is a no-op
- Files in `processed_dir` are excluded from scans
- Archive on success: moves file to `processed_dir` with counter suffix on collision

---

## Configuration (`agentic_pipeline/config.py`)

All values overridable via env vars:

| Knob | Env var | Default |
|------|---------|---------|
| Auto-approve threshold | `CONFIDENCE_THRESHOLD` | `0.7` |
| Processing timeout | `PROCESSING_TIMEOUT_SECONDS` | `600s` |
| Embedding timeout | `EMBEDDING_TIMEOUT_SECONDS` | `300s` |
| Worker poll interval | `WORKER_POLL_INTERVAL_SECONDS` | `5s` |
| Max retries | `MAX_RETRY_ATTEMPTS` | `3` |
| Pipeline DB | `AGENTIC_PIPELINE_DB` | — |
| Watch directory | `WATCH_DIR` | — |
| Processed directory | `PROCESSED_DIR` | — |

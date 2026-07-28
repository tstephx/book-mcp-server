---
status: active
tags: [project/book-mcp-server, format/reference]
type: project
created: '2026-03-04'
modified: '2026-07-28'
---

# Pipeline Architecture Reference

## State Machine

Books flow through these states in order:

```
DETECTED → HASHING → CLASSIFYING → SELECTING_STRATEGY → PROCESSING
         → VALIDATING → PENDING_APPROVAL → APPROVED → EMBEDDING → COMPLETE
```

Every validated book reaches `PENDING_APPROVAL`. The autonomy policy may then
approve it through the same audited action used for human and batch approval.

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
3. Write the per-book `approval_audit` and autonomy feedback records
4. Start embedding inline or spawn `_run_embedding_background`, according to `background`
5. Return the embedding outcome or queued status
```

If either governance record cannot be written, embedding does not start and the
book moves to `NEEDS_RETRY`. The background path calls `_complete_approved()` →
EMBEDDING → COMPLETE without blocking the caller.

### `reject_book(db_path, pipeline_id, reason, actor, retry)` → `dict`

Transitions to REJECTED (or NEEDS_RETRY if `retry=True`). Writes audit record.

### `rollback_book(db_path, pipeline_id, reason, actor)` → `dict`

Reverts COMPLETE/APPROVED book: removes chapters from library, transitions to ARCHIVED.

---

## Autonomy Modes (`agentic_pipeline/autonomy/`)

| Mode | Behavior |
|------|----------|
| `supervised` | All books require human approval (default) |
| `partial` | Eligible known types use `autonomy_config.auto_approve_threshold` (default `0.95`) |
| `confident` | Eligible known types use reviewed per-type thresholds from `autonomy_thresholds` |

**Escape hatch:** `agentic-pipeline escape-hatch "reason"` → immediately reverts to `supervised`.

### Thresholds
- `autonomy_config.auto_approve_threshold` (default `0.95`) controls partial mode
- Per-type thresholds stored in `autonomy_thresholds` table, updated by `CalibrationEngine`
- `CONFIDENCE_THRESHOLD` (default `0.7`) controls classifier LLM fallback, not approval
- Readiness gates: partial requires 100 processed + <15% override rate; confident requires 500 + <5%

### Guardrails and feedback

Automatic approval is denied when the escape hatch is active, the mode is
supervised or invalid, validation failed, review is requested, the book type or
confidence is invalid, the applicable threshold is unavailable, or the daily
automatic-approval cap has been reached. Policy errors also fail closed.

Each auto approval records the original processing confidence, active mode,
threshold, and decision reason in the per-book audit record. Its feedback begins
as `pending_review`; only reviewed approved/rejected outcomes are used for
accuracy and threshold calibration. Spot-check results convert pending feedback
into a reviewed outcome.

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
| Classifier fallback threshold | `CONFIDENCE_THRESHOLD` | `0.7` |
| Processing timeout | `PROCESSING_TIMEOUT_SECONDS` | `600s` |
| Embedding timeout | `EMBEDDING_TIMEOUT_SECONDS` | `300s` |
| Worker poll interval | `WORKER_POLL_INTERVAL_SECONDS` | `5s` |
| Max retries | `MAX_RETRY_ATTEMPTS` | `3` |
| Pipeline DB | `AGENTIC_PIPELINE_DB` | — |
| Watch directory | `WATCH_DIR` | — |
| Processed directory | `PROCESSED_DIR` | — |

Approval thresholds and guardrails are database settings in `autonomy_config`
and `autonomy_thresholds`; they are not environment variables.

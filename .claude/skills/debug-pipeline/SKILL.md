---
name: debug-pipeline
description: Diagnose a stuck or failed pipeline entry. Runs health check, state history, audit trail, and error log analysis.
disable-model-invocation: true
---

# Debug Pipeline

Arguments: `[book_id_or_pipeline_id]` (optional — if omitted, runs general health diagnostics)

## Context

Pipeline states flow: `DETECTED` -> `HASHING` -> `CLASSIFYING` -> `SELECTING_STRATEGY` -> `PROCESSING` -> `VALIDATING` -> `PENDING_APPROVAL` -> `APPROVED` -> `EMBEDDING` -> `COMPLETE`

Error states: `NEEDS_RETRY`, `FAILED`

DB path resolved via `AGENTIC_PIPELINE_DB` env var or default `~/Library/Application Support/book-library/library.db`.

## Steps

### 1. Run health check

```bash
agentic-pipeline health --json
```

Look for: stuck counts, error counts, queue depth.

### 2. Check for stuck pipelines

```bash
agentic-pipeline stuck
```

If stuck books exist, note their IDs and states.

### 3. If a specific book/pipeline ID was provided

Query its full state:

```bash
agentic-pipeline status <id>
```

Then query state history directly:

```sql
SELECT psh.from_state, psh.to_state, psh.transitioned_at, psh.trigger
FROM pipeline_state_history psh
JOIN processing_pipelines pp ON pp.id = psh.pipeline_id
WHERE pp.id = '<id>' OR pp.book_id = '<id>'
ORDER BY psh.transitioned_at DESC
LIMIT 15;
```

### 4. Check error log

```sql
SELECT id, current_state, error_log, retry_count, max_retries,
       last_heartbeat, timeout_at
FROM processing_pipelines
WHERE id = '<id>' OR book_id = '<id>';
```

### 5. Check audit trail

```bash
agentic-pipeline audit --book-id <id> --last 10
```

### 6. Synthesize diagnosis

Report:
- **Current state**: what state the book is in
- **Time stuck**: how long since last transition
- **Last error**: from error_log column
- **Retry status**: retry_count / max_retries
- **Recommended action**: one of:
  - `agentic-pipeline retry` — if NEEDS_RETRY with retries remaining
  - `agentic-pipeline stuck --recover` — if stuck in a processing state
  - `agentic-pipeline reingest <book_id>` — if corrupted/need full redo
  - `agentic-pipeline reject <pipeline_id> --retry` — if needs manual review first
  - Check `OPENAI_API_KEY` — if stuck in EMBEDDING state

### 7. If no specific ID provided

Summarize overall pipeline health:
- Total books by state
- Any stuck or failed entries
- Recent errors (last 24h)
- Queue depth and processing rate

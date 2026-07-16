---
status: active
tags: [project/book-mcp-server, format/reference]
type: project
created: '2026-03-05'
modified: '2026-03-05'
---

# CLI Commands Reference

Entry point: `agentic-pipeline` → `agentic_pipeline/cli.py` (Click)

31 commands organized by category.

## Pipeline Operations

| Command | Args / Options | Description |
|---------|---------------|-------------|
| `init` | — | Initialize the database with agentic pipeline tables |
| `process` | `BOOK_PATH` | Process a single book through the pipeline |
| `worker` | `[--watch-dir] [--processed-dir]` | Run the queue worker (processes books continuously) |
| `status` | `PIPELINE_ID` | Show status of a pipeline |
| `health` | `[--json]` | Show pipeline health status |
| `stuck` | `[--recover]` | List stuck pipelines |
| `retry` | `[--max-attempts/-m]` | Retry books in NEEDS_RETRY state |
| `reingest` | `BOOK_ID [--force-fallback]` | Reprocess a book through the full pipeline |
| `reprocess` | `[--flagged] [--execute] [--json]` | Re-queue books that fail extraction quality checks |
| `backfill` | `[--dry-run] [--execute]` | Register legacy library books in the pipeline |
| `version` | — | Show version |

## Approval Flow

| Command | Args / Options | Description |
|---------|---------------|-------------|
| `pending` | — | List books pending approval |
| `approve` | `PIPELINE_ID` | Approve a pending book and generate embeddings |
| `reject` | `PIPELINE_ID [--reason/-r] [--retry]` | Reject a pending book |
| `batch-approve` | `[--min-confidence] [--book-type] [--max-count] [--execute]` | Approve books matching filters |
| `batch-reject` | `[--book-type] [--max-confidence] [--reason] [--max-count] [--execute]` | Reject books matching filters |

## Autonomy Management

| Command | Args / Options | Description |
|---------|---------------|-------------|
| `autonomy` | — | Manage autonomy settings |
| `escape-hatch` | `REASON` | Activate escape hatch — immediately revert to supervised mode |
| `spot-check` | `[--list] [--days] [--reviewer]` | Interactively review a sample of auto-approved books |

## Library Maintenance

| Command | Args / Options | Description |
|---------|---------------|-------------|
| `library-status` | `[--json]` | Show library status dashboard |
| `library-issues` | — | Report library data quality issues (empty titles, duplicates, etc.) |
| `update-title` | `BOOK_ID CHAPTER_NUMBER NEW_TITLE` | Update a chapter's title |
| `validate` | `[--json]` | Check library books for quality issues |
| `audit-quality` | `[--json]` | Audit library books for extraction quality issues |
| `rechunk` | `[--fresh] [--swap] [--yes]` | Re-chunk the whole library into `chunks_staging`, embed, and score against the gold query sets (`eval/gold-queries*.json`). Hash-keyed: reruns reuse existing staged embeddings; interruption is resumable by rerunning. Production `chunks` untouched. Exit 0 = eval PASS, 1 = FAIL. `--fresh` drops staging first. `--yes` skips the embedding-cost confirmation. `--swap` applies a PASSed staging: stages any delta (books approved since the eval), backs up the DB, atomically replaces `chunks`, bumps `library_meta.data_version` (running MCP servers self-invalidate their chunk cache). Refuses without a PASS verdict. |

## Embedding & Chunking

| Command | Args / Options | Description |
|---------|---------------|-------------|
| `embed-library` | `[--dry-run] [--batch-size]` | Generate OpenAI embeddings for all unembedded chunks |
| `chunk-library` | `[--dry-run]` | Generate chunks for all unchunked chapters |

## Diagnostics

| Command | Args / Options | Description |
|---------|---------------|-------------|
| `classify` | `[--text/-t] [--provider/-p]` | Classify book text and show the result |
| `strategies` | — | List available processing strategies |
| `audit` | `[--last] [--actor] [--action] [--book-id]` | Query the audit trail |
| `doctor` | `[--fix] [--no-backup] [--manifest]` | Detect (and with `--fix`, repair) orphaned chunks, lost books, null content hashes, null book types. Exit codes: `0` clean report or `--fix` completed with no unresolved failures; `1` violations found (bare) or a fix category failed (e.g. CAS conflict archiving a lost book). `--fix` takes an automatic WAL-safe backup first (skippable via `--no-backup`) when anything will actually mutate, writes a lost-books manifest, and prints `reingest` commands for recoverable books. |

## Common Workflows

```bash
# Process a new book end-to-end
agentic-pipeline process ~/Documents/_ebooks/book.epub

# Watch for new books and auto-process
agentic-pipeline worker --watch-dir ~/Documents/_ebooks/agentic-book-pipeline \
  --processed-dir ~/Documents/_ebooks/agentic-book-pipeline/processed

# Check and fix data quality
agentic-pipeline library-issues
agentic-pipeline update-title "Clean Code" 3 "Functions"

# Review auto-approved books
agentic-pipeline spot-check --days 7
```

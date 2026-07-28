# Ingestion Integrity Persistence Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make new pipeline ingestions persist chapter content hashes and the classifier's complete book profile so integrity checks are clean before approval.

**Architecture:** `book-ingestion-python` will hash the exact rendered chapter text at the file-writing boundary and store that hash through its chapter repository. `book-mcp-server` will pass the already-persisted pipeline profile into its processing adapter, which will copy the complete profile onto the newly written `books` row after ingestion succeeds. The existing doctor remains a one-time repair mechanism for old rows, not part of normal ingestion.

**Tech Stack:** Python 3.12, SQLite, pytest, SHA-256, editable sibling package integration.

### Task 1: Persist exact rendered chapter hashes

**Files:**
- Modify: `/Users/taylorstephens/Dev/_Projects/book-ingestion-python/book_ingestion/storage/file_writer.py`
- Modify: `/Users/taylorstephens/Dev/_Projects/book-ingestion-python/book_ingestion/storage/database.py`
- Test: `/Users/taylorstephens/Dev/_Projects/book-ingestion-python/tests/processors/test_pipeline.py`

**Step 1: Write the failing tests**

- Verify `FileWriter.write_book()` sets `chapter["content_hash"]` to SHA-256 of the full rendered Markdown file.
- Verify `FileWriter.write_book_with_sections()` sets the hash to SHA-256 of the numbered section files joined with the same `"\n\n"` separator used by chapter readers.
- Verify `BookDatabase.initialize()` adds `chapters.content_hash` to a legacy database.
- Verify `BookDatabase.insert_chapter()` persists the supplied hash.

**Step 2: Run the tests to verify they fail**

Run:

```bash
../../.venv/bin/pytest tests/processors/test_pipeline.py -q
```

Expected: failures because `content_hash` is neither generated nor stored.

**Step 3: Write the minimal implementation**

- Render each chapter to a string before writing it.
- Hash that exact string with SHA-256.
- For split chapters, hash numbered section strings in reader order with `"\n\n"` between them; exclude `_index.md`.
- Add `content_hash TEXT` to the current schema and idempotently migrate legacy `chapters` tables.
- Include `content_hash` in chapter inserts.

**Step 4: Run focused and full dependency tests**

Run:

```bash
../../.venv/bin/pytest tests/processors/test_pipeline.py -q
../../.venv/bin/pytest -q --deselect tests/processors/test_semantic_chunker.py::TestSemanticChunker::test_detect_boundaries_basic
```

Expected: focused tests pass; full suite reports 233 or more passing tests with the known model-sensitive test deselected.

**Step 5: Commit**

```bash
git add book_ingestion/storage/database.py book_ingestion/storage/file_writer.py tests/processors/test_pipeline.py
git commit -m "fix: persist rendered chapter hashes"
```

### Task 2: Persist the classified book profile after ingestion

**Files:**
- Modify: `/Users/taylorstephens/Dev/_Projects/book-mcp-server/agentic_pipeline/adapters/processing_adapter.py`
- Modify: `/Users/taylorstephens/Dev/_Projects/book-mcp-server/agentic_pipeline/orchestrator/orchestrator.py`
- Test: `/Users/taylorstephens/Dev/_Projects/book-mcp-server/tests/test_processing_adapter.py`
- Test: `/Users/taylorstephens/Dev/_Projects/book-mcp-server/tests/test_orchestrator.py`

**Step 1: Write the failing tests**

- Verify a successful stored ingestion copies `book_type`, confidence, tags, reasoning, timestamp, and classifier identity to `books`.
- Verify `save_to_storage=False` does not attempt a database update.
- Verify the orchestrator passes the profile to processing on both the first attempt and the forced-fallback retry.

**Step 2: Run the tests to verify they fail**

Run:

```bash
../../.venv/bin/pytest tests/test_processing_adapter.py tests/test_orchestrator.py -q
```

Expected: failures because the adapter has no profile argument or persistence step.

**Step 3: Write the minimal implementation**

- Add an optional `book_profile` argument to `ProcessingAdapter.process_book()`.
- After successful storage, update the row by the returned `book_id` using the existing library classification columns.
- JSON-encode tags and write an ISO-8601 UTC timestamp.
- Add `book_profile` to `Orchestrator._run_processing()` and pass the current profile to both processing attempts.

**Step 4: Run focused and full pipeline tests**

Run:

```bash
../../.venv/bin/pytest tests/test_processing_adapter.py tests/test_orchestrator.py -q
../../.venv/bin/pytest -q
```

Expected: focused tests pass; full suite passes with integration tests deselected by project configuration.

**Step 5: Commit**

```bash
git add agentic_pipeline/adapters/processing_adapter.py agentic_pipeline/orchestrator/orchestrator.py tests/test_processing_adapter.py tests/test_orchestrator.py
git commit -m "fix: persist pipeline book classifications"
```

### Task 3: Integrate, repair the existing row, and audit

**Files:**
- No source changes expected.

**Step 1: Review both branch diffs**

Run `git diff <base>...HEAD` in each worktree and resolve every actionable finding.

**Step 2: Merge both branches to local `main`**

Merge `codex/persist-integrity` in each repository after verification. Do not push without explicit authorization.

**Step 3: Restart the worker**

Restart `com.taylorstephens.agentic-pipeline-worker` so it imports both merged changes.

**Step 4: Repair existing historical gaps once**

Run:

```bash
agentic-pipeline doctor --fix
```

This is expected to backfill the 11 hashes and one book type on pipeline `0888a863-37ac-469b-8ce8-84131acc6a9f`.

**Step 5: Perform a read-only audit**

- Run doctor without `--fix`.
- Check health and validation.
- Confirm all 11 chapter hashes are populated.
- Confirm the book profile fields match the pipeline profile.
- Confirm the record remains `pending_approval`.
- Confirm no chunks, embeddings, approval record, or approval audit entry was created.

---
name: db-schema
description: "Map of this project's SQLite schema. Use before answering any question about tables, columns, relationships, migrations, or delete/cascade behavior — including 'does this column exist' or anything touching src/database.py or agentic_pipeline/db/."
---

# Database Schema Map

Orientation only. This tells you where the schema lives and what will mislead
you. It is not a table reference — read the source for that.

## Canonical source

There is no single schema file. Read all three:

1. `book_ingestion/storage/database.py` (in the **book-ingestion-python** repo,
   not this one) — base `CREATE TABLE IF NOT EXISTS` for `books` and `chapters`.
2. `src/database.py`, `ensure_library_schema()` (~line 133) — library-side
   tables (`chapter_summaries`, `reading_progress`, `bookmarks`, `chunks`,
   `chapters_fts`) plus `ALTER TABLE ADD COLUMN` for tracking columns bolted
   onto `chapters` (`content_hash`, `file_mtime`, `embedding_updated_at`).
3. `agentic_pipeline/db/migrations.py`, `MIGRATIONS` list + `run_migrations()`
   (~line 270) — pipeline-side tables (`processing_pipelines`, autonomy/audit
   tables) plus later index migrations, e.g. the UNIQUE index on
   `chunks(chapter_id, chunk_index)` at line 259 — added separately from the
   `chunks` CREATE TABLE block above it, not visible if you only read the
   CREATE block.

The live database is at `~/Library/Application Support/book-library/library.db`,
overridable via `BOOK_DB_PATH` (src side) / `AGENTIC_PIPELINE_DB` (pipeline
side) — both default to the same file. **The `data/books.db` and
`data/pipeline.db` files this repo previously had checked out were 0-byte
placeholders, never read by runtime code — deleted 2026-08-14
(book-mcp-server#8).**

To verify a specific column actually exists:

    sqlite3 ~/"Library/Application Support/book-library/library.db" "PRAGMA table_info(chapters)"

## Known traps

- **`ON DELETE CASCADE` is now enforced on both connections (fixed
  book-mcp-server#9, PR #10, 2026-08-14).** The DDL declares `ON DELETE
  CASCADE` from `chapters`/`chunks`/`chapter_summaries`/`reading_progress`/
  `bookmarks` back to `books`/`chapters`. Both `src/database.py`'s
  `get_db_connection()` (line 23, used by every MCP tool in `src/tools/`, sets
  the pragma at line 42) and `agentic_pipeline/db/connection.py`'s
  `get_pipeline_db()` (line 15, sets it at line 36) now run `PRAGMA
  foreign_keys = ON` — before the fix, only the pipeline side did, so cascade
  silently didn't fire through the MCP-tool path even though both connect to
  the same `library.db` file. A new MCP delete tool built the normal way
  (`src.database.get_db_connection()` + `DELETE FROM books`) will now cascade
  correctly instead of orphaning rows. Note the pipeline code still doesn't
  trust its own cascade defensively — `orchestrator.py:288-289` and
  `cli.py:1230-1231` hand-delete `chapters_fts`/`chapter_summaries`/`chunks`/
  `chapters` rows before deleting the `books` row rather than relying on it;
  that's now redundant-but-harmless, not a bug.

- **Schema is split across a different repo and two files in this one.**
  `books` and `chapters`' base columns are created in
  **book-ingestion-python**, not here. This repo only ever `ALTER TABLE ADD
  COLUMN`s onto `chapters` (never redefines it), and separately creates its own
  tables via two independent migration runners (`ensure_library_schema()` vs
  `run_migrations()`) that are invoked from different entry points
  (`src/server.py:63` vs `agentic_pipeline/cli.py:30,1458`). Reading only one
  of the three sources under-describes the live schema.

- **Migration-only index, not in the CREATE block.** The UNIQUE index
  `idx_chunks_chapter_index` on `chunks(chapter_id, chunk_index)` lives in
  `agentic_pipeline/db/migrations.py` as a separate `MIGRATIONS` list entry
  (line 259), not inside the adjacent `chunks` `CREATE TABLE` block. A fresh
  database created only via `src/database.py`'s `ensure_library_schema()`
  (which also creates `chunks`, without this index) will be missing it until
  `agentic_pipeline`'s `run_migrations()` also runs.

## What this file is not

Not a table reference. Do not paste table definitions here — read them from
`book_ingestion/storage/database.py`, `src/database.py`, and
`agentic_pipeline/db/migrations.py`. This file tells you where to look and
what will mislead you.

---
title: "PRAGMA foreign_keys inconsistent across DB connections — ON DELETE CASCADE silently doesn't fire on the MCP-tool path"
number: 9
state: closed
labels: [bug]
repo: book-mcp-server
url: https://github.com/tstephx/book-mcp-server/issues/9
created: 2026-08-14
updated: 2026-08-14
---

## Description

The library.db schema declares `ON DELETE CASCADE` from `chapters`/`chunks`/`chapter_summaries`/`reading_progress`/`bookmarks` back to `books`/`chapters`. SQLite only enforces foreign key constraints (including cascades) on a connection where `PRAGMA foreign_keys = ON` has been run — it's a per-connection setting, not a database-level one.

`agentic_pipeline/db/connection.py`'s `get_pipeline_db()` (line 36) sets it. `src/database.py`'s `get_db_connection()` (line 23) — used by every MCP tool in `src/tools/` — does not. Both connect to the same `library.db` by default.

## Impact

Reading the DDL alone implies deleting a book always cleans up its chapters, chunks, summaries, reading progress, and bookmarks. In practice, that only holds true through the pipeline connection path. A future MCP tool that does `DELETE FROM books` via `src.database.get_db_connection()` (the pattern every existing `src/tools/` tool follows) will silently orphan rows in all five dependent tables — no error, no warning.

Notably, the pipeline-side code doesn't even trust its own cascade: `agentic_pipeline/orchestrator/orchestrator.py:288-289` and `agentic_pipeline/cli.py:1230-1231` hand-delete the dependent rows explicitly before deleting the `books` row, rather than relying on `ON DELETE CASCADE` firing.

## Suggested fix

Add `PRAGMA foreign_keys = ON` to `src/database.py`'s `get_db_connection()`, matching `get_pipeline_db()`. Then decide whether the pipeline side's belt-and-suspenders manual deletes are still needed (probably yes, for safety/explicitness) or can be simplified now that cascade is consistently enforced everywhere.

Discovered as a side effect of evaluating a db-schema skill for this repo (#8) — see `.claude/skills/db-schema/SKILL.md`'s "Known traps" section for the full evidence trail (includes the exact line numbers checked against live DDL).

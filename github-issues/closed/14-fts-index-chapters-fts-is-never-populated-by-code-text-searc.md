---
title: "FTS index (chapters_fts) is never populated by code; text_search depends on a manual rebuild_fts_index() call"
number: 14
state: closed
labels: [bug]
repo: book-mcp-server
url: https://github.com/tstephx/book-mcp-server/issues/14
created: 2026-10-30
updated: 2026-11-02
---

## Problem

Nothing in the code populates the `chapters_fts` FTS5 table. `src/database.py:152` creates it empty with the comment "rebuild_fts_index() populates it", but the only two writers have no caller:

- `src/utils/fts_search.py:164` `rebuild_fts_index()` — no caller in this repo, no CLI command, no script
- `src/utils/fts_search.py:220` `sync_fts_chapter()` — no caller

`full_text_search()` (same file, line 54) is called from `src/server.py` by the `text_search` tool, and my-mcp-portfolio's `book_library_client.py:169-173` also queries `chapters_fts` on the shared `library.db`. Both return nothing on a fresh or re-ingested library until someone calls `rebuild_fts_index()` by hand. `docs/archive/2026-02-11-part2-test-results.md:39` records exactly that happening.

book-ingestion-python does not reference `chapters_fts` either (checked 2026-10-30).

## Suggested fix

Either wire `sync_fts_chapter()` into the post-ingest/embedding path (and `rebuild_fts_index()` into `doctor --fix` or a `fts rebuild` CLI command), or document the manual step in `USER-GUIDE.md` and `ref/cli-commands.md`. The first is preferable: a search tool that depends on an undocumented manual step is a silent-failure mode.

## Evidence

Verified against `main` at 8ade59a: `rg -w rebuild_fts_index` and `rg -w sync_fts_chapter` across `src/`, `agentic_pipeline/`, `scripts/`, `tests/` return only the definitions and the `database.py` comment.

Found by the tech-debt scan, pass 2 (vault `Research/Project-Audits/2026-10-28-tech-debt-scan-wave-1-python.md`).

# Populate the FTS index from code, then delete the dead tiers and helpers (book-mcp-server #14, #15, #16, #17)

Lane A2 of the issue-burndown plan (`~/.claude/plans/2026-10-30-issue-burndown-concurrent-sessions.md`). Two PRs: #14 first, because its fix edits `src/utils/fts_search.py` and the sweep in the second PR deletes files beside it.

## Where things stand

Verified 2026-10-30. The canonical checkout `~/Dev/_Projects/book-mcp-server` is at `f942498` and **behind** `origin/main` (`8ade59a`, one commit touching `.claude/settings.json`, `.gitignore`, `.mcp.json`). Base every worktree on `origin/main` after a fetch.

- **#14.** `src/database.py:152` creates `chapters_fts` empty with the comment "rebuild_fts_index() populates it". The only writers are `src/utils/fts_search.py:164` `rebuild_fts_index()` and `:220` `sync_fts_chapter()`; neither has a caller in this repo, in `book-ingestion-python`, or in `my-mcp-portfolio` (which also queries `chapters_fts`, `src/domain/book_library_client.py:169-173`, on the shared `library.db`). `full_text_search()` (`fts_search.py:54`) backs the `text_search` tool in `src/server.py`. `docs/archive/2026-02-11-part2-test-results.md:39` records `text_search` returning nothing until someone ran the rebuild by hand.
- **#15.** `agentic_pipeline/cli.py:480` declares `--recover`; `stuck()` never reads it; `agentic_pipeline/health/stuck_detector.py` has only `detect()`. Documented as working in `docs/USER-GUIDE.md:248`, `ref/cli-commands.md:24`, and the debug-pipeline skill.
- **#16.** `src/utils/cache.py`: `get/set_embeddings` (74, 92) and `get/set_summary_embeddings` (176, 194) are tests-only; `src/server.py:189-193` `clear_cache("embeddings")` / `("summary_embeddings")` clear those empty tiers; `invalidate_chunk_embeddings` (165) has no caller.
- **#17.** Eight symbols with no caller and no test, listed in the issue with paths, plus `agentic_pipeline/mcp_server.py:22` `approve_book_tool(adjustments=)` whose docstring (line 34) promises application before ingestion while `approval/actions.py:373-406` only writes the dict to the audit row.
- **Shared database.** `library.db` is read by book-mcp-server, book-ingestion-python and my-mcp-portfolio (path per `BOOK_DB_PATH`; see this repo's CLAUDE.md). Nothing here changes the schema. Test FTS population against a copy of the DB, not the live file.
- **Gate.** `python -m pytest tests/ -v` (CLAUDE.md:79, 140). Hosted CI is dormant (CLAUDE.md:147); the gate is the tracked `scripts/pre-push-verify.sh` pre-push hook. Worktrees run `uv sync --locked` from `.config/wt.toml`; do not bypass it with pip.

## Task 1: `chapters_fts` is filled by the ingest path and rebuildable from the CLI (#14)

1. `git -C ~/Dev/_Projects/book-mcp-server fetch origin`, then preflight (`/concurrent-session-preflight tstephx/book-mcp-server 14`), claim, `wt switch -c book-mcp-server-fts-populate-14`.
2. Find where a chapter's content is written or embedded in the approve/embed path (`approval/actions.py` around line 229, the `_complete_approved` flow) and in `src/utils/embedding_sync.py`. Those are the places `sync_fts_chapter(chapter_id, content, title)` belongs.
3. Tests first: (a) after an approve/embed on a fixture DB, `chapters_fts` contains the chapter and `full_text_search()` finds it; (b) `rebuild_fts_index()` is idempotent on a populated table (no duplicate rows, run it twice); (c) the existing `tests/test_ensure_library_schema.py:124` still passes.
4. Wire `sync_fts_chapter` into the write path and expose `rebuild_fts_index` as a CLI command (match the existing click group in `agentic_pipeline/cli.py`; `doctor --fix` is the natural home if the doctor already repairs library state, otherwise a new `fts rebuild` command). Document the command in `ref/cli-commands.md` and mention the one-time rebuild for existing libraries in `docs/USER-GUIDE.md`.
5. Run the full suite and the pre-push hook. One PR, `Closes #14`. Merge, push, confirm on `origin`, release the claim.

### Constraints carried over
- Never modify `library.db` schema; test population on a copy. The DB is shared with two other repos.
- Do not install into the worktree venv with pip; `.config/wt.toml` runs `uv sync --locked`.
- Stage by name; the untracked `github-issues/` in the canonical checkout is not yours.
- Do not run `wt merge` with a dirty tracked tree (scripts#52, in flight elsewhere); commit or discard `.serena/project.yml` first if Serena rewrote it.

### Caution
Re-derive state first: `git fetch`, `git log --oneline -5 origin/main`, and the four issues' state; this prompt goes stale same-day. Run `/concurrent-session-preflight` before the first write and again before the final push. Before creating any worktree, run `python -m pytest tests/ -q` against the `origin/main` tip in the canonical checkout (after `git pull --ff-only`) and fix or flag a pre-existing failure first.

## Task 2: delete the no-op flag, the unfilled cache tiers, and the dead helpers (#15, #16, #17)

1. Preflight for #15, claim, `wt switch -c book-mcp-server-dead-code-sweep-15` based on `origin/main` after Task 1 merged.
2. #15: delete the `--recover` option at `cli.py:480` and the three doc mentions, or implement it by resetting stuck pipelines through the existing `reject --retry` path. Choose delete unless the implementation is under 30 lines with a test; say which in the PR.
3. #16: delete the chapter and summary embedding tiers from `cache.py`, their `clear_cache` branches in `server.py:189-193` and `embedding_sync.py:301`, and their tests. Give the chunk tier a targeted invalidation only if a caller needs one now; otherwise delete `invalidate_chunk_embeddings` too and note it.
4. #17: delete the eight symbols the issue lists (whole `src/utils/excerpt_utils.py`; `get_chapter_excerpt`; `batch_export_chapters`; `database_transaction`, `batch_processing_context`, `error_context`, keeping `embedding_model_context`; `tool_schemas.py:171` `validate_book_id`; `ProcessingAdapter.get_processing_status`). Also check `src/utils/validators.py:16` `validate_book_id`, which the issue notes is uncalled too. For `adjustments=`, reword the docstring at `mcp_server.py:34` to "recorded in the audit trail" and drop the parameter from the tool signature unless a caller passes it. Update `ref/mcp-tools.md:168` and `ref/module-map.md` where they name deleted symbols. Leave the issue's "also worth a look" MINORs alone unless they fall out of the same edit.
5. Full suite, pre-push hook. One PR, `Closes #15`, `Closes #16`, `Closes #17`. Merge, push, confirm, release, then `save_session` once for the lane.

### Constraints carried over
- Never modify `library.db` schema; test population on a copy. The DB is shared with two other repos.
- Do not install into the worktree venv with pip; `.config/wt.toml` runs `uv sync --locked`.
- Stage by name; the untracked `github-issues/` in the canonical checkout is not yours.
- Do not run `wt merge` with a dirty tracked tree (scripts#52, in flight elsewhere); commit or discard `.serena/project.yml` first if Serena rewrote it.

### Caution
Re-derive state first: `git fetch`, `git log --oneline -5 origin/main`, and the four issues' state; this prompt goes stale same-day. Run `/concurrent-session-preflight` before the first write and again before the final push. Before creating any worktree, run `python -m pytest tests/ -q` against the `origin/main` tip in the canonical checkout (after `git pull --ff-only`) and fix or flag a pre-existing failure first.

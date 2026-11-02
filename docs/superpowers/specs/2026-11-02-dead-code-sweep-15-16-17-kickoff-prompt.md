# Delete the no-op `--recover`, the unfilled embedding cache tiers, and the dead helpers (book-mcp-server #15, #16, #17)

Lane A2, Task 2. Task 1 (#14) is done: PR #18, squash `ee24e84` on `origin/main`. This prompt replaces "Task 2" of `docs/superpowers/specs/2026-10-30-issue-burndown-a2-fts-and-dead-code-kickoff-prompt.md`; read that section for the per-issue steps and keep this file for what has changed since.

## Where things stand

Verified 2026-11-02 on `origin/main` = `ee24e84`; issues #15, #16, #17 all OPEN, no branch, PR or claim for them.

- **Steps.** Follow Task 2 of the 2026-10-30 prompt (steps 1–5): #15 delete `--recover` (`agentic_pipeline/cli.py:479-480`, unless a working implementation is under 30 lines with a test) plus its doc mentions (`docs/USER-GUIDE.md`, `ref/cli-commands.md`, the debug-pipeline skill); #16 delete the chapter and summary embedding tiers in `src/utils/cache.py` (`get/set_embeddings` :74/:92, `get/set_summary_embeddings` :176/:194, `invalidate_chunk_embeddings` :165 if uncalled), their `clear_cache` branches in `src/server.py` (~:171-195) and `src/utils/embedding_sync.py`, and their tests; #17 delete the eight symbols listed in the issue, plus `src/utils/validators.py` `validate_book_id` if uncalled, and fix `approve_book_tool(adjustments=)` at `agentic_pipeline/mcp_server.py:26-37`. Re-grep every symbol for callers before deleting; line numbers drift.
- **Changed since the old prompt.** `ref/cli-commands.md` and `CLAUDE.md` now say 32 commands (`fts-rebuild` added). Deleting `--recover` removes a flag, not a command, so the count stays 32. New files from #14: `agentic_pipeline/library/fts_index.py`, `tests/test_fts_population.py`; do not touch them.
- **One PR**, `Closes #15`, `Closes #16`, `Closes #17`. Say in the PR whether `--recover` was deleted or implemented, and whether `invalidate_chunk_embeddings` was deleted.
- **Not this session's job (user-run, listed so nobody repeats them):** `agentic-pipeline fts-rebuild` against the live `library.db` (back up first), and `launchctl kickstart -k gui/$(id -u)/com.taylorstephens.agentic-pipeline-worker` after `main` is pulled in the canonical checkout. Do not run the rebuild against the live DB.
- **Environment facts, observed once, re-verify:** `book_ingestion` is not installed in the worktree venv, so `agentic_pipeline/adapters/processing_adapter.py` cannot be imported in a bare test (other tests mock it via `sys.modules`; see `tests/test_processing_adapter_chunks.py`). The PostToolUse `ruff format` hook reflows the whole touched file; `processing_adapter.py` has pre-existing unformatted hunks (e.g. `_persist_book_profile`), so after removing `get_processing_status` revert any reflow outside your deletion (`git diff` that file before committing).

## What this session does

Execute Task 2 as above. Branch `book-mcp-server-dead-code-sweep-15`, worktree via `wt switch -c book-mcp-server-dead-code-sweep-15` from the canonical checkout `~/Dev/_Projects/book-mcp-server` after `git pull --ff-only`. Tests first where behavior changes (e.g. `clear_cache("embeddings")` rejected or removed); deleting dead code needs the suite green before and after, not new tests. Then: full suite, pre-push hook, PR, merge, confirm on `origin`, release the claim, and `save_session` once for the lane.

## Constraints carried over

- Never modify the `library.db` schema; the DB is shared with `book-ingestion-python` and `my-mcp-portfolio`. Nothing here needs the live DB.
- Do not install into the worktree venv with pip; `.config/wt.toml` runs `uv sync --locked`.
- Stage by name; the untracked `github-issues/` in the canonical checkout is not yours.
- Do not run `wt merge` with a dirty tracked tree (scripts#52); commit or discard `.serena/project.yml` first if Serena rewrote it. Do not remove the worktree from inside its own session: `wt-live-session-guard` blocks it while the session's child processes hold the directory; end the session, then `wt remove`.
- Leave the issues' "also worth a look" MINORs alone unless they fall out of the same edit.

## Caution

Re-derive state first: `git fetch`, `git log --oneline -5 origin/main`, and the state of #15, #16, #17; this prompt goes stale same-day. Run the `concurrent-session-preflight` skill (`tstephx/book-mcp-server 15`, then claim) before the first write and again before the final push. Before creating the worktree, run `python -m pytest tests/ -q` on the `origin/main` tip in the canonical checkout (about 2–4 minutes; baseline at `ee24e84` is 672 passed, 3 skipped) and fix or flag a pre-existing failure first. Hosted CI is dormant; the gate is `scripts/pre-push-verify.sh` (lint + tests).

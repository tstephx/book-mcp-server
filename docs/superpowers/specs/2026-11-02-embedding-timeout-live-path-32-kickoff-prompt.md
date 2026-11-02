# Enforce the embedding timeout on the live EMBEDDING path (book-mcp-server #32)

One issue, one PR. Follow-up from #22 (PR #31). Make `EMBEDDING_TIMEOUT_SECONDS` actually apply to the code that runs EMBEDDING, then remove the dead method that currently owns it.

## Where things stand

Verified 2026-11-02 against `origin/main` at `a3ff31f`; issue #32 is OPEN. PR #31 (docs-only, fixes the `ref/` wording) may or may not be merged yet; check, because the `ref/pipeline-architecture.md` timeout row it edits needs updating again here. Re-grep every line number below before trusting it.

- **Dead owner of the timeout.** `agentic_pipeline/orchestrator/orchestrator.py:294` `Orchestrator._run_embedding` wraps `processing_adapter.generate_embeddings` in a `ThreadPoolExecutor` with `future.result(timeout=self.config.embedding_timeout)` and raises `PipelineTimeoutError`. No production caller. `config.embedding_timeout` is defined at `agentic_pipeline/config.py:23` (env `EMBEDDING_TIMEOUT_SECONDS`, default 300).
- **Live path, no timeout.** `agentic_pipeline/approval/actions.py:206` `_complete_approved` sets EMBEDDING, calls `ProcessingAdapter(db_path=db_path).generate_embeddings(book_id=...)` directly (about `:236`), and on a non-success result or exception moves to NEEDS_RETRY with `error_details={"failed_in": "embedding", ...}`. Callers: `approve_book()` (daemon thread via `_run_embedding_background`, or synchronous with `background=False` as the CLI does at `cli.py:87`), and `Orchestrator._complete_approved` (`orchestrator.py:603`) used by `_retry_one` (`:669`) and `run_worker` (`:733`).
- **Retry plumbing exists.** `_retry_one` checks `_get_failed_stage` (`orchestrator.py:618`) and, for `"embedding"`, goes straight back to `_complete_approved`. A timeout that records `failed_in: "embedding"` is therefore picked up with no new retry code.
- **Only test of the timeout.** `tests/test_orchestrator_timeouts.py:43-51` (exercises the dead method); `tests/test_orchestrator.py:226` also patches it.
- **Crash recovery gap (verified).** `run_worker` startup calls `repo.reset_stale_processing()` (`db/pipelines.py:442`), which only resets `PROCESSING` to `DETECTED`. A worker killed mid-embed (daemon thread dies with the process) leaves the book in EMBEDDING; only `health/stuck_detector.py` (EMBEDDING threshold 600s) notices, and it detects without recovering. Decide whether that is in scope for this PR or a separate issue (see below).
- **Not verified, check before relying on it.** Whether `generate_embeddings` is idempotent for a book on retry (`processing_adapter.py:232`), and what happens if a timed-out call is still running when the retry starts. A Python thread cannot be killed, so the timed-out call keeps going in the background.

## What this session does

Branch via `wt switch -c book-mcp-server-embedding-timeout-32`.

1. Write the failing test first (`tests/`): `_complete_approved` with `generate_embeddings` patched to sleep past a small `embedding_timeout` ends in NEEDS_RETRY with `failed_in == "embedding"` and a timeout message, and does not hang. Cover the synchronous and `background=False` entry points through the one shared function.
2. Read `generate_embeddings` and decide, with evidence, whether a late-finishing orphaned call can corrupt or double-write chunks when a retry starts. If it can, make the write path safe or document the limit in the PR; do not just hope.
3. Implement the timeout inside `approval.actions._complete_approved` (executor plus `future.result(timeout=...)`, `executor.shutdown(wait=False)`). `_complete_approved` takes only `db_path`, so it needs the timeout value: read it from `PipelineConfig` the way the rest of `agentic_pipeline/` does rather than adding a new env var.
4. Delete `Orchestrator._run_embedding` and move/replace its tests so the timeout is still covered on the real path. Remove the now-unused imports.
5. Update `ref/pipeline-architecture.md` (the `_run_embedding` row and the timeout config row), and `ref/module-map.md` if it lists the method.
6. Crash recovery: if it is a small, safe change (for example extending startup recovery to EMBEDDING, with the CAS-guarded `update_state` discipline), include it with tests. Otherwise file a separate issue and say so in the PR. Either way, state the choice.
7. Run `make lint && make test`, open one PR whose body says `Closes #32`.

## Constraints carried over

- Push to `main` only via branch + PR. Do not merge without being asked.
- Do not change embedding results, chunking, or the model. Only add the timeout and its failure handling.
- Don't hold an open sqlite write transaction while waiting on the embedding call (it lock-starves other writers).
- Never `git add` broadly; stage files by name. No secrets, no `.env`.
- Use `.venv` tooling as in `CLAUDE.md`; in the main checkout never run `uv sync`.

## Caution

- Re-derive current state first (`git log --oneline -10`, `gh issue view 32`, `gh pr view 31`, `git status`). This prompt goes stale same-day.
- Run the `concurrent-session-preflight` skill before starting and again immediately before the final push. Peer sessions were active in this repo when this prompt was written.
- Run the repo gate once on the base tip before branching (`make lint && make test`, venv activated) and flag any pre-existing failure instead of inheriting it. Run it from the worktree, not the canonical checkout.

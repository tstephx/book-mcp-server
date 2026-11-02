# Recover books stranded in EMBEDDING after a worker crash (book-mcp-server #34)

One issue, one PR. Split from #32 (PR #36). Make worker startup move books that a crash left in `EMBEDDING` to `NEEDS_RETRY`, so the existing retry loop finishes them.

## Where things stand

Verified 2026-11-02 against `origin/main` at `3634754`; issue #34 is OPEN, no open PR touches it. Re-grep every line number before trusting it.

- **The gap.** `Orchestrator.run_worker` (`agentic_pipeline/orchestrator/orchestrator.py:681`) calls `self.repo.reset_stale_processing()` at `:690`. That method (`agentic_pipeline/db/pipelines.py:442`) only resets `PROCESSING` to `DETECTED` (staleness via `julianday(updated_at)` against `STALE_PROCESSING_SECONDS = 600`, `pipelines.py:16`). `approve_book()` runs EMBEDDING on a daemon thread, so a killed worker leaves the book in `EMBEDDING` forever.
- **Detect-only today.** `agentic_pipeline/health/stuck_detector.py:11` has `"EMBEDDING": 600`, times `stuck_multiplier` 2.0 = 1200s before a book is reported. It reports, it never recovers.
- **Retry plumbing exists.** The live path `approval/actions.py` `_complete_approved` (about `:288-370`) already writes `NEEDS_RETRY` with `error_details={"embedding_error": ..., "failed_in": "embedding"}`. `_retry_one` in `orchestrator.py` sends `failed_in == "embedding"` straight back to `_complete_approved`. So recovery needs no new retry code, only the state move with that `error_details`.
- **Concurrency primitive.** `PipelineRepository.update_state(..., expected_state=)` (`pipelines.py:70`) raises `ConcurrentModificationError` if another actor moved the record. Use it, and catch that error per row.
- **Threshold constraint (from the issue).** The staleness cutoff must exceed `EMBEDDING_TIMEOUT_SECONDS` (default 300, `config.py:12`; `embedding_timeout_from_env()`), or a live CLI/MCP embed gets reset mid-flight. Note `updated_at` is set when EMBEDDING begins and not touched during the call, so cutoff = timeout + margin. Decide whether the margin is derived from the env value or a constant above the 600s `STALE_PROCESSING_SECONDS`; state the choice in the PR.
- **Existing tests to model on.** `tests/test_pipelines.py:137-207` (stale PROCESSING reset: fresh-not-reset, stale-reset, NULL `updated_at`).
- **Related open gap** (memory note `project_embedding_timeout_live_path`): a book whose earlier attempt committed chunk rows with NULL embeddings is caught at completion by `_unembedded_chapters`, not here. Out of scope.

## What this session does

1. Add a repo method (name it by the neighbors' style, e.g. `reset_stale_embedding(cutoff_seconds)`) that selects `EMBEDDING` rows with `updated_at` older than the cutoff, and for each calls `update_state(id, NEEDS_RETRY, error_details={"failed_in": "embedding", "embedding_error": "worker crashed mid-embedding (recovered at startup)"}, expected_state=PipelineState.EMBEDDING)`, skipping on `ConcurrentModificationError`. Return the count. Do not hold a write transaction across rows.
2. Call it from `run_worker` right after `reset_stale_processing()`, log the count like the existing `StartupRecovery` line, with the cutoff derived from `self.config.embedding_timeout`.
3. Tests first: fresh EMBEDDING row untouched; stale row becomes NEEDS_RETRY with `failed_in == "embedding"`; NULL `updated_at` untouched; a row moved by another actor mid-sweep is skipped, not raised; `_retry_one` picks the recovered row up (use the existing retry test fixtures).
4. Update `ref/pipeline-architecture.md` (worker startup recovery / state notes) and `ref/module-map.md` if it lists `pipelines.py` methods. Run the `sync-ref-docs` skill rather than editing by guess.
5. Verify against the DB itself, not only mocks: build a temp DB with a stranded EMBEDDING row and run the sweep. Do not touch the live library DB.

## Constraints carried over

- Branch + PR only; do not merge unless asked.
- Do not change embedding results, chunking, the model, or `reset_stale_processing`'s PROCESSING behavior.
- Don't hold an open sqlite write transaction while doing other work (it lock-starves the CLI subprocess).
- Live worker/DB actions (kickstart, running against the real library) are out of scope; if wanted, use the `/live-state-claim` skill and ask first.
- Never `git add` broadly; stage by name. No secrets, no `.env`. Use `.venv` tooling; never `uv sync` in the main checkout.

## Caution

- Re-derive current state first (`git log --oneline -10`, `gh issue view 34`, `git status`). This prompt goes stale same-day.
- Run the `concurrent-session-preflight` skill before starting and again immediately before the final push. A peer session (`book-mcp-server-2d`) and a sibling worktree (`...-spot-checks-pipeline-id-24`) were active in this repo when this was written.
- Run the repo gate once on the base tip before changes (`make lint && make test`, venv activated, from this worktree) and flag any pre-existing failure instead of inheriting it.
- Add `Refs #34` (not a closing keyword) in the prompt-only commit; the PR may use `Closes #34`.

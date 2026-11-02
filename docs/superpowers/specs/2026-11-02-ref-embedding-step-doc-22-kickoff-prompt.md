# Make `ref/pipeline-architecture.md` and the code agree on what runs the EMBEDDING step (book-mcp-server #22)

One issue, one PR. Follow-up from #17 (PR #20). Pick the doc-only fix or the delete-dead-code fix, state the choice in the PR, and land it.

## Where things stand

Verified 2026-11-02 against `origin/main` at `9580180`; issue #22 is OPEN, no branch or PR for it. Re-grep every line number below before trusting it.

- **The doc claim.** `ref/pipeline-architecture.md:68`, in the "Private State Machine Steps" table: `` `_run_embedding` | EMBEDDING → COMPLETE ``.
- **The method.** `agentic_pipeline/orchestrator/orchestrator.py:294` `Orchestrator._run_embedding`. No caller in `agentic_pipeline/` outside its own definition. Only tests reference it: `tests/test_orchestrator.py:226` (patched out) and `tests/test_orchestrator_timeouts.py:43-51` (asserts `PipelineTimeoutError` when the adapter hangs past `embedding_timeout`).
- **What actually runs EMBEDDING.** `agentic_pipeline/approval/actions.py`: `approve_book()` spawns a daemon thread targeting `_run_embedding_background` (`:280` def, `:439` spawn). `orchestrator._complete_approved` (`orchestrator.py:603`) delegates to `approval.actions._complete_approved` and is used by the worker loop (`:669`).
- **Premise correction.** Issue #22 and `CLAUDE.md` say embedding runs "inline" in `approve_book()`. `ref/module-map.md:84` and the code say `approve_book()` is **non-blocking** and embeds on a background thread (`ref/pipeline-architecture.md:89` also says "inline or spawn `_run_embedding_background`, according to `background`"). Read `approve_book()` and decide which description is true; do not copy "inline" into the fix.
- **Same table, possibly also stale.** The `_complete_approved` row says "called by background thread". Check it against `orchestrator.py:669` and `approval/actions.py` while the table is open.

## What this session does

Branch via `wt switch -c book-mcp-server-ref-embedding-22`.

1. Re-derive the facts above (grep callers of `_run_embedding`, read `approve_book()` and `_complete_approved`).
2. Choose and state in the PR one of:
   - (a) Recommended: doc-only. Fix the `:68` row (and the `_complete_approved` row if wrong) to name the real EMBEDDING path. Smallest change, no code risk.
   - (b) Delete `_run_embedding` and its tests. If so, keep the timeout guarantee: `test_orchestrator_timeouts.py` is the only test for the embedding timeout, so confirm the real path (`_run_embedding_background` / adapter) has equivalent coverage or say plainly that the guarantee is being dropped.
3. If you change code or the table, update `ref/module-map.md` and `CLAUDE.md` wording ("inline") where it contradicts what you verified.
4. Run the lint/test gate, open one PR that mentions `#22` in the body (closing keyword is fine in the PR body, not in a kickoff-only commit).

## Constraints carried over

- Push to `main` only via branch + PR. Do not merge without being asked.
- Docs-only is the default. Do not touch the embedding behavior itself.
- Never `git add` broadly; stage files by name. No secrets, no `.env`.

## Caution

- Re-derive current state first (`git log --oneline -10`, `gh issue view 22`, `git status`). This prompt goes stale same-day.
- Run the `concurrent-session-preflight` skill before starting and again immediately before the final push; a peer session `book-mcp-server-2d` was active in this repo when this prompt was written.
- Run the repo's gate once on the base tip before branching (`make lint && make test`, venv activated); flag any pre-existing failure instead of inheriting it. Run the gate from the worktree, not the canonical checkout (the pre-push hook only runs `uv sync` from linked worktrees).

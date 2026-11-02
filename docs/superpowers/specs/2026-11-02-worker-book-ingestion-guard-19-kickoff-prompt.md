# Stop the worker crash-looping silently when `book_ingestion` is missing (book-mcp-server #19)

One issue, one PR: make the missing-package failure loud in `scripts/run-worker.sh`, and stop the pre-push hook from removing the package from the canonical checkout's venv.

## Where things stand

Verified 2026-11-02; issue #19 OPEN, no branch or PR for it. The dead-code sweep (#15/#16/#17) already merged as PR #20 (`e67db16`); base on `origin/main` after it, and re-grep any line number cited here, since that PR deleted code and edited `CLAUDE.md` and `docs/`.

- **Cause.** `book_ingestion` is deliberately outside `pyproject.toml` deps (`pyproject.toml:40`, a local editable sibling `~/Dev/_Projects/book-ingestion-python`). `scripts/pre-push-verify.sh:66-67` runs `uv sync --locked --python 3.12 --extra dev`, which removes it from whichever checkout's `.venv` is pushing. The launchd worker (`com.taylorstephens.agentic-pipeline-worker`, KeepAlive) runs the **canonical** checkout's `.venv/bin/agentic-pipeline` via `scripts/run-worker.sh` (last line: `exec "$REPO_DIR/.venv/bin/agentic-pipeline" worker`). So a `git push` from `~/Dev/_Projects/book-mcp-server` itself breaks the worker; a push from a worktree only touches that worktree's own venv and is harmless.
- **Symptom.** `ModuleNotFoundError: No module named 'book_ingestion'` from `agentic_pipeline/adapters/processing_adapter.py:15`, repeated in `~/Library/Logs/agentic-pipeline-worker.err`. 6,523 crash runs on 2026-11-02 and a prior recurrence 2026-08-14 went unnoticed because a running worker keeps its interpreter in memory and only fails on the next restart.
- **Manual fix (already applied once).** `uv pip install -e /Users/taylorstephens/Dev/_Projects/book-ingestion-python --no-deps --python .venv/bin/python3`, then `launchctl kickstart -k gui/$(id -u)/com.taylorstephens.agentic-pipeline-worker`. Plain `pip` does not work here (no pip in the venv; the shell `pip` alias hits system Python). `--no-deps` matters: it avoids disturbing the pinned `constraint-dependencies`.
- **Other consumers of the canonical venv.** `.config/wt.toml` comments say `periodical-parser` and `book-ingestion-python` launch servers from this repo's main `.venv` by absolute path; observed once, re-verify before relying on it. They are hurt by the same `uv sync`.
- **Hook resolution.** The installed hook is a symlink `.git/hooks/pre-push -> ../../scripts/pre-push-verify.sh` in the shared git dir, so it resolves to the **canonical** checkout's copy of the script, not a worktree's. A change to the hook in this PR takes effect only after it is merged and the canonical checkout pulls. Do not push anything from the canonical checkout in the meantime.

## What this session does

Branch `book-mcp-server-worker-guard-19` via `wt switch -c book-mcp-server-worker-guard-19`. Tests first where a script can be exercised.

1. **`scripts/run-worker.sh`.** Before the `exec`, check that the venv can import `book_ingestion` (`"$REPO_DIR/.venv/bin/python" -c 'import book_ingestion'`). On failure: print one actionable line to stderr (package name, the exact `uv pip install -e ... --no-deps --python .venv/bin/python3` command, and the kickstart command), `sleep` a few seconds to slow the launchd restart rate, then `exit 1`. Keep the existing env setup unchanged. Allow the interpreter path to be overridden by an env var (e.g. `WORKER_PYTHON`) so a test can point it at a fake.
2. **`scripts/pre-push-verify.sh`.** Choose and state in the PR one of:
   - (a) Recommended: when the push comes from the main checkout (`git rev-parse --git-dir` equals `--git-common-dir`), do not mutate the venv. Skip `uv sync`, or use a non-mutating check if `uv sync --check` exists in the installed uv (verify with `uv sync --help`; do not assume), and run `make lint && make test` against the existing venv.
   - (b) Run the sync as today, then re-install the editable sibling if it was importable before the sync. This leaves a window where the worker venv lacks the package, so it is the weaker option.
   Keep everything else in the hook (portability check, `unset GIT_*`, lint + test) intact.
3. **Tests.** Cover run-worker.sh with a fake interpreter: missing package exits non-zero with the message on stderr and does not `exec` the worker; present package execs. Cover the hook change at least with a shell-level test or a documented manual check; if the hook cannot be unit-tested cheaply, say so in the PR and give the manual steps.
4. **Docs.** Update the "Pre-push gate" paragraph in `CLAUDE.md` and `docs/shared-mcp-provider.md` or the worker setup notes where they describe the venv, with the new behavior and the manual reinstall command. If `ref/` mentions `run-worker.sh`, update it.
5. **Verify without touching the live worker.** Use a temp venv or the fake interpreter. Do not run `uv sync` in the canonical checkout and do not kickstart the live worker unless you re-verify afterward with `launchctl print gui/$(id -u)/com.taylorstephens.agentic-pipeline-worker` (expect `state = running`) and `.venv/bin/python -c "import book_ingestion"`.
6. **Ship.** Full suite, pre-push hook (it runs from the worktree and syncs the worktree's venv, which is fine), one PR `Closes #19`, merge, confirm on `origin`, then in the canonical checkout `git pull --ff-only` and confirm the worker venv still imports `book_ingestion`. Release the claim. `save_session` once.

## Constraints carried over

- Never modify the `library.db` schema; nothing here needs the live DB.
- Do not install into any worktree venv with pip; `.config/wt.toml` runs `uv sync --locked`. The only `uv pip install` in this task is the documented reinstall into the canonical venv, and only if you find it broken.
- Do not push from the canonical checkout (`~/Dev/_Projects/book-mcp-server`) until this PR is merged; its `uv sync` is the bug.
- Stage by name; the untracked `github-issues/` in the canonical checkout is not yours.
- Do not run `wt merge` (this repo is PR-based; `.config/wt.toml` blocks it) and do not remove your own worktree from inside its session (`wt-live-session-guard`); end the session, then `wt remove`.
- Do not bypass hooks with `--no-verify` or `--no-hooks`.

## Caution

Re-derive state first: `git fetch`, `git log --oneline -5 origin/main`, `gh issue view 19`, and that #15/#16/#17 are closed. Run the `concurrent-session-preflight` skill (`tstephx/book-mcp-server 19`, branch `book-mcp-server-worker-guard-19`, then claim) before the first write and again before the final push. Before creating the worktree, run `python -m pytest tests/ -q` on the `origin/main` tip in the canonical checkout (about 2–4 minutes; last known baseline 672 passed, 3 skipped) and fix or flag a pre-existing failure first. Hosted CI is dormant; the gate is the pre-push hook (lint + tests).

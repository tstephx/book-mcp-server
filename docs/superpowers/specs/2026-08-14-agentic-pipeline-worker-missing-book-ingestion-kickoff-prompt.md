# Kickoff: install the missing `book_ingestion` dependency so agentic-pipeline-worker can run

## Where things stand

Spun out of `~/Dev/scripts/2026-08-14-launchagent-health-audit-kickoff-prompt.md`
(Task 2, `scripts` repo, "launchagent-health-audit" branch, 2026-08-14) — that
session triaged 9 launchd jobs `launchagent-doctor.py --verbose` flagged with
a nonzero last exit code. One of them, `com.taylorstephens.agentic-pipeline-worker`
(last exit code 1), is this project's worker and needs project-specific
investigation the scripts-repo session wasn't scoped for.

**Confirmed error** (`~/Library/Logs/agentic-pipeline-worker.err`, tail):

```
ModuleNotFoundError: No module named 'book_ingestion'
  File ".../agentic_pipeline/adapters/processing_adapter.py", line 15, in <module>
    from book_ingestion import BookIngestionApp, ProcessingMode
```

The venv (`.venv/lib/python3.12/site-packages/`) has `ebooklib` but no
`book_ingestion` package at all. This isn't a broken install — `pyproject.toml`
documents it as deliberate:

```
# converters/ are guarded by pytest.importorskip("book_ingestion").
# "book-ingestion",  # Install separately: pip install -e /path/to/book-ingestion-python
...
# NB book-ingestion is intentionally absent (installed separately as an editable
```

(exact line numbers not re-cited here — re-grep `book_ingestion` in
`pyproject.toml` when this executes, in case it's moved.)

So the worker's `cli.py` → `orchestrator.py` → `adapters/processing_adapter.py`
import chain has a hard, non-optional dependency on `book_ingestion` at
runtime, but the project's own packaging comments say it's supposed to be
installed separately via `pip install -e /path/to/book-ingestion-python` —
and evidently that install step was never done (or was done once, then lost
— e.g. a venv rebuild) in whatever environment `com.taylorstephens.agentic-pipeline-worker.plist`
runs under (`ProgramArguments`: `/bin/bash
/Users/taylorstephens/Dev/_Projects/book-mcp-server/scripts/run-worker.sh`,
`KeepAlive = true`, `RunAtLoad = true` — it's a long-running daemon, not a
scheduled job, so it's been crash-looping since whenever this gap opened).

**Not yet located in this session:** where `book-ingestion-python` actually
lives on disk (its repo). The scripts-repo triage session didn't search for
it — start there.

## What this session does

1. Re-derive current state first (see Caution) — this gap may already be
   fixed, or the plist/error may have changed.
2. Locate the `book-ingestion-python` repo (check `~/Dev/_Projects/`,
   `~/Dev/`, and any workspace registry this project's CLAUDE.md points to).
3. Decide the right install: `pip install -e /path/to/book-ingestion-python`
   into `book-mcp-server`'s `.venv` is what the packaging comment
   prescribes — confirm that's still correct (check whether
   `book-ingestion-python` itself has changed shape, e.g. renamed package,
   moved entry points) before just running it.
4. Install it, then verify the worker actually starts clean:
   `/Users/taylorstephens/Dev/_Projects/book-mcp-server/.venv/bin/agentic-pipeline
   worker` (or however `run-worker.sh` invokes it — read that script first)
   no longer raises `ModuleNotFoundError`.
5. Restart the LaunchAgent and confirm a clean run:
   `launchctl kickstart -k gui/501/com.taylorstephens.agentic-pipeline-worker`,
   then check `~/Library/Logs/agentic-pipeline-worker.err` for a fresh,
   clean start (not a repeat of the same traceback), and re-run
   `python3 ~/Dev/scripts/launchagent-doctor.py --verbose` to confirm it
   drops out of "Nonzero last exit code" once the daemon has been up long
   enough to report a fresh exit status (or stays absent from that section
   entirely if it's now running cleanly and hasn't exited again).
6. If `book-ingestion-python` can't be found, or installing it surfaces its
   own compatibility problems (version mismatch, missing deps of its own),
   stop and report rather than guessing around it — this kickoff assumed
   the fix is "just install the sibling package," per the project's own
   packaging comment, but that assumption needs verifying, not blindly
   executing.

## Constraints carried over

- Only touch this project (`book-mcp-server`) and, if found, the
  `book-ingestion-python` repo it depends on — don't touch other projects.
- `com.taylorstephens.agentic-pipeline-worker.plist` is a `com.taylorstephens.*`
  job — safe to `launchctl kickstart`/reload after fixing the underlying
  dependency, per the parent kickoff's carried-over constraints.
- Don't touch `com.apple.*` or other vendor launchd jobs.

## Caution

Re-derive current state before trusting anything above:

- `tail -50 ~/Library/Logs/agentic-pipeline-worker.err` — confirm the
  `ModuleNotFoundError: No module named 'book_ingestion'` is still the
  live failure, not something that's changed.
- `launchctl print gui/501/com.taylorstephens.agentic-pipeline-worker | grep -i "last exit\|state"`
  — confirm current state.
- `git -C ~/Dev/_Projects/book-mcp-server log -5 --oneline` and `git status`
  — confirm nothing about the packaging setup changed since this was
  written (2026-08-14).
- `ListAgents` — this workspace routinely runs multiple concurrent
  sessions; run the `concurrent-session-preflight` skill before making
  changes, since another session could plausibly already be on this.

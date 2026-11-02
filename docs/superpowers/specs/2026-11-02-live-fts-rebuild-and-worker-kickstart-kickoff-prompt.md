# Backfill `chapters_fts` on the live library and redeploy the launchd worker (book-mcp-server #14 rollout)

Operational follow-through for PR #18 (squash `ee24e84`, closes #14). The code is merged; the live `library.db` and the running worker have not picked it up. No code changes and no PR: this session runs commands against live state and reports.

## Where things stand

Observed 2026-11-02, re-verify before acting:

- **Live DB:** `~/Library/Application Support/book-library/library.db`, about 3.1 GB, WAL mode (`-wal`/`-shm` sidecars present). Shared by book-mcp-server, `book-ingestion-python` and `my-mcp-portfolio` (which also queries `chapters_fts`).
- **Index state:** `chapters` = 8294 rows, `chapters_fts` = 8083 rows. So the index is partially filled from an earlier hand-run rebuild, not empty: about 211 chapters are missing, and some rows may be stale. `fts-rebuild` deletes every FTS row and re-indexes all chapters, so it fixes both.
- **Expected rebuild errors:** 6 books have unreachable legacy relative `data/` chapter paths (3 reingested, 3 with gone epubs; list in `<db-dir>/doctor/awaiting-redownload.md`). Their chapters will count as `errors` in the rebuild output and, because the rebuild clears the table first, **lose their FTS rows** if they had any. That is the command's designed behavior; report the count, do not treat it as failure. They stay searchable through chunk search.
- **`fts-rebuild`:** `agentic-pipeline fts-rebuild` (`agentic_pipeline/cli.py`), exit 0 on success even with per-chapter errors, exit 1 if `chapters_fts` is missing or the rebuild fails. It commits every 100 chapters, so `text_search` sees a partial index while it runs.
- **Worker:** launchd label `com.taylorstephens.agentic-pipeline-worker` (KeepAlive), started by `scripts/run-worker.sh`, which runs the canonical checkout's working tree (`~/_Projects` is a symlink to `~/Dev/_Projects`). `launchctl list` showed no PID and last exit status 1 at write time; observed once, so check `launchctl print gui/$(id -u)/com.taylorstephens.agentic-pipeline-worker` and `agentic-pipeline health` before assuming it is running or healthy. If it is crash-looping, report the cause from its log instead of kickstarting blindly.
- **Canonical checkout:** `~/Dev/_Projects/book-mcp-server`, `main` at or after `bbe8e6a`; only `github-issues/` is untracked (not yours).

## What this session does

Single task: make the live system match merged `main`. Do the steps in order; stop and report at any failed check.

1. **Re-derive state.** `git -C ~/Dev/_Projects/book-mcp-server fetch origin && git status -sb`; `gh issue view 14` (expect CLOSED); `agentic-pipeline health` and `agentic-pipeline pending` to see whether the pipeline is mid-run. If any pipeline is in PROCESSING/EMBEDDING, wait or ask: the rebuild and the worker both write the same DB.
2. **Pull main** in the canonical checkout: `git pull --ff-only`. Do not use a worktree for this session; the worker runs this tree. Confirm `agentic_pipeline/library/fts_index.py` exists afterward.
3. **Disk check, then backup.** The DB is about 3 GB; confirm at least 7 GB free (`df -h` on the DB's volume) before backing up. Take a WAL-safe backup with the same method `doctor` uses (`health.doctor.create_backup`), or `sqlite3 <db> ".backup '<db>.backup-fts-$(date +%Y%m%d-%H%M%S)'"`. Verify the backup opens and has the same `chapters` count. Use a name that does not match `*.backup-doctor-*` unless you want it pruned.
4. **Rebuild.** `source .venv/bin/activate && agentic-pipeline fts-rebuild > <scratchpad>/fts-rebuild.log 2>&1; echo $?` (redirect to a file; do not trust inline capture). Report `entries_before`, `entries_after`, `errors`. Expect `entries_after` close to 8294 minus the errors from the legacy-path books.
5. **Verify.** `sqlite3` count of `chapters_fts` equals `entries_after`; no duplicate `chapter_id` (`SELECT chapter_id, count(*) FROM chapters_fts GROUP BY 1 HAVING count(*) > 1` returns nothing); a `text_search`-style `MATCH` for a term you know is in a book (pick one from a chapter file) returns it. Run `agentic-pipeline doctor` (bare, report only; do **not** `--fix`).
6. **Kickstart the worker** after the rebuild finishes: `launchctl kickstart -k gui/$(id -u)/com.taylorstephens.agentic-pipeline-worker`. Then confirm it is running (`launchctl print ...` shows a PID and `state = running`), check its log for import errors for 60 seconds, and run `agentic-pipeline health`.
7. **Report** one short summary: backup path, rebuild counts, verification results, worker state. Release any claim. `save_session` once.

## Constraints carried over

- Never modify the `library.db` schema; this session only reads it and runs `fts-rebuild`, which rewrites `chapters_fts` rows.
- No `doctor --fix`, no `rechunk`, no reingest, no `approve` in this session.
- The backup in step 3 is mandatory before step 4; if the backup fails or disk is short, stop.
- Do not delete the backup afterward; tell the user its path and size.
- Do not install anything with pip into any venv (`uv` only; `.config/wt.toml` runs `uv sync --locked`).
- Stage by name if anything is committed (nothing should be); the untracked `github-issues/` is not yours.
- Do not bypass `wt-live-session-guard` or other hooks.

## Caution

Re-derive current state first (`git fetch`, `git log --oneline -5 origin/main`, issue #14, worker status, pending pipelines); this prompt goes stale same-day. This session writes to shared live state, so run the `concurrent-session-preflight` skill (`tstephx/book-mcp-server 14`, branch label `live-fts-rebuild-14`) before step 3 and claim it; a peer running `fts-rebuild`, `doctor --fix`, `rechunk` or an approval at the same time would contend for the DB write lock. Hosted CI is dormant and not relevant here. If the book-library MCP server is open in Claude Desktop during the rebuild, a `text_search` made mid-rebuild may return partial results; that is expected.

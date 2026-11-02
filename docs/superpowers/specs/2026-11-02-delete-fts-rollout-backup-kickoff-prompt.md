# Delete the FTS-rollout safety backup `library.db.backup-fts-20261102-124502` (not before 2026-11-09)

Last open item from the #14 `chapters_fts` rollout. No code changes and no PR: this session runs read-only health gates against the live library, then deletes three named backup files, and reports.

## Where things stand

Observed 2026-11-02, re-verify before acting:

- **The backup:** `~/Library/Application Support/book-library/library.db.backup-fts-20261102-124502` (3,137,032,192 bytes, about 2.9 GiB), taken at 12:45 on 2026-11-02 immediately before the rebuild, plus sidecars `...-124502-shm` (32 KB) and `...-124502-wal` (0 B). Its state is pre-rebuild: `chapters` = 8294, `chapters_fts` = 8083. It is the only pre-rebuild copy; the older `backup-pre-completed-at-20260728` was already deleted on 2026-11-02.
- **Why it exists:** rollback for the `fts-rebuild` run (PR #18, `ee24e84`, closes #14). The rebuild was verified afterwards: `chapters_fts` = `chapters` = 8294, no duplicate `chapter_id`, bare `agentic-pipeline doctor` clean, `MATCH` spot checks pass, and all 8294 chapters are readable through `agentic_pipeline/library/chapter_reader.read_chapter_content` (investigated in #27, closed as not a bug). The kickoff prompt for the rollout is marked done (PR #30).
- **Why wait until 2026-11-09:** a week of normal `text_search` use is the real test. Nothing but time makes deletion safer.
- **Disk:** the data volume was about 88% used with about 55 GiB free on 2026-11-02, so deleting is housekeeping, not an emergency.
- **Tracking note:** `~/Dev/Obsidian-Second-Brain/Action-Tracker/Backlog/later/2026-11-02-delete-fts-rollout-backup-book-library.md` (scheduled 2026-11-09). Its `dod` matches this prompt.

## What this session does

Do the steps in order; stop and report at any failed check. If a step fails, do not delete anything.

1. **Re-derive state.** Check today's date: if it is before 2026-11-09, stop and tell the user. Run `ls -lT` on the three backup files and confirm all three exist. Run `ListAgents`; if a peer is working on `library.db` (rebuild, `doctor --fix`, `rechunk`, approvals), wait or ask.
2. **Health gate.** `agentic-pipeline doctor` (bare, report only; do **not** `--fix`) must end with `integrity: OK`. `agentic-pipeline health` must show nothing stuck or in `needs_retry` that you don't understand. If not clean, stop: the backup may be needed to compare against.
3. **Search gate.** Against the live DB (`sqlite3`, read-only): `chapters` count equals `chapters_fts` count (they will be larger than 8294 if books were approved since; equality is the check, not 8294); `SELECT chapter_id, count(*) FROM chapters_fts GROUP BY 1 HAVING count(*) > 1` returns nothing; a `MATCH` for a distinctive term you confirm is in a chapter file returns that book (for example `reflog` for Mastering Git, as of 2026-11-02). If the book-library MCP server is connected, also call its `text_search` tool once with the same term, since the earlier checks went through `sqlite3` directly, not through the tool.
4. **Open-handle check.** `lsof` on the three backup files must return nothing.
5. **Delete exactly these three paths, by full name, no glob:** `library.db.backup-fts-20261102-124502`, `library.db.backup-fts-20261102-124502-shm`, `library.db.backup-fts-20261102-124502-wal`, all under `~/Library/Application Support/book-library/`. Then `ls` the directory to confirm they are gone and `df -h` the volume; report the space freed.
6. **Close the tracking note.** In the vault note above, set `status: done` and append one line with the date and "deleted, gates passed". Do not triage other notes.
7. **Report** one short summary: the gate results, the three files deleted, space freed. `save_session` once.

## Constraints carried over

- Delete only the three named backup files. No globs, no `rm -rf`, nothing else in that directory.
- Never touch `library.db` or its `-wal`/`-shm`, the `books/` directory, or `doctor/` and `rechunk/` under the DB directory.
- No `doctor --fix`, no `fts-rebuild`, no `rechunk`, no reingest, no `approve`, no schema change. Read-only against the live DB.
- If any gate in steps 1 to 4 fails, do not delete; report what failed.
- Nothing in this repo should change, so nothing should be committed. The vault note is outside the repo.
- Do not bypass hooks.

## Caution

This prompt was written 2026-11-02 and is meant to run on or after 2026-11-09, so assume the numbers have moved: row counts have grown with new approvals, and the worker, peer sessions and open issues may have changed. Re-derive current state first (date, `ls`, `doctor`, `health`, `ListAgents`, `gh issue list --repo tstephx/book-mcp-server`) before trusting anything above. The example search term and the 8294 figure are observations from one day, not contracts. This prompt claims no issue or branch, so the `concurrent-session-preflight` claim steps do not apply; the `ListAgents` check in step 1 is the collision guard.

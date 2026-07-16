# Integrity Doctor — Design

**Date:** 2026-07-15 (refined via interview same day)
**Status:** Approved for planning
**Command:** `agentic-pipeline doctor` (report) / `doctor --fix` (repair)
**MCP:** `doctor_report` (read-only) on the agentic-pipeline server

## Problem

The library database lies in four measured ways (live numbers, 2026-07-15,
after PR #2 merged):

| Violation | Count | Consequence |
|---|---|---|
| Orphaned chunks — `chapter_id`/`book_id` resolves to nothing | 2,396 of 19,210 embedded (12.5%) | Embedded content invisible to search; INNER JOINs drop it silently |
| Lost books — pipeline `state='complete'` but no `books` row | 25 books (46 dead ids) | Pipeline claims success for books the library does not have |
| `chapters.content_hash` NULL | 840 of 7,797 (10.8%) | The duplicate-chapter check filters NULL hashes, so it can never fire — a vacuous check |
| `books.book_type` NULL | 47 of 261 | Any tool filtering by type silently misses these books |

The orphans are not lost content in place: they are stale remnants of old
reingests, from before `_cleanup_book_data` handled chunks. Grouping them by
dead `book_id` and matching pipeline `source_path` basenames against live
books shows (these figures cover all 2,449 orphaned chunks — the 2,396
embedded above plus 53 never-embedded; doctor treats both identically):

- **90 chunks** — book exists live under another id (pure duplicates)
- **2,090 chunks** — book has a pipeline row but **no live copy**: 25 distinct
  books fell out of the library while their pipelines report `complete`.
  20 have source files available (original path or `processed/`);
  5 do not (incl. Refactoring UI; Modern CSS with Tailwind)
- **269 chunks** — no trace of their book id anywhere
- Only 23 orphans have exact-content twins in the live index (hash equality;
  boundaries differ across reingests, so this undercounts book-level coverage)

## Decisions

From the brainstorm:

1. **Delete all orphans, with a manifest.** The chunks are already unusable —
   search requires joins they cannot satisfy. A manifest converts silent loss
   into a visible re-acquisition list. No reconstruction of synthetic
   book/chapter rows.
2. **Package as a permanent `doctor` command, not a one-shot script.** The
   detector doubles as the recurrence invariant; check and fix share one
   codebase so they cannot drift apart.
3. **Re-ingestion of the 20 recoverable books is operational, not doctor's
   job.** Doctor prints the `reingest` commands; a human runs them. Ingestion
   is slow, API-metered, and already has a correct, race-safe tool.
4. **No schema-level foreign keys now.** Rebuilding `chunks` on a live 700 MB
   DB is high blast-radius; may be a later hardening step once doctor exists.

From the design interview:

5. **`--fix` archives the 25 dead pipelines** (`complete → archived`, with
   `expected_state=COMPLETE` so the CAS guards it, and the reason recorded in
   the transition's `agent_output`: `doctor: book absent from library; see
   manifest`). Otherwise `check_lost_books` fires forever and
   `integrity: 25 issues` becomes wallpaper — a check that always fails is as
   useless as one that cannot. After `--fix`, only NEW drift fires.
6. **`health` runs the full checks inline (~1–2 s); the worker loop runs
   none.** No cache, no staleness — a stale "OK" is the exact lie class this
   tool exists to kill. Revisit only if `health` becomes machine-polled.
7. **Manifest lives in a fixed directory beside the DB:**
   `<db-dir>/doctor/manifest-<ts>.md`. Always findable regardless of cwd; the
   tool stays self-contained (no vault coupling). `--manifest PATH`
   overrides; the absolute path is printed. Vault capture is the user's
   call, via `/capture`.
8. **Backup retention: keep the newest 2 doctor backups, prune older** —
   matching only the doctor backup filename pattern, never other files.
   Bound: ~1.4 GB. When a `--fix` run finds nothing fixable, the backup is
   skipped entirely (nothing to protect).
9. **MCP surface: `doctor_report` only, read-only,** registered in both
   `agentic_pipeline/mcp_server.py` and the `agentic_mcp_server.py` wrapper
   (the registration test pattern requires both). Repair is deliberately NOT
   reachable from a chat surface — `--fix` is CLI-only.
10. **No confirmation prompt: `--fix` is the consent,** consistent with the
    house `--execute` convention. Bare `doctor` is the always-safe dry run;
    the automatic backup is the safety net.
11. **`book_type` backfill validates against the `BookType` enum.** A profile
    value is copied only when it is a real enum member and not `unknown`
    (writing `unknown` as an answer is not an answer). Anything else —
    legacy strings, `unknown`, vocabulary drift — lands in
    `FixReport.skipped` with the offending value. Drift becomes a finding,
    not a write.
12. **Single-phase deletion.** All 2,449 orphans go at once, including those
    of the 20 re-ingestable books: held chunks would do nothing while held
    (unjoinable), the kept backup preserves them one-run-deep, and a failed
    re-ingest is debugged from its source file. No per-book pending state.

## Architecture

New module `agentic_pipeline/health/doctor.py`, joining the existing
`health/` package (`stuck_detector.py`, `monitor.py`).

```
cli.py: doctor(--fix, --no-backup, --manifest PATH)
   │
   ├── doctor.run_checks(db_path) -> list[Finding]
   │       check_orphaned_chunks      (chapter_id or book_id unresolvable)
   │       check_lost_books           (complete pipeline, no books row;
   │                                   annotated: source available? live copy?)
   │       check_null_content_hash    (fixable subset: file_path exists)
   │       check_null_book_type       (fixable subset: pipeline book_profile
   │                                   carries a valid, non-unknown BookType)
   │
   └── doctor.apply_fixes(db_path, findings, backup, manifest_path) -> FixReport

mcp_server.py: doctor_report() -> dict   (run_checks, JSON-shaped; read-only)
agentic_mcp_server.py: registers doctor_report in the wrapper
```

`Finding` is a dataclass: `category` (str enum of the four checks), `count`,
`fixable_count`, `details` (per-item dicts). `FixReport` records, per
category: attempted, fixed, skipped (with reasons). Checks are pure reads;
each is a query that CAN return violations — every check gets a
seeded-violation test proving it fails when it should.

## Fix semantics (`--fix`), in order

1. **Backup** — WAL-safe `sqlite3.backup()` to
   `<db>.backup-YYYYMMDD-HHMMSS` beside the DB; then prune doctor backups
   beyond the newest 2 (pattern-matched only). Skipped when checks found
   nothing fixable. `--no-backup` skips explicitly. The path is printed.
2. **Manifest** — write all 25 lost books to
   `<db-dir>/doctor/manifest-<ts>.md` (or `--manifest PATH`):
   title/basename, chunk count, re-ingestable vs source-gone, and a
   ~200-char content sample from one chunk for identification. Absolute
   path printed.
3. **Delete orphans** — all of them, single transaction; count reported.
4. **Archive dead pipelines** — the 25 `complete`-without-book rows go
   `complete → archived` via `update_state(..., expected_state=COMPLETE)`,
   reason in `agent_output`. A pipeline that moved since the check (CAS or
   ownership failure) is skipped and reported, not forced.
5. **Backfill `content_hash`** — for chapters whose `file_path` exists, using
   the same hash routine `src/utils/embedding_sync.py` uses when it writes
   `content_hash`, so backfilled values are byte-compatible with existing
   ones. Chapters with missing files are reported as unfixable — not
   silently skipped.
6. **Backfill `book_type`** — copy `book_type` and `confidence` from the
   pipeline's `book_profile` into `books.book_type` /
   `books.classification_confidence`, stamping
   `books.classified_by = 'backfill:doctor'` — only for values passing the
   enum validation in Decision 11. The rest (including the 1 book with no
   profile) are reported.
7. **Print re-ingest commands** for the 20 recoverable books.

Properties:

- `--fix` re-runs the checks first and fixes only what it just found — never
  stale findings passed across invocations.
- Fix categories are independent: a failure in one is recorded in
  `FixReport.skipped` and the rest proceed (the `skipped`-list pattern from
  `BatchOperations`).
- **Idempotent:** a second `--fix` finds nothing — and therefore also takes
  no backup. This is a test, not a hope.
- No confirmation prompt; no worker lock. Orphan deletion touches only rows
  no live query can reach; backfills are additive single-column updates;
  the pipeline archiving is CAS-guarded. (Reasoning recorded so a future
  reader doesn't add a lock or prompt reflexively.)

## Invariant wiring

`agentic-pipeline health` calls `run_checks()` inline and prints one line:
`integrity: OK` or `integrity: N issues — run 'agentic-pipeline doctor'`.
Because `--fix` archives the dead pipelines, the line returns to `OK` after
repair and fires only on new drift. The worker loop is untouched.

## Testing

TDD throughout, per the project's contract-testing rules:

- Each check: seeded-violation test (check MUST fail) + clean-DB test
  (MUST pass). A check that cannot fail is this codebase's recurring defect
  class; these tests exist to prevent recruiting doctor into it.
- Enum completeness: every `Finding.category` value maps to a fix handler or
  an explicit "report-only" marker; no category silently unhandled.
- Type assertions on `Finding`/`FixReport` fields.
- Idempotency: seeded DB → `--fix` → checks clean → `--fix` again → zero
  changes and zero new backup.
- Backup rotation: three sequential fixing runs leave exactly 2 backups;
  non-doctor files in the directory are never touched.
- Archive step: dead pipelines end `archived` with the reason recorded; a
  concurrently-moved pipeline is skipped, not forced (seed via direct state
  change between check and fix).
- `book_type` validation: valid member copied with `classified_by` stamp;
  `unknown` and off-enum strings skipped and reported.
- Manifest content asserted (lost books present, split correctly, samples
  non-empty, path printed).
- Hash-compat: backfilled `content_hash` equals what `embedding_sync`
  computes for the same file.
- `health` integration: violations present ⇒ the integrity line reports
  them; clean DB ⇒ `integrity: OK`.
- MCP: `doctor_report` registered in both server entry points (existing
  wrapper-registration test pattern) and returns JSON-shaped findings.

## Out of scope

- Running the 20 re-ingests (operational; commands printed).
- Reconstructing the 5 source-gone books (manifest lists them for
  re-acquisition; their chunks are deleted like all orphans).
- Re-chunking / chunk-size correction (separate project B).
- Foreign-key schema enforcement (possible later hardening).
- The remaining leftover items (rejection-reason persistence, PDF
  completeness check) — separate small project C.
- Any `--fix` capability over MCP.

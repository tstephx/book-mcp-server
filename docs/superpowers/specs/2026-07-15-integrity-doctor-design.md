# Integrity Doctor — Design

**Date:** 2026-07-15
**Status:** Approved for planning
**Command:** `agentic-pipeline doctor` (report) / `doctor --fix` (repair)

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

## Decisions (made during brainstorm)

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
   │       check_null_book_type       (fixable subset: pipeline book_profile exists)
   │
   └── doctor.apply_fixes(db_path, findings, backup, manifest_path) -> FixReport
```

`Finding` is a dataclass: `category` (str enum of the four checks), `count`,
`fixable_count`, `details` (per-item dicts). `FixReport` records, per
category: attempted, fixed, skipped (with reasons). Checks are pure reads;
each is a query that CAN return violations — every check gets a
seeded-violation test proving it fails when it should.

## Fix semantics (`--fix`), in order

1. **Backup** — WAL-safe `sqlite3.backup()` to
   `<db>.backup-YYYYMMDD-HHMMSS` beside the DB. Skippable only via explicit
   `--no-backup`. (~700 MB per backup; the report prints the path.)
2. **Manifest** — write all 25 lost books to `--manifest` (default
   `doctor-manifest-<ts>.md` in cwd): title/basename, chunk count,
   re-ingestable vs source-gone, and a ~200-char content sample from one
   chunk for identification.
3. **Delete orphans** — single transaction; count reported.
4. **Backfill `content_hash`** — for chapters whose `file_path` exists, using
   the same hash routine `src/utils/embedding_sync.py` uses when it writes
   `content_hash`, so backfilled values are byte-compatible with existing
   ones. Chapters with missing files are reported as unfixable — not
   silently skipped.
5. **Backfill `book_type`** — copy `book_type` and `confidence` from the
   pipeline's `book_profile` JSON into `books.book_type` /
   `books.classification_confidence`, stamping
   `books.classified_by = 'backfill:doctor'`. The 1 book with no profile is
   reported, not guessed.
6. **Print re-ingest commands** for the 20 recoverable books.

Properties:

- `--fix` re-runs the checks first and fixes only what it just found — never
  stale findings passed across invocations.
- Fix categories are independent: a failure in one is recorded in
  `FixReport.skipped` and the rest proceed (the `skipped`-list pattern from
  `BatchOperations`).
- **Idempotent:** a second `--fix` finds nothing. This is a test, not a hope.
- No worker lock is required: orphan deletion touches only rows no live query
  can reach, and both backfills are additive single-column updates. (Noting
  the reasoning here so a future reader doesn't add a lock reflexively.)

## Invariant wiring

`agentic-pipeline health` calls `run_checks()` and prints one line:
`integrity: OK` or `integrity: N issues — run 'agentic-pipeline doctor'`.
Recurring drift is then visible on the command operators already run.

## Testing

TDD throughout, per the project's contract-testing rules:

- Each check: seeded-violation test (check MUST fail) + clean-DB test
  (MUST pass). A check that cannot fail is this codebase's recurring defect
  class; these tests exist to prevent recruiting doctor into it.
- Enum completeness: every `Finding.category` value maps to a fix handler or
  an explicit "report-only" marker; no category silently unhandled.
- Type assertions on `Finding`/`FixReport` fields.
- Idempotency: seeded DB → `--fix` → checks clean → `--fix` again → zero
  changes.
- Manifest content asserted (lost books present, split correctly, samples
  non-empty).
- Hash-compat: backfilled `content_hash` equals what `embedding_sync`
  computes for the same file.
- `health` integration: violations present ⇒ the integrity line reports them.

## Out of scope

- Running the 20 re-ingests (operational; commands printed).
- Reconstructing the 5 source-gone books (manifest lists them for
  re-acquisition; their chunks are deleted like all orphans).
- Re-chunking / chunk-size correction (separate project B).
- Foreign-key schema enforcement (possible later hardening).
- The remaining leftover items (rejection-reason persistence, PDF
  completeness check) — separate small project C.

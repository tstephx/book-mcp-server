---
status: active
tags: [project/book-mcp-server, format/adr]
type: project
created: '2026-08-14'
modified: '2026-08-14'
---

# ADR 008: db-schema Skills Evaluation Across DB-Backed Repos

**Status:** Accepted
**Date:** 2026-08-14

## Context

career-coach-mcp piloted a `db-schema` Claude Code skill (`.claude/skills/db-schema/SKILL.md`)
after `end_reason` (a `work_history` column) turned out to be constrained only in
application code, not the DB schema — reading the DDL alone gave a false sense of
what was and wasn't validated. Issue #8 asked whether that pattern is worth
extending to the other DB-backed repos in this account, filed here because
book-mcp-server has the worst 0-byte-placeholder problem, but the actual work
spans several repos.

### Part 1 — the `library.db` ambiguity

Four filesystem paths shared the name `library.db` as of 2026-07-22 (when the
underlying task note was written). Diffed `sqlite_master` across the three that
had real content:

| Path | Rows in schema dump | Relationship |
|---|---|---|
| `~/Library/Application Support/book-library/library.db` (2.9 GB) | 260 lines — 20+ tables (`books`, `chapters`, `chunks`, FTS tables, `processing_pipelines`, `autonomy_*`, `audit_*`, etc.) | **Canonical.** Lives outside every repo. book-mcp-server and briefcase are pure readers. |
| `document-intelligence/data/library.db` → renamed `documents.db` (86,016 B) | 54 lines, 4 tables | **Unrelated schema.** Coincidental filename only — already resolved by a same-day rename commit (`9baba72`) that updated all 17 references cleanly. Not part of the ambiguity going forward. |
| `book-ingestion-python/data/library.db` (28,672 B) | 28 lines, 3 tables (`books`, `chapters`, `processing_checkpoints`) | **Reduced dev-copy subset** of the canonical schema — same table names, fewer columns, none of the pipeline/audit/autonomy tables. |

**Resolution: three genuinely different schemas, not one canonical schema
duplicated four ways.** The rename already resolved document-intelligence's
share of the ambiguity; the canonical DB and book-ingestion-python's dev copy
remain genuinely related (subset relationship) but are not interchangeable —
code that assumes the dev copy has the same 20 tables as canonical will be
wrong.

### Part 2 — per-repo gate

Same bar as career-coach-mcp's pilot: name one real schema-vs-code
disagreement, or explicitly skip with a reason. Investigated five repos in
parallel. All five had a genuine trap — none were skipped.

| Repo | Skill added | The trap |
|---|---|---|
| book-mcp-server | `.claude/skills/db-schema/SKILL.md` | `PRAGMA foreign_keys = ON` is only set on the pipeline-side connection, not the MCP-tool-side one — both open the same `library.db`, so `ON DELETE CASCADE` in the DDL only fires through one of the two code paths. A future MCP delete tool built the way every existing tool is built would silently orphan rows. |
| document-intelligence | `.claude/skills/db-schema/SKILL.md` | The `achievements` table (and `achievement_sources`, `coverage_reports`) is fully defined and wired end-to-end in code, but its only caller is itself never invoked — the real `cli.py extract` path writes straight to JSON instead. 0 rows is expected, not a bug. |
| book-ingestion-python | `.claude/skills/db-schema/SKILL.md` | `CLAUDE.md` asserts code "must point at the canonical DB... never a repo-local `data/library.db`" — false for the primary CLI (`cli.py`/`cli_enhanced.py`, 12 call sites), which still defaults to the local dev copy via `config/config.json`. A recent commit (`2930459`) fixed the docs and one script, not the config default. |
| briefcase | `.claude/skills/db-schema/SKILL.md` | `chapters_fts` (used for all search) has zero sync triggers against `chapters` and has drifted — 211 chapters (2.5%) belonging to completed books are silently unsearchable, no error surfaced. |
| Claude-Innit | `.claude/skills/db-schema/SKILL.md` | `vault_embeddings` is documented in-source as "deprecated, superseded by `vault_chunk_embeddings`" but its writer is still fully wired and simply never called — 0 rows forever, by design, not failure. Real vault embedding data (118,973 rows) lives in `vault_chunk_embeddings` instead. |

All five skills match career-coach-mcp's shape exactly: orientation only,
never pastes table definitions, states the canonical source + live DB path,
and lists known traps with file:line evidence. None needed the
subagent-dispatch alternative the issue floated — no precedent for that shape
was found anywhere in the account, and the plain-skill pattern never embeds
the schema regardless of the underlying DB's size, so file size isn't driven
by schema size.

### Part 3 — 0-byte placeholder audit

All five resolved by deletion (each confirmed unreferenced by any code —
grepped for literal path strings, not just module-path false positives — and
either gitignored+untracked already, or outside any repo entirely):

- `book-mcp-server/data/books.db` — dead; `get_db_path()` defaults straight to canonical `library.db`, never this path.
- `book-mcp-server/data/pipeline.db` — same reason.
- `~/Library/Application Support/book-library/pipeline.db` (beside canonical, outside any repo) — dead; canonical `library.db` already has `processing_pipelines` etc. inline, this sibling file was never wired to anything.
- `book-ingestion-python/book_ingestion/data/library.db` — dead; distinct from (and not to be confused with) the real dev copy at `book-ingestion-python/data/library.db`, which is not a placeholder.
- `Claude-Innit/memory.db` — dead; the live DB is `Claude-Innit/data/innit.db` (5.4 GB), unrelated to this stray root-level file. (`memory.db` elsewhere in this account's docs/session notes refers to the Python module `memory/db.py`, not this file — a naming collision worth knowing about if it comes up again.)

## Decision

Add a `db-schema` skill to all five candidate repos (book-mcp-server,
document-intelligence, book-ingestion-python, briefcase, Claude-Innit).
Delete all five 0-byte placeholders. Treat the `library.db` naming ambiguity
as resolved: three distinct schemas, not one, with document-intelligence's
share already closed by its independent rename.

## Consequences

- Every DB-backed repo checked had a real trap — the career-coach-mcp pattern
  generalizes, this wasn't a one-off. Worth treating "does this repo have a
  db-schema skill" as a standing question for any future DB-backed repo, not
  just these five.
- Two traps found here point past documentation into real follow-up work,
  tracked separately, not fixed by this ADR: book-ingestion-python's CLI
  still writes to the wrong DB by default (needs a code fix routing `Config`
  through `AGENTIC_PIPELINE_DB`), and book-mcp-server's inconsistent
  `PRAGMA foreign_keys` is a live correctness gap for any future delete tool.
- No `.claude/skills/db-schema/SKILL.md` existed in any of these five repos
  before this ADR — all five are net-new additions, not edits.

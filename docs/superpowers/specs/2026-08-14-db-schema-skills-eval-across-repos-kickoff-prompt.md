# Kickoff: evaluate db-schema skills across DB-backed repos (book-mcp-server#8)

## Where things stand

Spun out of `docs/superpowers/specs/2026-08-14-close-out-career-kb-book-mcp-behavioral-studio-issues-kickoff-prompt.md`
(lane 8 of an 8-lane batch closing 76 open workspace issues, clustered by
repo). That kickoff prompt scoped `tstephx/book-mcp-server#8` into a
single-repo lane alongside `career-kb-mcp#1` and `behavioral-studio#1`, but
flagged in advance that #8 "is broader than a typical single-repo doc fix
(spans 'DB-backed repos' plural) — read fully before scoping the commit,"
and its own constraints said: "If book-mcp-server#8 turns out to be
genuinely cross-repo in scope, stop and flag it rather than silently
narrowing it to fit this lane."

Having now read the full issue body (`gh issue view 8 --repo
tstephx/book-mcp-server`), it is genuinely cross-repo. This kickoff prompt
supersedes that lane's step 2 for issue #8 — the lane closed #1 and left #8
open, to be picked up as its own properly-scoped session via this file.

**Issue #8 full scope** (`tstephx/book-mcp-server#8`, still OPEN as of
2026-08-14, source: Action-Tracker task note
`2026-07-22-evaluate-db-schema-skills-across-db-backed-repos.md`):

Follow-on to the `career-coach-mcp` constraint-trap task (the pilot for a
"db-schema skill"). Question: is a db-schema skill worth it outside
`career-coach-mcp`? Filed against `book-mcp-server` because that repo is a
reader of the canonical library DB and has the worst 0-byte-placeholder
problem, but the work itself touches several repos plus a file outside any
repo.

### Part 1 — resolve the `library.db` ambiguity first

Four paths share the name `library.db` (a fifth repo reads the canonical
one without owning a copy) — do not assume same filename = same schema:

| Path | Size | Note |
|---|---|---|
| `~/Library/Application Support/book-library/library.db` | 2.9 GB | the canonical DB — **lives outside every repo**, `book-mcp-server` is a reader of it, same as `briefcase`, not an owner |
| `_Projects/document-intelligence/data/library.db` | 86,016 B | own DB, same filename, presumed different schema |
| `_Projects/book-ingestion-python/data/library.db` | 28,672 B | dev copy |
| `_Projects/book-ingestion-python/book_ingestion/data/library.db` | 0 B | placeholder |
| `_Projects/briefcase/` | — | no `.db` of its own; reads the canonical one |

Confirm by diffing `sqlite_master` across the canonical DB,
`document-intelligence`'s, and `book-ingestion-python`'s before writing
anything about schema count. Re-verify these sizes/paths still hold before
trusting the table above — re-`ls -la`/`du -h` each path first, this was
written 2026-08-14 from the issue body only.

### Part 2 — per repo, same gate as `career-coach-mcp`'s pilot

For each candidate repo (at minimum: `book-mcp-server`,
`document-intelligence`, `book-ingestion-python`, `briefcase` — check
whether `Claude-Innit` also qualifies given its 0-byte `memory.db` in Part
3), name one real schema-vs-code disagreement, or explicitly skip the repo
with a recorded reason. Sizing rule: under ~200 lines, a plain skill;
only reach for the subagent-dispatch template when keeping the schema out
of context is worth the dispatch cost.

### Part 3 — audit the 0-byte `.db` placeholders

Five found, each needs a resolution (delete / gitignore / document as an
intentional fixture) — re-confirm each is still 0 bytes before acting,
sizes may have changed since 2026-07-22:

- `book-mcp-server/data/books.db` (0 B)
- `book-mcp-server/data/pipeline.db` (0 B)
- `Claude-Innit/memory.db` (0 B)
- `book-ingestion-python/book_ingestion/data/library.db` (0 B)
- `~/Library/Application Support/book-library/pipeline.db` (0 B, beside the canonical DB)

### Done when (issue's own bar)

The `library.db` ambiguity is resolved in writing — one schema or several,
stated with evidence — AND each candidate repo has either a db-schema
skill or a recorded reason it was skipped AND every 0-byte `.db`
placeholder is either deleted, gitignored, or documented as an intentional
fixture.

## What this session does

1. Re-derive current state first (see Caution below) — this issue may have
   drifted since 2026-08-14.
2. Diff `sqlite_master` across the canonical `library.db`,
   `document-intelligence/data/library.db`, and
   `book-ingestion-python/data/library.db` to settle Part 1 with evidence
   (e.g. `sqlite3 <path> ".schema" > /tmp/schema-<name>.sql` per DB, then
   `diff`).
3. Write up the ambiguity resolution (one schema or several) somewhere
   durable — this repo's docs, or wherever `career-coach-mcp`'s pilot
   writeup lives, for consistency.
4. Walk each candidate repo for Part 2's gate; for repos that get a
   db-schema skill, follow whatever pattern `career-coach-mcp`'s pilot
   established (find it first — don't invent a new shape).
5. Resolve each of the five 0-byte placeholders per Part 3.
6. Close `tstephx/book-mcp-server#8` referencing the writeup and the
   per-repo skill/skip decisions once the "Done when" bar is met.

## Constraints carried over

- This spans multiple repos deliberately — that's the reason it was split
  out, not a mistake to narrow back down. Touch `book-mcp-server`,
  `document-intelligence`, `book-ingestion-python`, `briefcase`, and
  `Claude-Innit` as needed; don't expand further without checking back.
- The canonical `library.db` under `~/Library/Application Support/
  book-library/` is outside every repo — treat it as read-mostly
  (schema diffing, not schema changes) unless the writeup explicitly
  calls for a migration, which is out of scope for this issue.
- Don't invent a sixth db-schema-skill template shape — match whatever
  `career-coach-mcp`'s pilot already established, or explicitly document
  why a repo's case doesn't fit it.

## Caution

Written 2026-08-14 from `gh issue view 8 --repo tstephx/book-mcp-server`
only — no source file, DB, or schema was actually read while drafting
this. Before starting:

- `gh issue view 8 --repo tstephx/book-mcp-server` — re-confirm still open
  and scope hasn't changed.
- `git -C /Users/taylorstephens/Dev/_Projects/book-mcp-server log --oneline -5 && git status`,
  and likewise for `document-intelligence`, `book-ingestion-python`,
  `briefcase`, and `Claude-Innit` if touching them.
- Re-check every path/size cited above — `ls -la` / `du -h` each — before
  treating them as current fact.
- `ListAgents` — this workspace routinely runs multiple concurrent
  sessions (this prompt was itself written by lane 8 of an 8-lane
  concurrent batch). Run the `concurrent-session-preflight` skill before
  claiming this issue or branching, in case another session already
  picked it up.

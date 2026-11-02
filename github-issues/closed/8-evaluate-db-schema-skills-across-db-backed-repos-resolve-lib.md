---
title: "Evaluate db-schema skills across DB-backed repos — resolve library.db ambiguity, gate on real schema-vs-code disagreement, audit 0-byte placeholders"
number: 8
state: closed
labels: [documentation]
repo: book-mcp-server
url: https://github.com/tstephx/book-mcp-server/issues/8
created: 2026-08-14
updated: 2026-08-14
---

## Summary
Follow-on to the career-coach-mcp constraint-trap task (the pilot). Is a db-schema skill worth it outside career-coach-mcp? Spans multiple repos — filed here since book-mcp-server is one of the readers of the canonical library DB and is where the 0-byte placeholder problem is worst.

## 1. Resolve the library.db ambiguity first
Four repos are in the frame:
| Path | Size | Note |
|---|---|---|
| `~/Library/Application Support/book-library/library.db` | 2.9 GB | the canonical DB |
| `_Projects/document-intelligence/data/library.db` | 86,016 B | own DB, same filename |
| `_Projects/book-ingestion-python/data/library.db` | 28,672 B | dev copy |
| `_Projects/book-ingestion-python/book_ingestion/data/library.db` | 0 B | placeholder |
| `_Projects/briefcase/` | — | no `.db` of its own; reads the canonical one |

**Correction to carry into the writeup:** the 2.9 GB canonical DB does not live inside book-mcp-server. It sits in `~/Library/Application Support/book-library/`, outside every repo. book-mcp-server is a *reader* of it, same as briefcase — the repo owns the schema, not the file.

Three files share the name `library.db` and are almost certainly not the same schema. Confirm by diffing `sqlite_master` across the canonical DB, document-intelligence's, and book-ingestion-python's before assuming either way.

## 2. Then, per repo — same gate as career-coach-mcp
Name one real schema-vs-code disagreement, or skip the repo. Sizing rule: under ~200 lines, plain skill; only reach for the subagent template when keeping the schema out of context is worth the cost of a dispatch.

## 3. Audit the 0-byte .db placeholders
Five found: `book-mcp-server/data/books.db` (0B), `book-mcp-server/data/pipeline.db` (0B), `Claude-Innit/memory.db` (0B), `book-ingestion-python/book_ingestion/data/library.db` (0B), `~/Library/Application Support/book-library/pipeline.db` (0B, beside the canonical DB). For each: delete, gitignore, or document as an intentional fixture.

## Done when
The library.db ambiguity is resolved in writing — one schema or several, stated with evidence — AND each candidate repo has either a db-schema skill or a recorded reason it was skipped AND every 0-byte .db placeholder is either deleted, gitignored, or documented as an intentional fixture.

---
Source: Action-Tracker task note `2026-07-22-evaluate-db-schema-skills-across-db-backed-repos.md`

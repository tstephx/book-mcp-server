# ADR 007: Slug ID Resolution for Book Tool Parameters

**Status:** Accepted
**Date:** 2026-02-24

## Context

All MCP tools that operate on a specific book require a `book_id` parameter. Book IDs
are UUIDs (e.g., `a3f2c1d0-...`), which Claude cannot guess and users cannot remember.
This made tools like `get_book_info` and `get_chapter` awkward — you had to first call
`list_books` to find the ID before calling the actual tool.

## Decision

Replace `validate_book_id()` with `resolve_book_id()` in `src/utils/validators.py`.
Resolution follows a two-step fallback:

1. **UUID fast path** — if the input is a valid UUID format, look it up directly
2. **Fuzzy title match** — `SELECT id FROM books WHERE title LIKE '%<input>%' LIMIT 1`
3. **Did-you-mean error** — if no match, return a helpful error suggesting the closest
   title (matched by first significant word)

All call sites in `book_tools.py`, `chapter_tools.py`, and `server.py` use
`resolve_book_id()`. Schema field validators in `tool_schemas.py` accept any non-empty
string — resolution is deferred to query time, not at schema validation.

## Consequences

- Claude (and users) can now pass partial titles like `"clean code"` instead of a UUID
- Fuzzy matching is case-insensitive via LIKE — sufficient for single-word or phrase slugs
- The did-you-mean fallback uses first-word heuristics, which is weak for short/common
  words (known limitation — improvement deferred)
- No performance impact: UUID fast path skips the LIKE query for well-formed IDs
- If multiple books match a partial title, the first alphabetical match wins (arbitrary
  but deterministic)

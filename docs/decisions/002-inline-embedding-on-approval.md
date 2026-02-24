# ADR 002: Inline Embedding During Approval

**Status:** Accepted
**Date:** 2026-01-28

## Context

After a book is approved, it needs embeddings generated before it appears in
semantic search. Two approaches:
1. Separate embedding worker that polls for APPROVED books
2. Run embedding inline inside `approve_book()` before returning

## Decision

Run the full `APPROVED → EMBEDDING → COMPLETE` flow inline inside `approve_book()`.
No separate embedding worker process is needed.

## Consequences

- Approval calls block until embedding completes (can take 30-120s for large books)
- Simpler operations: one process to manage instead of two
- MCP approval tool (`approve` command) returns only after the book is fully searchable
- Embedding timeout configured via `EMBEDDING_TIMEOUT_SECONDS` (default 300s)
- If embedding fails, book stays in EMBEDDING state and can be retried

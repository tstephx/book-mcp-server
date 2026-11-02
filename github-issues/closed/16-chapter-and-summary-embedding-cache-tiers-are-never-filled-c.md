---
title: "Chapter and summary embedding cache tiers are never filled; clear_cache("embeddings"/"summary_embeddings") clears empty tiers"
number: 16
state: closed
labels: [bug]
repo: book-mcp-server
url: https://github.com/tstephx/book-mcp-server/issues/16
created: 2026-10-30
updated: 2026-11-02
---

## Problem

Two of the three embedding cache tiers in `src/utils/cache.py` are never filled by production code, so the `clear_cache` tool's `"embeddings"` and `"summary_embeddings"` options clear empty tiers.

- `get_embeddings` / `set_embeddings` (`cache.py:74`, `:92`): called only from `tests/test_summary_embeddings.py:306,312`
- `get_summary_embeddings` / `set_summary_embeddings` (`cache.py:176`, `:194`): tests only (6 and 5 test references, 0 in `src/`)
- `src/server.py:189-193` `clear_cache("embeddings")` and `clear_cache("summary_embeddings")` invalidate those two tiers; `embedding_sync.py:301` does the same
- `invalidate_chunk_embeddings` (`cache.py:165`) has no caller at all, so the chunk-embedding cache that search actually uses can only be cleared via `clear_all`

Net effect: the documented per-tier cache clearing does not clear the tier that matters, and two tiers exist only to satisfy tests.

## Suggested fix

Either fill the chapter and summary tiers from the paths that compute those embeddings (`generate_summary_embeddings` in `server.py:120`, the chapter path in `embedding_sync.py`), or delete the two unused tiers, their `clear_cache` branches and their tests, and give the chunk tier a real targeted invalidation. Deleting is the smaller change.

## Evidence

Verified against `main` at 8ade59a with `rg -w` across `src/`, `agentic_pipeline/`, `scripts/`, `tests/`. Found by the tech-debt scan, pass 2 (vault `Research/Project-Audits/2026-10-28-tech-debt-scan-wave-1-python.md`).

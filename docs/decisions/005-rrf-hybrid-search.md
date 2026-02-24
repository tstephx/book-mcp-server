# ADR 005: RRF Hybrid Search (FTS5 + Semantic)

**Status:** Accepted
**Date:** 2025-12-22

## Context

Book search needs to handle both exact-match queries ("what does WAL mode mean")
and conceptual queries ("explain container isolation"). Neither FTS5 keyword
search nor pure semantic search handles both cases well alone.

## Decision

Use Reciprocal Rank Fusion (RRF) to combine FTS5 keyword search results with
semantic vector search results. Optional MMR (Maximal Marginal Relevance)
re-ranking for diversity when requested.

## Consequences

- Both query types return good results without tuning per-query
- Slightly higher latency than either search alone (two queries + merge)
- RRF is parameter-free (uses standard k=60 constant) — no threshold tuning needed
- FTS5 index and embedding vectors must both be populated for full hybrid benefit
- `hybrid_search` tool in `src/tools/hybrid_search_tools.py`

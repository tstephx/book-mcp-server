# ADR 003: OpenAI text-embedding-3-large for Semantic Search

**Status:** Accepted
**Date:** 2026-02-14

## Context

The book library uses vector embeddings for semantic search across 185+ books.
Migration from `text-embedding-3-small` (1536 dims) to `text-embedding-3-large`
(3072 dims) was evaluated after observing poor semantic search relevance on
technical topics.

## Decision

Use `text-embedding-3-large` (3072 dimensions) for all new and re-generated
embeddings. Existing embeddings were migrated in full (no mixed-model index).

## Consequences

- Significantly better semantic search relevance on technical/domain-specific queries
- 2x storage per embedding vector (3072 vs 1536 floats)
- Higher OpenAI API cost per embedding call
- `OPENAI_API_KEY` required in environment for both MCP server and pipeline worker
- All tests asserting model name or dimensions must use `text-embedding-3-large` / `3072`

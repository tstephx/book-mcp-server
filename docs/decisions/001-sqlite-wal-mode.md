# ADR 001: SQLite with WAL Mode for Pipeline Database

**Status:** Accepted
**Date:** 2026-01-28

## Context

The agentic pipeline needs a persistent store for book state (pipeline stages,
metadata, approval history). Options considered: PostgreSQL, SQLite default
journal mode, SQLite WAL mode.

The pipeline runs on a single machine with a single writer (the worker process)
and occasional readers (CLI commands, MCP server). No cross-machine distribution
is needed.

## Decision

Use SQLite with WAL (Write-Ahead Logging) mode. All pipeline connections go
through `get_pipeline_db()` with `timeout=10` and `row_factory=sqlite3.Row`.

## Consequences

- No external database dependency — pipeline works out of the box
- WAL mode allows concurrent reads while writing (CLI + worker don't block each other)
- 10-second timeout prevents indefinite lock waits with a clear error
- Single-machine only — scaling to distributed workers would require a different store
- DB path configured via `AGENTIC_PIPELINE_DB` env var

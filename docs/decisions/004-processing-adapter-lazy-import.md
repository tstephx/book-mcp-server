# ADR 004: ProcessingAdapter with Lazy Import

**Status:** Accepted
**Date:** 2026-01-28

## Context

Book processing logic lives in the `book-ingestion-python` project. Options for
reuse:
1. Copy processing code into this repo
2. Install book-ingestion as a Python package dependency
3. Import it at runtime as a sibling project (lazy import with path manipulation)

## Decision

Use `ProcessingAdapter` in `agentic_pipeline/processing_adapter.py` which
lazy-imports `book-ingestion-python` at call time. The sibling project path
is resolved relative to this project's root.

## Consequences

- No code duplication — single source of truth for processing logic
- No package publishing required — works with local dev checkouts
- Import errors surface at processing time, not at server startup
- Both projects must exist as siblings on the same filesystem
- Path resolution is fragile if directory layout changes
- Tests mock `ProcessingAdapter` to avoid the cross-project dependency

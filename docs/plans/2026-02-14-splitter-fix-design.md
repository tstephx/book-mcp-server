# EPUB Chapter Splitter Fix — Design

## Problem

78 of 206 library books (38%) fail our new extraction quality gates. Root cause: the chapter splitter in `book-ingestion-python` silently degrades when anchor/TOC detection partially fails, producing books with 1-3 mega-chapters instead of triggering the fixed-size fallback.

### Failure Chain

1. EPUB anchor resolution finds only 2-3 of 10+ expected chapters (fingerprint mismatch after text cleaning)
2. Since `len(anchors) >= MIN_ANCHORS (3)`, the splitter uses them instead of falling back
3. Result: 3 chapters at 15-50k words each — passes confidence checks but fails quality checks
4. No post-split validation catches this before the book enters the library

### Failure Modes (from audit)

| Issue | Count | Description |
|-------|-------|-------------|
| Oversized chapters | 43 | Chapters exceeding 20,000 words |
| Too few chapters | 39 | Fewer than 7 chapters detected |
| Lopsided ratios | 34 | Largest chapter >4x the median |
| Low total words | 5 | Under 5,000 total words captured |
| Suspicious titles | 2 | File paths or table references as chapter titles |

## Solution: Hybrid Fix (Two Projects)

### Part 1: Quality-Aware Fallback in book-ingestion-python

Add a post-split quality gate inside `ChapterSplitter.split_with_stats()` that mirrors our extraction validator thresholds:

1. After initial split, check: `chapter_count >= 7`, no chapter > 20k words, max/median ratio <= 4x
2. If check fails, identify oversized chapters and re-split at heading boundaries (blank line + title-case line) or paragraph boundaries (double newlines)
3. Re-check quality. If still fails, fall back to fixed-size 2500-word chunks for oversized chapters only
4. Log which fallback was used and why

Key principle: never degrade a good split. Only re-split chapters exceeding thresholds.

Support a `force_fallback=True` parameter that skips anchor detection entirely and uses heading-based + size-based splitting directly (used by pipeline retry).

### Part 2: Pipeline Retry in book-mcp-server

When a book fails `ExtractionValidator` at VALIDATING, retry once with `force_fallback=True`:

1. First validation failure: transition to `NEEDS_RETRY`, store failure reasons in metadata
2. On retry: call `ProcessingAdapter` with `force_fallback=True`
3. Second validation failure: reject with both attempt results logged
4. Max 1 retry — no infinite loops

New metadata fields on `processing_pipelines`: `retry_count` (int), `retry_reason` (text).

### Part 3: Reprocess 78 Flagged Books

CLI command `agentic-pipeline reprocess --flagged`:

- Re-runs quality check against library, identifies failing books
- For each: deletes existing chapters, resets pipeline state to `DETECTED`, re-queues
- Dry-run by default (`--execute` to actually run)
- Backs up old chapter counts in audit trail
- Processes one book at a time

## Testing

**book-ingestion-python:**
- Unit tests for quality gate (thresholds, re-splitting, fallback)
- Integration tests with real EPUBs that currently produce bad splits
- Regression tests for books that already split well

**book-mcp-server:**
- Unit tests for retry mechanism (NEEDS_RETRY transitions, max 1 retry)
- Integration test for reprocess CLI command
- End-to-end: mock book fails validation, retries with force_fallback, verify final state

## Scope

- EPUBs only (PDFs deferred to separate effort)
- Changes in both `book-ingestion-python` and `book-mcp-server`
- Reprocess all 78 flagged books after fix is deployed

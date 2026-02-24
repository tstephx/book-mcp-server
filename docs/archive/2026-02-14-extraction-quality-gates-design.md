<!-- project: book-mcp-server -->

# Extraction Quality Gates

**Date:** 2026-02-14
**Status:** Approved
**Problem:** The VALIDATING pipeline state is a no-op. Books with 1 chapter, 49k-word mega-chapters, and nonsensical titles pass with 0.95 confidence because the confidence score reflects classifier certainty about book *type*, not extraction *quality*.

## Thresholds (Very Strict)

| Check | Threshold | Action |
|-------|-----------|--------|
| Min chapters | 7 | Reject |
| Max chapter word count | 20,000 | Reject |
| Max-to-median chapter ratio | 4.0x | Reject |
| Min chapter word count | 100 | Warning only |
| Min total word count | 5,000 | Reject |
| Max duplicate chapter ratio | 10% | Reject |
| Suspicious chapter titles | Pattern match | Reject |

Suspicious title patterns: file extensions (`.zip`, `.exe`, `.rar`), table references (`Table \d+`), bare numbers, filesystem paths.

All reasons are collected (not short-circuited) so the rejection message lists every problem.

## Components

### 1. ExtractionValidator

**Location:** `agentic_pipeline/validation/extraction_validator.py`

```python
@dataclass
class ValidationResult:
    passed: bool
    reasons: list[str]       # empty if passed
    warnings: list[str]      # non-fatal issues (e.g. tiny chapters)
    metrics: dict            # chapter_count, max_word_count, median_word_count, ratio, etc.

class ExtractionValidator:
    MIN_CHAPTERS = 7
    MAX_CHAPTER_WORDS = 20_000
    MAX_TO_MEDIAN_RATIO = 4.0
    MIN_CHAPTER_WORDS = 100
    MIN_TOTAL_WORDS = 5_000
    MAX_DUPLICATE_RATIO = 0.1

    def validate(self, book_id: str, db_path: str) -> ValidationResult
```

Queries `chapters` table for the book's per-chapter word counts, titles, and content hashes. Runs all 7 checks.

### 2. Shared Check Logic

**Location:** `agentic_pipeline/validation/__init__.py`

A pure function `check_extraction_quality(chapter_count, word_counts, titles, content_hashes) -> ValidationResult` that both `ExtractionValidator.validate()` and the retroactive audit call. No DB dependency — takes pre-fetched data.

### 3. Orchestrator Integration

**Location:** `orchestrator.py` at the VALIDATING state (~line 266)

```python
# VALIDATING
self._transition(pipeline_id, PipelineState.VALIDATING)
validator = ExtractionValidator()
validation = validator.validate(book_id=pipeline_id, db_path=self.config.db_path)

if not validation.passed:
    self._transition(pipeline_id, PipelineState.REJECTED)
    reason = "; ".join(validation.reasons)
    self.repo.update_state(pipeline_id, PipelineState.REJECTED,
                           error_details={"validation_reasons": validation.reasons})
    return {"pipeline_id": pipeline_id, "state": "rejected", "reason": reason}
```

No state machine changes needed — VALIDATING can already reach REJECTED.

### 4. Retroactive Library Audit

**CLI command:** `agentic-pipeline audit-quality [--json]`

Runs the same 7 checks against all books in the library. Read-only — reports flagged books but doesn't modify anything.

```
Quality Audit: 206 books
  Pass: 174
  Fail:  32

Flagged books:
  Clean Code                   | 1 ch  | Only 1 chapters extracted (min 7)
  ...
```

Uses `LibraryValidator` in `backfill/validator.py` extended with the shared `check_extraction_quality()` function.

### 5. Tests

**Location:** `tests/test_extraction_validator.py`

- Unit tests for each of the 7 checks in isolation (synthetic data, temp DB)
- Edge cases: exactly-at-threshold, 0 chapters, all chapters identical
- Orchestrator integration: bad extraction auto-rejects at VALIDATING
- CLI audit: runs against test DB, produces correct output

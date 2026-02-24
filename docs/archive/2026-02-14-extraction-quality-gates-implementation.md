<!-- project: book-mcp-server -->

# Extraction Quality Gates Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add extraction quality validation to the pipeline so books with bad splits (too few chapters, mega-chapters, garbage titles, duplicates) are auto-rejected at the VALIDATING state instead of reaching approval.

**Architecture:** A pure function `check_extraction_quality()` holds all threshold logic. `ExtractionValidator` wraps it with DB queries. The orchestrator calls the validator at VALIDATING. A CLI command `audit-quality` runs the same checks retroactively against the full library.

**Tech Stack:** Python, SQLite, Click, Rich, pytest

---

### Task 1: Create validation package with shared check logic

**Files:**
- Create: `agentic_pipeline/validation/__init__.py`
- Create: `agentic_pipeline/validation/extraction_validator.py`
- Test: `tests/test_extraction_validator.py`

**Step 1: Write failing tests for the pure check function**

Create `tests/test_extraction_validator.py`:

```python
"""Tests for extraction quality validation."""

import pytest
from agentic_pipeline.validation import check_extraction_quality


class TestCheckExtractionQuality:
    """Tests for the pure check_extraction_quality function."""

    def test_good_book_passes(self):
        """A well-extracted book passes all checks."""
        result = check_extraction_quality(
            chapter_count=12,
            word_counts=[3000, 4000, 3500, 4200, 3800, 4100, 3700, 3900, 4000, 3600, 4300, 3500],
            titles=["Chapter 1: Intro", "Chapter 2: Basics", "Chapter 3: Advanced",
                    "Chapter 4: Patterns", "Chapter 5: Testing", "Chapter 6: Deploy",
                    "Chapter 7: Monitor", "Chapter 8: Scale", "Chapter 9: Security",
                    "Chapter 10: Debug", "Chapter 11: Optimize", "Chapter 12: Summary"],
            content_hashes=[f"hash_{i}" for i in range(12)],
        )
        assert result.passed is True
        assert result.reasons == []

    def test_too_few_chapters_rejects(self):
        result = check_extraction_quality(
            chapter_count=3,
            word_counts=[5000, 6000, 5500],
            titles=["Ch 1", "Ch 2", "Ch 3"],
            content_hashes=["a", "b", "c"],
        )
        assert result.passed is False
        assert any("min 7" in r for r in result.reasons)

    def test_exactly_7_chapters_passes(self):
        result = check_extraction_quality(
            chapter_count=7,
            word_counts=[3000] * 7,
            titles=[f"Chapter {i}" for i in range(7)],
            content_hashes=[f"hash_{i}" for i in range(7)],
        )
        assert result.passed is True

    def test_mega_chapter_rejects(self):
        word_counts = [3000] * 9 + [25000]
        result = check_extraction_quality(
            chapter_count=10,
            word_counts=word_counts,
            titles=[f"Chapter {i}" for i in range(10)],
            content_hashes=[f"hash_{i}" for i in range(10)],
        )
        assert result.passed is False
        assert any("20,000" in r or "20000" in r for r in result.reasons)

    def test_exactly_20k_words_passes(self):
        word_counts = [3000] * 9 + [20000]
        result = check_extraction_quality(
            chapter_count=10,
            word_counts=word_counts,
            titles=[f"Chapter {i}" for i in range(10)],
            content_hashes=[f"hash_{i}" for i in range(10)],
        )
        # 20000 is at threshold, not over — should pass the max check
        # But 20000 / median(3000) = 6.67x which fails the ratio check
        assert result.passed is False
        assert any("median" in r.lower() for r in result.reasons)

    def test_lopsided_ratio_rejects(self):
        word_counts = [2000] * 9 + [10000]
        result = check_extraction_quality(
            chapter_count=10,
            word_counts=word_counts,
            titles=[f"Chapter {i}" for i in range(10)],
            content_hashes=[f"hash_{i}" for i in range(10)],
        )
        assert result.passed is False
        assert any("median" in r.lower() for r in result.reasons)

    def test_exactly_4x_median_passes(self):
        # median of [3000]*6 + [12000] = 3000, ratio = 4.0 exactly
        word_counts = [3000] * 6 + [12000]
        result = check_extraction_quality(
            chapter_count=7,
            word_counts=word_counts,
            titles=[f"Chapter {i}" for i in range(7)],
            content_hashes=[f"hash_{i}" for i in range(7)],
        )
        assert result.passed is True

    def test_tiny_chapter_warns_but_passes(self):
        word_counts = [3000] * 6 + [50]
        result = check_extraction_quality(
            chapter_count=7,
            word_counts=word_counts,
            titles=[f"Chapter {i}" for i in range(7)],
            content_hashes=[f"hash_{i}" for i in range(7)],
        )
        assert result.passed is True
        assert any("100" in w for w in result.warnings)

    def test_low_total_words_rejects(self):
        word_counts = [500] * 8
        result = check_extraction_quality(
            chapter_count=8,
            word_counts=word_counts,
            titles=[f"Chapter {i}" for i in range(8)],
            content_hashes=[f"hash_{i}" for i in range(8)],
        )
        assert result.passed is False
        assert any("5,000" in r or "5000" in r for r in result.reasons)

    def test_duplicate_chapters_rejects(self):
        # 3 out of 10 share the same hash = 30% > 10%
        content_hashes = ["unique_1", "unique_2", "dupe", "unique_3", "dupe",
                          "unique_4", "unique_5", "dupe", "unique_6", "unique_7"]
        result = check_extraction_quality(
            chapter_count=10,
            word_counts=[3000] * 10,
            titles=[f"Chapter {i}" for i in range(10)],
            content_hashes=content_hashes,
        )
        assert result.passed is False
        assert any("duplicate" in r.lower() for r in result.reasons)

    def test_10_percent_duplicates_passes(self):
        # 1 out of 10 is a dupe = 10%, at threshold
        content_hashes = ["unique_1", "unique_2", "unique_3", "unique_4", "unique_5",
                          "unique_6", "unique_7", "unique_8", "unique_9", "unique_1"]
        result = check_extraction_quality(
            chapter_count=10,
            word_counts=[3000] * 10,
            titles=[f"Chapter {i}" for i in range(10)],
            content_hashes=content_hashes,
        )
        # 1 duplicate out of 10 = 10%, which is at threshold (not over)
        assert result.passed is True

    def test_suspicious_title_file_extension_rejects(self):
        titles = ["Chapter 1", "Chapter 2", "Chapter 3", "Chapter 4",
                  "Chapter 5", "Chapter 6", "WinRAR/7-Zip for Windows"]
        result = check_extraction_quality(
            chapter_count=7,
            word_counts=[3000] * 7,
            titles=titles,
            content_hashes=[f"hash_{i}" for i in range(7)],
        )
        assert result.passed is False
        assert any("WinRAR" in r or "suspicious" in r.lower() for r in result.reasons)

    def test_suspicious_title_table_reference_rejects(self):
        titles = [f"Chapter {i}" for i in range(6)] + ["Table 4.8"]
        result = check_extraction_quality(
            chapter_count=7,
            word_counts=[3000] * 7,
            titles=titles,
            content_hashes=[f"hash_{i}" for i in range(7)],
        )
        assert result.passed is False
        assert any("Table 4.8" in r for r in result.reasons)

    def test_suspicious_title_bare_number_rejects(self):
        titles = [f"Chapter {i}" for i in range(6)] + ["42"]
        result = check_extraction_quality(
            chapter_count=7,
            word_counts=[3000] * 7,
            titles=titles,
            content_hashes=[f"hash_{i}" for i in range(7)],
        )
        assert result.passed is False
        assert any("42" in r for r in result.reasons)

    def test_zero_chapters_rejects(self):
        result = check_extraction_quality(
            chapter_count=0,
            word_counts=[],
            titles=[],
            content_hashes=[],
        )
        assert result.passed is False
        assert any("min 7" in r for r in result.reasons)

    def test_all_identical_chapters_rejects(self):
        result = check_extraction_quality(
            chapter_count=10,
            word_counts=[3000] * 10,
            titles=[f"Chapter {i}" for i in range(10)],
            content_hashes=["same_hash"] * 10,
        )
        assert result.passed is False
        assert any("duplicate" in r.lower() for r in result.reasons)

    def test_multiple_failures_collected(self):
        """All reasons are collected, not short-circuited."""
        result = check_extraction_quality(
            chapter_count=2,
            word_counts=[50, 30000],
            titles=["Table 1.1", "setup.exe"],
            content_hashes=["a", "a"],
        )
        assert result.passed is False
        assert len(result.reasons) >= 3  # few chapters + mega chapter + duplicates + titles

    def test_metrics_populated(self):
        result = check_extraction_quality(
            chapter_count=10,
            word_counts=[3000] * 10,
            titles=[f"Chapter {i}" for i in range(10)],
            content_hashes=[f"hash_{i}" for i in range(10)],
        )
        assert result.metrics["chapter_count"] == 10
        assert result.metrics["total_words"] == 30000
        assert result.metrics["max_word_count"] == 3000
        assert result.metrics["median_word_count"] == 3000
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_extraction_validator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_pipeline.validation'`

**Step 3: Create the validation package and implement `check_extraction_quality`**

Create `agentic_pipeline/validation/__init__.py`:

```python
"""Extraction quality validation."""

from agentic_pipeline.validation.extraction_validator import (
    check_extraction_quality,
    ExtractionValidator,
    ValidationResult,
)

__all__ = ["check_extraction_quality", "ExtractionValidator", "ValidationResult"]
```

Create `agentic_pipeline/validation/extraction_validator.py`:

```python
"""Extraction quality validation for processed books."""

import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from agentic_pipeline.db.connection import get_pipeline_db

# Thresholds (very strict)
MIN_CHAPTERS = 7
MAX_CHAPTER_WORDS = 20_000
MAX_TO_MEDIAN_RATIO = 4.0
MIN_CHAPTER_WORDS = 100
MIN_TOTAL_WORDS = 5_000
MAX_DUPLICATE_RATIO = 0.1

# Suspicious title patterns
_SUSPICIOUS_PATTERNS = [
    re.compile(r"\.(zip|exe|rar|msi|dmg|pkg|tar|gz|7z)\b", re.IGNORECASE),
    re.compile(r"^Table\s+\d+", re.IGNORECASE),
    re.compile(r"^\d+$"),
    re.compile(r"[/\\][A-Za-z]:[/\\]"),  # Windows paths
    re.compile(r"^[/\\](?:usr|tmp|var|etc|home)[/\\]", re.IGNORECASE),  # Unix paths
]


@dataclass
class ValidationResult:
    """Result of extraction quality validation."""

    passed: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def check_extraction_quality(
    chapter_count: int,
    word_counts: list[int],
    titles: list[str],
    content_hashes: list[str],
) -> ValidationResult:
    """Check extraction quality against thresholds.

    Pure function — no DB dependency. Both the pipeline validator
    and the retroactive audit call this.

    Args:
        chapter_count: Number of chapters extracted.
        word_counts: Per-chapter word counts.
        titles: Per-chapter titles.
        content_hashes: Per-chapter content hashes for duplicate detection.

    Returns:
        ValidationResult with pass/fail, reasons, warnings, and metrics.
    """
    reasons = []
    warnings = []

    total_words = sum(word_counts) if word_counts else 0
    max_wc = max(word_counts) if word_counts else 0
    median_wc = int(statistics.median(word_counts)) if word_counts else 0

    # 1. Min chapters
    if chapter_count < MIN_CHAPTERS:
        reasons.append(f"Only {chapter_count} chapters extracted (min {MIN_CHAPTERS})")

    # 2. Max chapter word count
    for i, wc in enumerate(word_counts):
        if wc > MAX_CHAPTER_WORDS:
            title = titles[i] if i < len(titles) else f"Chapter {i}"
            reasons.append(
                f"Chapter '{title}' has {wc:,} words (max {MAX_CHAPTER_WORDS:,})"
            )

    # 3. Max-to-median ratio
    if median_wc > 0 and max_wc / median_wc > MAX_TO_MEDIAN_RATIO:
        ratio = max_wc / median_wc
        reasons.append(
            f"Largest chapter is {ratio:.1f}x median ({max_wc:,} vs {median_wc:,}) "
            f"— likely missed splits (max {MAX_TO_MEDIAN_RATIO}x)"
        )

    # 4. Min chapter word count (warning only)
    for i, wc in enumerate(word_counts):
        if wc < MIN_CHAPTER_WORDS:
            title = titles[i] if i < len(titles) else f"Chapter {i}"
            warnings.append(
                f"Chapter '{title}' has only {wc} words (under {MIN_CHAPTER_WORDS})"
            )

    # 5. Min total words
    if total_words < MIN_TOTAL_WORDS:
        reasons.append(
            f"Only {total_words:,} total words extracted (min {MIN_TOTAL_WORDS:,})"
        )

    # 6. Duplicate chapters
    if content_hashes:
        hash_counts = Counter(content_hashes)
        duplicate_count = sum(c - 1 for c in hash_counts.values() if c > 1)
        duplicate_ratio = duplicate_count / len(content_hashes)
        if duplicate_ratio > MAX_DUPLICATE_RATIO:
            reasons.append(
                f"{duplicate_ratio:.0%} duplicate chapters detected "
                f"(max {MAX_DUPLICATE_RATIO:.0%})"
            )

    # 7. Suspicious titles
    for title in titles:
        for pattern in _SUSPICIOUS_PATTERNS:
            if pattern.search(title):
                reasons.append(
                    f"Suspicious chapter title: '{title}' — looks like TOC fallback artifact"
                )
                break

    metrics = {
        "chapter_count": chapter_count,
        "total_words": total_words,
        "max_word_count": max_wc,
        "median_word_count": median_wc,
        "max_to_median_ratio": round(max_wc / median_wc, 1) if median_wc > 0 else 0,
    }

    return ValidationResult(
        passed=len(reasons) == 0,
        reasons=reasons,
        warnings=warnings,
        metrics=metrics,
    )


class ExtractionValidator:
    """Validates extraction quality by querying the chapters table."""

    def validate(self, book_id: str, db_path: str) -> ValidationResult:
        """Validate a processed book's extraction quality.

        Args:
            book_id: The book ID (same as pipeline_id).
            db_path: Path to the library database.

        Returns:
            ValidationResult from check_extraction_quality.
        """
        with get_pipeline_db(db_path) as conn:
            rows = conn.execute(
                "SELECT title, word_count, content_hash FROM chapters WHERE book_id = ? ORDER BY chapter_number",
                (book_id,),
            ).fetchall()

        chapter_count = len(rows)
        word_counts = [row["word_count"] or 0 for row in rows]
        titles = [row["title"] or "" for row in rows]
        content_hashes = [row["content_hash"] or "" for row in rows]

        return check_extraction_quality(chapter_count, word_counts, titles, content_hashes)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_extraction_validator.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add agentic_pipeline/validation/__init__.py agentic_pipeline/validation/extraction_validator.py tests/test_extraction_validator.py
git commit -m "feat: add extraction quality validation with 7 checks"
```

---

### Task 2: Wire ExtractionValidator into the orchestrator

**Files:**
- Modify: `agentic_pipeline/orchestrator/orchestrator.py:265-274`
- Test: `tests/test_extraction_validator.py` (add integration test)

**Step 1: Write failing integration test**

Add to `tests/test_extraction_validator.py`:

```python
import tempfile
import sqlite3
from agentic_pipeline.db.migrations import run_migrations
from agentic_pipeline.validation import ExtractionValidator


class TestExtractionValidatorDB:
    """Integration tests for ExtractionValidator with real DB."""

    def _create_test_db(self):
        """Create a temp DB with library tables."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = tmp.name
        tmp.close()
        run_migrations(db_path)
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS books (
                id TEXT PRIMARY KEY, title TEXT, author TEXT, word_count INTEGER,
                source_file TEXT, processing_status TEXT, added_date TEXT,
                book_type TEXT, classification_confidence REAL, suggested_tags TEXT,
                classification_reasoning TEXT, classified_at TEXT, classified_by TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chapters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id TEXT, chapter_number INTEGER, title TEXT,
                file_path TEXT, word_count INTEGER, embedding BLOB,
                embedding_model TEXT, content_hash TEXT, file_mtime REAL,
                embedding_updated_at TEXT
            )
        """)
        conn.commit()
        conn.close()
        return db_path

    def test_good_book_passes_db(self):
        db_path = self._create_test_db()
        conn = sqlite3.connect(db_path)
        conn.execute("INSERT INTO books (id, title) VALUES ('book1', 'Good Book')")
        for i in range(10):
            conn.execute(
                "INSERT INTO chapters (book_id, chapter_number, title, word_count, content_hash, file_path) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("book1", i, f"Chapter {i}: Topic", 4000, f"hash_{i}", f"path/{i}.md"),
            )
        conn.commit()
        conn.close()

        validator = ExtractionValidator()
        result = validator.validate("book1", db_path)
        assert result.passed is True

    def test_bad_book_rejected_db(self):
        db_path = self._create_test_db()
        conn = sqlite3.connect(db_path)
        conn.execute("INSERT INTO books (id, title) VALUES ('book2', 'Bad Book')")
        conn.execute(
            "INSERT INTO chapters (book_id, chapter_number, title, word_count, content_hash, file_path) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("book2", 1, "Table 4.8", 49000, "hash_1", "path/1.md"),
        )
        conn.commit()
        conn.close()

        validator = ExtractionValidator()
        result = validator.validate("book2", db_path)
        assert result.passed is False
        assert len(result.reasons) >= 2  # few chapters + mega chapter + suspicious title

    def test_book_with_no_chapters(self):
        db_path = self._create_test_db()
        conn = sqlite3.connect(db_path)
        conn.execute("INSERT INTO books (id, title) VALUES ('book3', 'Empty')")
        conn.commit()
        conn.close()

        validator = ExtractionValidator()
        result = validator.validate("book3", db_path)
        assert result.passed is False
```

**Step 2: Run tests to verify they pass**

Run: `python -m pytest tests/test_extraction_validator.py::TestExtractionValidatorDB -v`
Expected: PASS (ExtractionValidator implementation already exists from Task 1)

**Step 3: Wire into the orchestrator**

In `agentic_pipeline/orchestrator/orchestrator.py`, replace lines 265-267:

```python
        # VALIDATING
        self._transition(pipeline_id, PipelineState.VALIDATING)
        # Validation now uses processing result quality metrics
```

With:

```python
        # VALIDATING
        self._transition(pipeline_id, PipelineState.VALIDATING)
        from agentic_pipeline.validation import ExtractionValidator

        validator = ExtractionValidator()
        validation = validator.validate(
            book_id=pipeline_id, db_path=str(self.config.db_path)
        )

        if not validation.passed:
            reason = "; ".join(validation.reasons)
            self.logger.error(pipeline_id, "ValidationFailed", reason)
            self.repo.update_state(
                pipeline_id,
                PipelineState.REJECTED,
                error_details={"validation_reasons": validation.reasons, "metrics": validation.metrics},
            )
            return {
                "pipeline_id": pipeline_id,
                "state": PipelineState.REJECTED.value,
                "reason": reason,
                "metrics": validation.metrics,
            }
```

**Step 4: Run full test suite to verify no regressions**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add agentic_pipeline/orchestrator/orchestrator.py tests/test_extraction_validator.py
git commit -m "feat: wire extraction validator into orchestrator VALIDATING state"
```

---

### Task 3: Add `audit-quality` CLI command

**Files:**
- Modify: `agentic_pipeline/cli.py`
- Test: `tests/test_extraction_validator.py` (add CLI test)

**Step 1: Write failing test for the CLI command**

Add to `tests/test_extraction_validator.py`:

```python
from click.testing import CliRunner
from agentic_pipeline.cli import main


class TestAuditQualityCLI:
    """Tests for the audit-quality CLI command."""

    def test_audit_quality_runs(self, monkeypatch, tmp_path):
        """audit-quality command runs and produces output."""
        import sqlite3
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        # Create minimal tables
        conn.execute("""
            CREATE TABLE books (
                id TEXT PRIMARY KEY, title TEXT, author TEXT, word_count INTEGER,
                source_file TEXT, processing_status TEXT, added_date TEXT,
                book_type TEXT, classification_confidence REAL, suggested_tags TEXT,
                classification_reasoning TEXT, classified_at TEXT, classified_by TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE chapters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id TEXT, chapter_number INTEGER, title TEXT,
                file_path TEXT, word_count INTEGER, embedding BLOB,
                embedding_model TEXT, content_hash TEXT, file_mtime REAL,
                embedding_updated_at TEXT
            )
        """)
        # Insert a good book
        conn.execute("INSERT INTO books (id, title) VALUES ('good', 'Good Book')")
        for i in range(10):
            conn.execute(
                "INSERT INTO chapters (book_id, chapter_number, title, word_count, content_hash, file_path) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("good", i, f"Chapter {i}", 4000, f"hash_{i}", f"p/{i}.md"),
            )
        # Insert a bad book
        conn.execute("INSERT INTO books (id, title) VALUES ('bad', 'Bad Book')")
        conn.execute(
            "INSERT INTO chapters (book_id, chapter_number, title, word_count, content_hash, file_path) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("bad", 1, "Only Chapter", 500, "hash_0", "p/0.md"),
        )
        conn.commit()
        conn.close()

        monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(db_path))
        runner = CliRunner()
        result = runner.invoke(main, ["audit-quality"])
        assert result.exit_code == 0
        assert "Good Book" not in result.output or "pass" in result.output.lower()
        assert "Bad Book" in result.output

    def test_audit_quality_json(self, monkeypatch, tmp_path):
        """audit-quality --json produces parseable JSON."""
        import sqlite3
        import json as json_module
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE books (
                id TEXT PRIMARY KEY, title TEXT, author TEXT, word_count INTEGER,
                source_file TEXT, processing_status TEXT, added_date TEXT,
                book_type TEXT, classification_confidence REAL, suggested_tags TEXT,
                classification_reasoning TEXT, classified_at TEXT, classified_by TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE chapters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id TEXT, chapter_number INTEGER, title TEXT,
                file_path TEXT, word_count INTEGER, embedding BLOB,
                embedding_model TEXT, content_hash TEXT, file_mtime REAL,
                embedding_updated_at TEXT
            )
        """)
        conn.execute("INSERT INTO books (id, title) VALUES ('b1', 'Test Book')")
        conn.execute(
            "INSERT INTO chapters (book_id, chapter_number, title, word_count, content_hash, file_path) "
            "VALUES ('b1', 1, 'Ch1', 200, 'h1', 'p/1.md')",
        )
        conn.commit()
        conn.close()

        monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(db_path))
        runner = CliRunner()
        result = runner.invoke(main, ["audit-quality", "--json"])
        assert result.exit_code == 0
        data = json_module.loads(result.output)
        assert "total" in data
        assert "flagged" in data
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_extraction_validator.py::TestAuditQualityCLI -v`
Expected: FAIL — `audit-quality` command doesn't exist yet

**Step 3: Implement the CLI command**

Add to `agentic_pipeline/cli.py` (after existing `audit` command):

```python
@main.command("audit-quality")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def audit_quality(as_json: bool):
    """Audit library books for extraction quality issues."""
    import json as json_module
    from .db.config import get_db_path
    from .db.connection import get_pipeline_db
    from .validation import check_extraction_quality

    db_path = get_db_path()

    with get_pipeline_db(db_path) as conn:
        books = conn.execute("SELECT id, title FROM books").fetchall()

        results = []
        for book in books:
            rows = conn.execute(
                "SELECT title, word_count, content_hash FROM chapters WHERE book_id = ? ORDER BY chapter_number",
                (book["id"],),
            ).fetchall()

            chapter_count = len(rows)
            word_counts = [r["word_count"] or 0 for r in rows]
            titles = [r["title"] or "" for r in rows]
            content_hashes = [r["content_hash"] or "" for r in rows]

            validation = check_extraction_quality(chapter_count, word_counts, titles, content_hashes)

            if not validation.passed:
                results.append({
                    "book_id": book["id"],
                    "title": book["title"],
                    "chapter_count": chapter_count,
                    "reasons": validation.reasons,
                    "metrics": validation.metrics,
                })

    if as_json:
        console.print(json_module.dumps({
            "total": len(books),
            "passed": len(books) - len(results),
            "flagged": len(results),
            "books": results,
        }, indent=2))
        return

    console.print(f"\n[bold]Quality Audit: {len(books)} books[/bold]")
    console.print(f"  [green]Pass: {len(books) - len(results)}[/green]")
    console.print(f"  [red]Fail: {len(results)}[/red]\n")

    if not results:
        console.print("[green]All books pass quality checks.[/green]")
        return

    table = Table(title="Flagged Books")
    table.add_column("Title", max_width=45)
    table.add_column("Ch", justify="right")
    table.add_column("Issues")

    for book in results:
        table.add_row(
            book["title"][:45],
            str(book["chapter_count"]),
            "; ".join(book["reasons"][:2]) + ("..." if len(book["reasons"]) > 2 else ""),
        )

    console.print(table)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_extraction_validator.py::TestAuditQualityCLI -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add agentic_pipeline/cli.py tests/test_extraction_validator.py
git commit -m "feat: add audit-quality CLI command for retroactive library validation"
```

---

### Task 4: Run retroactive audit against production library

**Step 1: Run the audit**

Run: `agentic-pipeline audit-quality`

Review the output — document which books are flagged and why.

**Step 2: Run with JSON for a permanent record**

Run: `agentic-pipeline audit-quality --json > docs/test-results/2026-02-14-quality-audit.json`

**Step 3: Commit the audit results**

```bash
git add docs/test-results/2026-02-14-quality-audit.json
git commit -m "docs: initial quality audit of existing library"
```

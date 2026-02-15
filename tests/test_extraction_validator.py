"""Tests for extraction quality validation."""

import hashlib
import sqlite3
import tempfile
from pathlib import Path
from statistics import median

import pytest

from agentic_pipeline.validation.extraction_validator import (
    ExtractionValidator,
    ValidationResult,
    check_extraction_quality,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash(text: str) -> str:
    """Return a unique content hash for test data."""
    return hashlib.sha256(text.encode()).hexdigest()


def _good_defaults(overrides: dict | None = None):
    """Return kwargs for check_extraction_quality that pass all checks."""
    defaults = dict(
        chapter_count=10,
        word_counts=[1000] * 10,
        titles=[f"Chapter {i}" for i in range(1, 11)],
        content_hashes=[_hash(f"unique-{i}") for i in range(10)],
    )
    if overrides:
        defaults.update(overrides)
    return defaults


# ===========================================================================
# Pure function tests
# ===========================================================================


class TestCheckExtractionQuality:
    """Tests for the pure check_extraction_quality function."""

    # --- Passing cases ---

    def test_all_checks_pass(self):
        result = check_extraction_quality(**_good_defaults())
        assert result.passed is True
        assert result.reasons == []
        assert result.warnings == []

    def test_metrics_populated(self):
        result = check_extraction_quality(**_good_defaults())
        assert "chapter_count" in result.metrics
        assert "total_words" in result.metrics
        assert "max_word_count" in result.metrics
        assert "median_word_count" in result.metrics
        assert "duplicate_ratio" in result.metrics

    # --- Check 1: Min 7 chapters ---

    def test_reject_fewer_than_7_chapters(self):
        result = check_extraction_quality(
            chapter_count=6,
            word_counts=[1000] * 6,
            titles=[f"Ch {i}" for i in range(6)],
            content_hashes=[_hash(f"u-{i}") for i in range(6)],
        )
        assert result.passed is False
        assert any("chapter" in r.lower() and "7" in r for r in result.reasons)

    def test_exactly_7_chapters_passes(self):
        result = check_extraction_quality(
            chapter_count=7,
            word_counts=[1000] * 7,
            titles=[f"Ch {i}" for i in range(7)],
            content_hashes=[_hash(f"u-{i}") for i in range(7)],
        )
        assert result.passed is True

    # --- Check 2: Max 20,000 words per chapter ---

    def test_reject_chapter_over_20000_words(self):
        wc = [1000] * 9 + [20001]
        result = check_extraction_quality(
            chapter_count=10,
            word_counts=wc,
            titles=[f"Ch {i}" for i in range(10)],
            content_hashes=[_hash(f"u-{i}") for i in range(10)],
        )
        assert result.passed is False
        assert any("20,000" in r or "20000" in r for r in result.reasons)

    def test_exactly_20000_words_passes(self):
        # Use 5000 base so 20000/5000 = 4.0x (exactly at ratio limit)
        wc = [5000] * 9 + [20000]
        result = check_extraction_quality(
            chapter_count=10,
            word_counts=wc,
            titles=[f"Ch {i}" for i in range(10)],
            content_hashes=[_hash(f"u-{i}") for i in range(10)],
        )
        assert result.passed is True

    # --- Check 3: Max chapter can't be >4x median ---

    def test_reject_chapter_over_4x_median(self):
        # median of [1000]*9 + [4001] = 1000; 4001/1000 = 4.001 > 4
        wc = [1000] * 9 + [4001]
        result = check_extraction_quality(
            chapter_count=10,
            word_counts=wc,
            titles=[f"Ch {i}" for i in range(10)],
            content_hashes=[_hash(f"u-{i}") for i in range(10)],
        )
        assert result.passed is False
        assert any("4x" in r or "4.0" in r or "ratio" in r.lower() for r in result.reasons)

    def test_exactly_4x_median_passes(self):
        wc = [1000] * 9 + [4000]
        result = check_extraction_quality(
            chapter_count=10,
            word_counts=wc,
            titles=[f"Ch {i}" for i in range(10)],
            content_hashes=[_hash(f"u-{i}") for i in range(10)],
        )
        assert result.passed is True

    # --- Check 4: Min 100 words per chapter (warning only) ---

    def test_warn_chapter_under_100_words(self):
        wc = [99] + [1000] * 9
        result = check_extraction_quality(
            chapter_count=10,
            word_counts=wc,
            titles=[f"Ch {i}" for i in range(10)],
            content_hashes=[_hash(f"u-{i}") for i in range(10)],
        )
        # Should warn but still pass (if other checks ok)
        assert result.passed is True
        assert len(result.warnings) > 0
        assert any("100" in w for w in result.warnings)

    # --- Check 5: Min 5,000 total words ---

    def test_reject_under_5000_total_words(self):
        wc = [499] * 10  # 4990 total
        result = check_extraction_quality(
            chapter_count=10,
            word_counts=wc,
            titles=[f"Ch {i}" for i in range(10)],
            content_hashes=[_hash(f"u-{i}") for i in range(10)],
        )
        assert result.passed is False
        assert any("5,000" in r or "5000" in r for r in result.reasons)

    def test_exactly_5000_total_words_passes(self):
        wc = [500] * 10  # 5000 total
        result = check_extraction_quality(
            chapter_count=10,
            word_counts=wc,
            titles=[f"Ch {i}" for i in range(10)],
            content_hashes=[_hash(f"u-{i}") for i in range(10)],
        )
        assert result.passed is True

    # --- Check 6: >10% duplicate chapters ---

    def test_reject_over_10_percent_duplicates(self):
        # 10 chapters, 3 copies of same hash = 2 duplicates = 20% > 10%
        hashes = [_hash(f"u-{i}") for i in range(7)] + [_hash("dup")] * 3
        result = check_extraction_quality(
            chapter_count=10,
            word_counts=[1000] * 10,
            titles=[f"Ch {i}" for i in range(10)],
            content_hashes=hashes,
        )
        assert result.passed is False
        assert any("duplicate" in r.lower() for r in result.reasons)

    def test_exactly_10_percent_duplicates_passes(self):
        # 10 chapters, need exactly 10% duplicates = 1 duplicate
        # Hash appearing 2x means 1 duplicate. 1/10 = 10% exactly -> passes
        hashes = [_hash(f"u-{i}") for i in range(9)] + [_hash("u-0")]
        result = check_extraction_quality(
            chapter_count=10,
            word_counts=[1000] * 10,
            titles=[f"Ch {i}" for i in range(10)],
            content_hashes=hashes,
        )
        assert result.passed is True

    def test_no_duplicates_passes(self):
        result = check_extraction_quality(**_good_defaults())
        assert result.passed is True

    # --- Check 7: Suspicious chapter titles ---

    @pytest.mark.parametrize("title", [
        "chapter1.zip",
        "book.exe",
        "archive.rar",
        "setup.msi",
        "installer.dmg",
        "package.pkg",
        "data.tar",
        "backup.gz",
        "files.7z",
    ])
    def test_reject_file_extension_titles(self, title):
        titles = [f"Ch {i}" for i in range(9)] + [title]
        result = check_extraction_quality(
            chapter_count=10,
            word_counts=[1000] * 10,
            titles=titles,
            content_hashes=[_hash(f"u-{i}") for i in range(10)],
        )
        assert result.passed is False
        assert any("suspicious" in r.lower() or "title" in r.lower() for r in result.reasons)

    def test_reject_table_reference_title(self):
        titles = [f"Ch {i}" for i in range(9)] + ["Table 42"]
        result = check_extraction_quality(
            chapter_count=10,
            word_counts=[1000] * 10,
            titles=titles,
            content_hashes=[_hash(f"u-{i}") for i in range(10)],
        )
        assert result.passed is False

    def test_reject_bare_number_title(self):
        titles = [f"Ch {i}" for i in range(9)] + ["42"]
        result = check_extraction_quality(
            chapter_count=10,
            word_counts=[1000] * 10,
            titles=titles,
            content_hashes=[_hash(f"u-{i}") for i in range(10)],
        )
        assert result.passed is False

    def test_reject_windows_path_title(self):
        titles = [f"Ch {i}" for i in range(9)] + ["/C:\\Windows\\System32"]
        result = check_extraction_quality(
            chapter_count=10,
            word_counts=[1000] * 10,
            titles=titles,
            content_hashes=[_hash(f"u-{i}") for i in range(10)],
        )
        assert result.passed is False

    def test_reject_unix_path_title(self):
        titles = [f"Ch {i}" for i in range(9)] + ["/usr/local/bin"]
        result = check_extraction_quality(
            chapter_count=10,
            word_counts=[1000] * 10,
            titles=titles,
            content_hashes=[_hash(f"u-{i}") for i in range(10)],
        )
        assert result.passed is False

    @pytest.mark.parametrize("path_prefix", ["/tmp/", "/var/log/", "/etc/config/", "/home/user/"])
    def test_reject_various_unix_paths(self, path_prefix):
        titles = [f"Ch {i}" for i in range(9)] + [f"{path_prefix}something"]
        result = check_extraction_quality(
            chapter_count=10,
            word_counts=[1000] * 10,
            titles=titles,
            content_hashes=[_hash(f"u-{i}") for i in range(10)],
        )
        assert result.passed is False

    def test_table_reference_case_insensitive(self):
        titles = [f"Ch {i}" for i in range(9)] + ["TABLE 5"]
        result = check_extraction_quality(
            chapter_count=10,
            word_counts=[1000] * 10,
            titles=titles,
            content_hashes=[_hash(f"u-{i}") for i in range(10)],
        )
        assert result.passed is False

    # --- All reasons collected (no short-circuit) ---

    def test_multiple_failures_all_reported(self):
        # Few chapters, low total words, duplicates
        hashes = [_hash("same")] * 5 + [_hash(f"u-{i}") for i in range(1)]
        result = check_extraction_quality(
            chapter_count=6,
            word_counts=[100] * 6,  # 600 total, under 5000
            titles=[f"Ch {i}" for i in range(5)] + ["file.exe"],
            content_hashes=hashes,
        )
        assert result.passed is False
        # Should have multiple rejection reasons
        assert len(result.reasons) >= 3  # min chapters, min total words, duplicates, suspicious title

    # --- Edge cases ---

    def test_zero_chapters(self):
        result = check_extraction_quality(
            chapter_count=0,
            word_counts=[],
            titles=[],
            content_hashes=[],
        )
        assert result.passed is False

    def test_median_zero_word_count_no_crash(self):
        """If all chapters have 0 words, ratio check should not divide by zero."""
        result = check_extraction_quality(
            chapter_count=10,
            word_counts=[0] * 10,
            titles=[f"Ch {i}" for i in range(10)],
            content_hashes=[_hash(f"u-{i}") for i in range(10)],
        )
        assert result.passed is False  # Will fail on total words < 5000


# ===========================================================================
# ExtractionValidator (DB integration) tests
# ===========================================================================


@pytest.fixture
def temp_library_db():
    """Create a temp DB with library tables (books + chapters)."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE books (
            id TEXT PRIMARY KEY,
            title TEXT,
            author TEXT,
            word_count INTEGER,
            source_file TEXT,
            processing_status TEXT,
            added_date TEXT,
            book_type TEXT,
            classification_confidence REAL,
            suggested_tags TEXT,
            classification_reasoning TEXT,
            classified_at TEXT,
            classified_by TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE chapters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            book_id TEXT,
            chapter_number INTEGER,
            title TEXT,
            file_path TEXT,
            word_count INTEGER,
            embedding BLOB,
            embedding_model TEXT,
            content_hash TEXT,
            file_mtime TEXT,
            embedding_updated_at TEXT
        )
    """)
    conn.commit()
    conn.close()
    yield db_path
    db_path.unlink(missing_ok=True)


def _insert_book(db_path: Path, book_id: str, title: str = "Test Book"):
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO books (id, title) VALUES (?, ?)",
        (book_id, title),
    )
    conn.commit()
    conn.close()


def _insert_chapters(
    db_path: Path,
    book_id: str,
    count: int,
    word_count: int = 1000,
    title_prefix: str = "Chapter",
    duplicate_hashes: int = 0,
):
    """Insert chapters. duplicate_hashes = number of chapters sharing the same hash."""
    conn = sqlite3.connect(str(db_path))
    for i in range(count):
        if duplicate_hashes > 0 and i >= count - duplicate_hashes:
            ch = _hash("duplicate-content")
        else:
            ch = _hash(f"{book_id}-{i}")
        conn.execute(
            """INSERT INTO chapters (book_id, chapter_number, title, word_count, content_hash)
               VALUES (?, ?, ?, ?, ?)""",
            (book_id, i + 1, f"{title_prefix} {i + 1}", word_count, ch),
        )
    conn.commit()
    conn.close()


class TestExtractionValidator:
    """Integration tests for ExtractionValidator querying a real DB."""

    def test_validate_good_book(self, temp_library_db):
        book_id = "good-book-1"
        _insert_book(temp_library_db, book_id)
        _insert_chapters(temp_library_db, book_id, count=10, word_count=1000)

        validator = ExtractionValidator()
        result = validator.validate(book_id, str(temp_library_db))

        assert result.passed is True
        assert result.reasons == []

    def test_validate_book_not_found(self, temp_library_db):
        validator = ExtractionValidator()
        result = validator.validate("nonexistent", str(temp_library_db))

        assert result.passed is False
        assert any("no chapters" in r.lower() or "not found" in r.lower() for r in result.reasons)

    def test_validate_too_few_chapters(self, temp_library_db):
        book_id = "small-book"
        _insert_book(temp_library_db, book_id)
        _insert_chapters(temp_library_db, book_id, count=3, word_count=1000)

        validator = ExtractionValidator()
        result = validator.validate(book_id, str(temp_library_db))

        assert result.passed is False

    def test_validate_with_duplicates(self, temp_library_db):
        book_id = "dup-book"
        _insert_book(temp_library_db, book_id)
        # 10 chapters, 5 share same hash -> 4 duplicates = 40%
        _insert_chapters(temp_library_db, book_id, count=10, word_count=1000, duplicate_hashes=5)

        validator = ExtractionValidator()
        result = validator.validate(book_id, str(temp_library_db))

        assert result.passed is False
        assert any("duplicate" in r.lower() for r in result.reasons)

    def test_validate_metrics_populated(self, temp_library_db):
        book_id = "metrics-book"
        _insert_book(temp_library_db, book_id)
        _insert_chapters(temp_library_db, book_id, count=10, word_count=1000)

        validator = ExtractionValidator()
        result = validator.validate(book_id, str(temp_library_db))

        assert result.metrics["chapter_count"] == 10
        assert result.metrics["total_words"] == 10000

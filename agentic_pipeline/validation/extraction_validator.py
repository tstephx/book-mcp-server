"""Extraction quality validation for ingested books.

Provides a pure-function validator (check_extraction_quality) that can be
called with raw data, and a DB-aware wrapper (ExtractionValidator) that
queries the chapters table and delegates to the pure function.
"""

from __future__ import annotations

import logging
import re
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from statistics import median
from typing import Optional

from agentic_pipeline.db.connection import get_pipeline_db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Suspicious title patterns
# ---------------------------------------------------------------------------

_FILE_EXT_RE = re.compile(r"\.(zip|exe|rar|msi|dmg|pkg|tar|gz|7z)\b", re.IGNORECASE)
_TABLE_REF_RE = re.compile(r"^Table\s+\d+", re.IGNORECASE)
_BARE_NUMBER_RE = re.compile(r"^\d+$")
_WINDOWS_PATH_RE = re.compile(r"[/\\][A-Za-z]:[/\\]")
_UNIX_PATH_RE = re.compile(r"^[/\\](?:usr|tmp|var|etc|home)[/\\]")

_SUSPICIOUS_PATTERNS = [
    _FILE_EXT_RE,
    _TABLE_REF_RE,
    _BARE_NUMBER_RE,
    _WINDOWS_PATH_RE,
    _UNIX_PATH_RE,
]

# ---------------------------------------------------------------------------
# Thresholds (very strict)
# ---------------------------------------------------------------------------

MIN_CHAPTERS = 3
MAX_CHAPTER_WORDS = 20_000
MAX_TO_MEDIAN_RATIO = 4.0
MIN_CHAPTER_WORDS = 100
MIN_TOTAL_WORDS = 5_000
MAX_DUPLICATE_RATIO = 0.10

# Fraction of the source file's words the extraction must retain. Anchor-based
# detection can find a handful of chapters, report high confidence, and silently
# drop the rest of the book; only comparing against the source catches that.
MIN_SOURCE_COVERAGE = 0.5

_HTML_SUFFIXES = (".xhtml", ".html", ".htm")
_SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style).*?</\1>")


class _TextExtractor(HTMLParser):
    """Collect character data, ignoring tags."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        self.chunks.append(data)


def count_source_words(source_path: str) -> Optional[int]:
    """Count words in a source EPUB (HTML documents) or PDF (page text).

    Returns None when the count can't be established (missing file,
    unsupported format, corrupt archive, or a PDF with fewer than 100
    extractable words — image-only scans make coverage meaningless).
    Callers treat None as "skip the check" rather than as a failure.
    """
    path = Path(source_path)
    if not path.is_file():
        return None

    suffix = path.suffix.lower()
    if suffix == ".epub":
        return _count_epub_words(path)
    if suffix == ".pdf":
        return _count_pdf_words(path)
    return None


def _count_epub_words(path: Path) -> Optional[int]:
    try:
        with zipfile.ZipFile(path) as archive:
            total = 0
            for name in archive.namelist():
                if not name.lower().endswith(_HTML_SUFFIXES):
                    continue
                try:
                    html = archive.read(name).decode("utf-8", "ignore")
                except (KeyError, OSError):
                    continue
                parser = _TextExtractor()
                parser.feed(_SCRIPT_STYLE_RE.sub(" ", html))
                total += len(" ".join(parser.chunks).split())
    except (zipfile.BadZipFile, OSError) as e:
        logger.warning(f"Could not read source words from {path.name}: {e}")
        return None

    return total


def _count_pdf_words(path: Path) -> Optional[int]:
    try:
        import fitz  # PyMuPDF — lazy: validators run in contexts without it loaded
    except ImportError:
        logger.warning("PyMuPDF not available; skipping PDF source word count")
        return None

    try:
        with fitz.open(str(path)) as doc:
            total = sum(len(page.get_text().split()) for page in doc)
    except Exception as e:
        logger.warning(f"Could not read source words from {path.name}: {e}")
        return None

    if total < 100:
        # image-only/scanned PDF: a near-zero denominator makes the
        # coverage ratio meaningless — treat as "can't establish a count"
        return None
    return total


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Outcome of an extraction quality check."""

    passed: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pure validation function
# ---------------------------------------------------------------------------


def check_extraction_quality(
    chapter_count: int,
    word_counts: list[int],
    titles: list[str],
    content_hashes: list[str],
    raw_source_words: Optional[int] = None,
) -> ValidationResult:
    """Run all extraction quality checks and return a ValidationResult.

    All checks are evaluated (no short-circuiting) so that every problem is
    surfaced in a single pass.

    Args:
        raw_source_words: Word count of the source file, for the completeness
            check. None or 0 skips it — the other checks only measure internal
            consistency and cannot tell a whole book from a tenth of one.
    """
    reasons: list[str] = []
    warnings: list[str] = []
    metrics: dict = {}

    # Precompute common values
    total_words = sum(word_counts)
    max_wc = max(word_counts) if word_counts else 0
    med_wc = median(word_counts) if word_counts else 0

    metrics["chapter_count"] = chapter_count
    metrics["total_words"] = total_words
    metrics["max_word_count"] = max_wc
    metrics["median_word_count"] = med_wc

    # --- Check 1: Min chapters ---
    if chapter_count < MIN_CHAPTERS:
        reasons.append(f"Too few chapters: {chapter_count} (minimum {MIN_CHAPTERS} required)")

    # --- Check 2: Max words per chapter ---
    if max_wc > MAX_CHAPTER_WORDS:
        reasons.append(f"Chapter exceeds max word count: {max_wc:,} words (max {MAX_CHAPTER_WORDS:,})")

    # --- Check 3: Max chapter can't be >Nx median ---
    if med_wc > 0 and max_wc / med_wc > MAX_TO_MEDIAN_RATIO:
        ratio = max_wc / med_wc
        reasons.append(
            f"Largest chapter is {ratio:.1f}x the median ({max_wc:,} vs {med_wc:,}) — max allowed is {MAX_TO_MEDIAN_RATIO}x"
        )

    # --- Check 4: Min words per chapter (warning only) ---
    short_chapters = [i for i, wc in enumerate(word_counts) if wc < MIN_CHAPTER_WORDS]
    if short_chapters:
        warnings.append(f"{len(short_chapters)} chapter(s) under {MIN_CHAPTER_WORDS} words")

    # --- Check 5: Min total words ---
    if total_words < MIN_TOTAL_WORDS:
        reasons.append(f"Total word count too low: {total_words:,} (minimum {MIN_TOTAL_WORDS:,} required)")

    # --- Check 6: Duplicate chapters by content_hash ---
    # Exclude NULL/empty hashes — chapters without a hash can't be compared
    known_hashes = [h for h in content_hashes if h]
    hash_counts = Counter(known_hashes)
    duplicate_count = sum(c - 1 for c in hash_counts.values() if c > 1)
    comparable_count = len(known_hashes) if known_hashes else chapter_count
    dup_ratio = duplicate_count / comparable_count if comparable_count > 0 else 0.0
    metrics["duplicate_ratio"] = dup_ratio
    if dup_ratio > MAX_DUPLICATE_RATIO:
        reasons.append(
            f"Too many duplicate chapters: {duplicate_count}/{chapter_count} "
            f"({dup_ratio:.0%}) share content hashes (max {MAX_DUPLICATE_RATIO:.0%})"
        )

    # --- Check 7: Suspicious chapter titles ---
    suspicious: list[str] = []
    for title in titles:
        for pattern in _SUSPICIOUS_PATTERNS:
            if pattern.search(title):
                suspicious.append(title)
                break
    if suspicious:
        examples = suspicious[:5]
        reasons.append(f"{len(suspicious)} suspicious chapter title(s): {examples}")

    # --- Check 8: Source completeness ---
    # Skipped when the source word count is unavailable (unreadable file,
    # unsupported format, or a scanned/image-only PDF under the word floor).
    if raw_source_words:
        coverage = total_words / raw_source_words
        metrics["source_coverage"] = coverage
        if coverage < MIN_SOURCE_COVERAGE:
            reasons.append(
                f"Extraction kept only {coverage:.0%} of the source "
                f"({total_words:,} of {raw_source_words:,} words, "
                f"minimum {MIN_SOURCE_COVERAGE:.0%})"
            )

    passed = len(reasons) == 0
    return ValidationResult(passed=passed, reasons=reasons, warnings=warnings, metrics=metrics)


# ---------------------------------------------------------------------------
# Library-wide scan
# ---------------------------------------------------------------------------


def find_flagged_books(db_path: str) -> tuple[int, list[dict]]:
    """Scan all library books and return those failing quality checks.

    Returns:
        (total_book_count, list_of_flagged_book_dicts) where each dict has
        keys: book_id, title, source_file, chapter_count, reasons, metrics.
    """
    with get_pipeline_db(db_path) as conn:
        books = conn.execute("SELECT id, title, source_file FROM books").fetchall()

        flagged = []
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
                flagged.append(
                    {
                        "book_id": book["id"],
                        "title": book["title"],
                        "source_file": book["source_file"],
                        "chapter_count": chapter_count,
                        "reasons": validation.reasons,
                        "metrics": validation.metrics,
                    }
                )

    return len(books), flagged


# ---------------------------------------------------------------------------
# DB-aware validator class
# ---------------------------------------------------------------------------


class ExtractionValidator:
    """Queries the chapters table for a book and validates extraction quality."""

    def validate(self, book_id: str, db_path: str, source_path: Optional[str] = None) -> ValidationResult:
        """Load chapter data from DB and run quality checks.

        Args:
            book_id: The book's ID in the books/chapters tables.
            db_path: Path to the SQLite database.
            source_path: Original book file, enabling the completeness check.
                Omitted or unreadable skips that check.

        Returns:
            ValidationResult with pass/fail, reasons, warnings, and metrics.
        """
        with get_pipeline_db(db_path) as conn:
            cursor = conn.execute(
                "SELECT title, word_count, content_hash FROM chapters WHERE book_id = ?",
                (book_id,),
            )
            rows = cursor.fetchall()

        if not rows:
            return ValidationResult(
                passed=False,
                reasons=[f"No chapters found for book_id='{book_id}'"],
                metrics={"chapter_count": 0},
            )

        chapter_count = len(rows)
        word_counts = [row["word_count"] or 0 for row in rows]
        titles = [row["title"] or "" for row in rows]
        content_hashes = [row["content_hash"] or "" for row in rows]

        return check_extraction_quality(
            chapter_count=chapter_count,
            word_counts=word_counts,
            titles=titles,
            content_hashes=content_hashes,
            raw_source_words=count_source_words(source_path) if source_path else None,
        )

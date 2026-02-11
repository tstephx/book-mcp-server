#!/usr/bin/env python3
"""
Audit chapter quality across the library.

Analyzes chapter data for all books and flags issues:
- Over-fragmentation (too many tiny chapters — front matter, appendices split out)
- Under-fragmentation (too few massive chapters — chapter detection failures)
- Title quality problems (generic names, PDF artifacts, non-chapter content)
- EPUB TOC mismatch (when source EPUB is available)

Usage:
    python scripts/audit_chapters.py                          # Summary table, all books
    python scripts/audit_chapters.py --severity bad           # Only problem books
    python scripts/audit_chapters.py --book-id <uuid>         # Deep dive on one book
    python scripts/audit_chapters.py --format json            # Machine-readable output
    python scripts/audit_chapters.py --format csv             # CSV output
"""

import argparse
import csv
import io
import json as json_module
import logging
import re
import sqlite3
import sys
from dataclasses import dataclass, field, asdict
from math import sqrt
from pathlib import Path

# Add project root to path so we can import agentic_pipeline
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic_pipeline.db.config import get_db_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Non-chapter title patterns — these suggest over-counting
# ---------------------------------------------------------------------------
NON_CHAPTER_PATTERNS = [
    r"^preface$",
    r"^foreword$",
    r"^introduction$",
    r"^appendix",
    r"^glossary$",
    r"^index$",
    r"^bibliography$",
    r"^references$",
    r"^acknowledg",
    r"^about the author",
    r"^copyright",
    r"^dedication$",
    r"^colophon$",
    r"^epilogue$",
    r"^prologue$",
    r"^table of contents$",
    r"^title page$",
    r"^half title",
    r"^also by",
    r"^praise for",
    r"^other books",
    r"^further reading$",
    r"^notes$",
]
NON_CHAPTER_RE = re.compile(
    "|".join(NON_CHAPTER_PATTERNS), re.IGNORECASE
)

# Generic / low-quality title patterns
GENERIC_TITLE_PATTERNS = [
    r"^section\s+\d+$",
    r"^part\s+\d+$",
    r"^chapter\s+\d+$",
    r"^\(title missing\)$",
    r"^untitled$",
    r"^\d+$",  # bare number
    r"^ch\.?\s*\d+$",
]
GENERIC_TITLE_RE = re.compile(
    "|".join(GENERIC_TITLE_PATTERNS), re.IGNORECASE
)

# PDF artifact patterns in titles
ARTIFACT_PATTERNS = [
    r"page\s+\d+",
    r"\bpp?\.\s*\d+",
    r"<[^>]+>",  # HTML/XML tags
    r"\x0c",     # form feed
    r"_{3,}",    # underscores (separator lines)
]
ARTIFACT_RE = re.compile("|".join(ARTIFACT_PATTERNS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ChapterInfo:
    chapter_number: int
    title: str
    word_count: int


@dataclass
class TitleIssue:
    chapter_number: int
    title: str
    issue: str  # "generic", "artifact", "non_chapter"


@dataclass
class BookAudit:
    book_id: str
    title: str
    author: str
    book_type: str
    chapter_count: int
    total_words: int
    avg_words: float
    min_words: int
    max_words: int
    cv: float  # coefficient of variation
    severity: str  # "good", "warning", "bad"
    issues: list[str] = field(default_factory=list)
    title_issues: list[TitleIssue] = field(default_factory=list)
    chapters: list[ChapterInfo] = field(default_factory=list)
    non_chapter_count: int = 0
    epub_toc_count: int | None = None
    source_file: str | None = None


# ---------------------------------------------------------------------------
# EPUB TOC parsing
# ---------------------------------------------------------------------------
def get_epub_toc_count(epub_path: str) -> int | None:
    """Parse an EPUB's TOC and return the number of entries, or None on failure."""
    try:
        from ebooklib import epub
    except ImportError:
        return None

    path = Path(epub_path)
    if not path.exists() or not path.suffix.lower() == ".epub":
        return None

    try:
        book = epub.read_epub(str(path), options={"ignore_ncx": False})
        toc = book.toc
        if not toc:
            return None
        # Flatten nested TOC (tuples are sections with children)
        count = 0
        stack = list(toc)
        while stack:
            item = stack.pop()
            if isinstance(item, tuple):
                # (Section, [children])
                stack.extend(item[1])
                count += 1
            else:
                count += 1
        return count
    except Exception as e:
        logger.debug(f"Could not parse EPUB TOC {epub_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Core audit logic
# ---------------------------------------------------------------------------
def audit_book(
    book_row: sqlite3.Row,
    chapters: list[sqlite3.Row],
) -> BookAudit:
    """Audit a single book and return its report."""
    book_id = book_row["id"]
    title = book_row["title"] or "(Unknown)"
    author = book_row["author"] or ""
    book_type = book_row["book_type"] or ""
    source_file = book_row["source_file"] or None
    total_words = book_row["word_count"] or 0

    chapter_infos = []
    word_counts = []
    title_issues: list[TitleIssue] = []
    non_chapter_count = 0

    for ch in chapters:
        ch_num = ch["chapter_number"]
        ch_title = ch["title"] or ""
        ch_words = ch["word_count"] or 0

        chapter_infos.append(ChapterInfo(ch_num, ch_title, ch_words))
        word_counts.append(ch_words)

        # Check title quality
        if GENERIC_TITLE_RE.search(ch_title.strip()):
            title_issues.append(TitleIssue(ch_num, ch_title, "generic"))
        if ARTIFACT_RE.search(ch_title):
            title_issues.append(TitleIssue(ch_num, ch_title, "artifact"))
        if NON_CHAPTER_RE.search(ch_title.strip()):
            title_issues.append(TitleIssue(ch_num, ch_title, "non_chapter"))
            non_chapter_count += 1

    chapter_count = len(word_counts)
    if chapter_count == 0:
        return BookAudit(
            book_id=book_id, title=title, author=author, book_type=book_type,
            chapter_count=0, total_words=total_words,
            avg_words=0, min_words=0, max_words=0, cv=0,
            severity="bad", issues=["no_chapters"],
            source_file=source_file,
        )

    avg_words = sum(word_counts) / chapter_count
    min_words = min(word_counts)
    max_words = max(word_counts)

    # Coefficient of variation
    if avg_words > 0:
        variance = sum((w - avg_words) ** 2 for w in word_counts) / chapter_count
        std_dev = sqrt(variance)
        cv = std_dev / avg_words
    else:
        cv = 0

    # --- Severity heuristics ---
    issues: list[str] = []
    severity = "good"

    # Over-fragmentation
    if avg_words < 1500:
        issues.append(f"over_fragmented (avg {avg_words:.0f} words)")
        severity = "bad"
    elif avg_words < 2000:
        issues.append(f"possibly_over_fragmented (avg {avg_words:.0f} words)")
        severity = max(severity, "warning", key=_severity_rank)

    # Under-fragmentation
    if avg_words > 15000:
        issues.append(f"under_fragmented (avg {avg_words:.0f} words)")
        severity = "bad"
    elif avg_words > 10000:
        issues.append(f"possibly_under_fragmented (avg {avg_words:.0f} words)")
        severity = max(severity, "warning", key=_severity_rank)

    # Giant chapters
    giant_chapters = [w for w in word_counts if w > 20000]
    if len(giant_chapters) >= 2:
        issues.append(f"{len(giant_chapters)} chapters > 20K words")
        severity = "bad"
    elif len(giant_chapters) == 1:
        issues.append(f"1 chapter > 20K words ({max_words:,})")
        severity = max(severity, "warning", key=_severity_rank)

    # High variance
    if cv > 1.5:
        issues.append(f"high_variance (CV={cv:.2f})")
        severity = max(severity, "warning", key=_severity_rank)

    # Title quality
    artifact_issues = [t for t in title_issues if t.issue == "artifact"]
    if artifact_issues:
        issues.append(f"{len(artifact_issues)} titles with artifacts")
        severity = "bad"

    generic_issues = [t for t in title_issues if t.issue == "generic"]
    if generic_issues:
        issues.append(f"{len(generic_issues)} generic titles")
        severity = max(severity, "warning", key=_severity_rank)

    if non_chapter_count > 0:
        issues.append(f"{non_chapter_count} non-chapter entries")
        # Only warning — some non-chapter entries are expected
        severity = max(severity, "warning", key=_severity_rank)

    # EPUB TOC comparison
    epub_toc_count = None
    if source_file:
        epub_toc_count = get_epub_toc_count(source_file)
        if epub_toc_count is not None:
            ratio = chapter_count / epub_toc_count if epub_toc_count > 0 else 0
            if ratio > 1.5:
                issues.append(
                    f"epub_over_count (DB:{chapter_count} vs TOC:{epub_toc_count})"
                )
                severity = max(severity, "warning", key=_severity_rank)
            elif ratio < 0.65:
                issues.append(
                    f"epub_under_count (DB:{chapter_count} vs TOC:{epub_toc_count})"
                )
                severity = max(severity, "warning", key=_severity_rank)

    return BookAudit(
        book_id=book_id,
        title=title,
        author=author,
        book_type=book_type,
        chapter_count=chapter_count,
        total_words=total_words,
        avg_words=avg_words,
        min_words=min_words,
        max_words=max_words,
        cv=cv,
        severity=severity,
        issues=issues,
        title_issues=title_issues,
        chapters=chapter_infos,
        non_chapter_count=non_chapter_count,
        epub_toc_count=epub_toc_count,
        source_file=source_file,
    )


_SEVERITY_ORDER = {"good": 0, "warning": 1, "bad": 2}


def _severity_rank(s: str) -> int:
    return _SEVERITY_ORDER.get(s, 0)


# ---------------------------------------------------------------------------
# Run audit across library
# ---------------------------------------------------------------------------
def run_audit(
    db_path: Path,
    book_id: str | None = None,
) -> list[BookAudit]:
    """Run the audit and return results for all (or one) book."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        if book_id:
            books = conn.execute(
                "SELECT id, title, author, word_count, source_file, book_type "
                "FROM books WHERE id = ?",
                (book_id,),
            ).fetchall()
            if not books:
                logger.error(f"Book not found: {book_id}")
                return []
        else:
            books = conn.execute(
                "SELECT id, title, author, word_count, source_file, book_type "
                "FROM books ORDER BY title"
            ).fetchall()

        results: list[BookAudit] = []
        for book_row in books:
            chapters = conn.execute(
                "SELECT chapter_number, title, word_count "
                "FROM chapters WHERE book_id = ? ORDER BY chapter_number",
                (book_row["id"],),
            ).fetchall()
            results.append(audit_book(book_row, chapters))

        return results
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------
def print_table(audits: list[BookAudit], severity_filter: str) -> None:
    """Print a Rich-style summary table."""
    try:
        from rich.console import Console
        from rich.table import Table

        _print_rich_table(audits, severity_filter)
    except ImportError:
        _print_plain_table(audits, severity_filter)


def _print_rich_table(audits: list[BookAudit], severity_filter: str) -> None:
    from rich.console import Console
    from rich.table import Table

    filtered = _filter_severity(audits, severity_filter)

    console = Console()

    # Aggregate stats
    total = len(audits)
    good = sum(1 for a in audits if a.severity == "good")
    warning = sum(1 for a in audits if a.severity == "warning")
    bad = sum(1 for a in audits if a.severity == "bad")

    console.print()
    console.print(f"[bold]Chapter Quality Audit[/bold]  "
                  f"({total} books: [green]{good} good[/green], "
                  f"[yellow]{warning} warning[/yellow], "
                  f"[red]{bad} bad[/red])")
    console.print()

    table = Table(show_header=True, header_style="bold")
    table.add_column("Title", max_width=40, no_wrap=False)
    table.add_column("Ch", justify="right")
    table.add_column("Avg Words", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("CV", justify="right")
    table.add_column("Sev", justify="center")
    table.add_column("Issues", no_wrap=False)

    severity_style = {"good": "green", "warning": "yellow", "bad": "red"}

    for a in filtered:
        style = severity_style.get(a.severity, "")
        issues_str = "; ".join(a.issues) if a.issues else ""
        table.add_row(
            a.title[:40],
            str(a.chapter_count),
            f"{a.avg_words:,.0f}",
            f"{a.min_words:,}",
            f"{a.max_words:,}",
            f"{a.cv:.2f}",
            f"[{style}]{a.severity}[/{style}]",
            issues_str,
        )

    console.print(table)

    # Issue type breakdown (across filtered books only)
    issue_types: dict[str, int] = {}
    for a in filtered:
        for issue in a.issues:
            # Strip leading count ("2 chapters..." -> "chapters...")
            key = re.sub(r"^\d+\s+", "", issue)
            # Strip parenthetical detail ("over_fragmented (avg 831 words)" -> "over_fragmented")
            key = re.sub(r"\s*\(.*\)$", "", key)
            # Normalize singular/plural
            key = re.sub(r"\bchapters\b", "chapter", key)
            issue_types[key] = issue_types.get(key, 0) + 1

    if issue_types:
        console.print()
        console.print("[bold]Issue Distribution:[/bold]")
        for itype, count in sorted(issue_types.items(), key=lambda x: -x[1]):
            console.print(f"  {itype}: {count}")
    console.print()


def _print_plain_table(audits: list[BookAudit], severity_filter: str) -> None:
    filtered = _filter_severity(audits, severity_filter)

    total = len(audits)
    good = sum(1 for a in audits if a.severity == "good")
    warning = sum(1 for a in audits if a.severity == "warning")
    bad = sum(1 for a in audits if a.severity == "bad")

    print(f"\nChapter Quality Audit  ({total} books: {good} good, "
          f"{warning} warning, {bad} bad)\n")

    header = f"{'Title':<42} {'Ch':>3} {'Avg':>7} {'Min':>6} {'Max':>7} {'CV':>5} {'Sev':<8} Issues"
    print(header)
    print("-" * len(header))

    for a in filtered:
        issues_str = "; ".join(a.issues) if a.issues else ""
        print(
            f"{a.title[:40]:<42} {a.chapter_count:>3} "
            f"{a.avg_words:>7,.0f} {a.min_words:>6,} {a.max_words:>7,} "
            f"{a.cv:>5.2f} {a.severity:<8} {issues_str}"
        )
    print()


def print_deep_dive(audit: BookAudit) -> None:
    """Print detailed report for a single book."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel

        console = Console()
        console.print()
        console.print(Panel(
            f"[bold]{audit.title}[/bold]\n"
            f"Author: {audit.author}\n"
            f"Type: {audit.book_type}\n"
            f"ID: {audit.book_id}",
            title="Book Details",
        ))

        # Stats
        console.print(f"\n[bold]Chapter Statistics:[/bold]")
        console.print(f"  Chapters: {audit.chapter_count}")
        console.print(f"  Total words: {audit.total_words:,}")
        console.print(f"  Avg words/chapter: {audit.avg_words:,.0f}")
        console.print(f"  Min: {audit.min_words:,}  Max: {audit.max_words:,}")
        console.print(f"  Coefficient of variation: {audit.cv:.2f}")
        if audit.epub_toc_count is not None:
            console.print(f"  EPUB TOC entries: {audit.epub_toc_count}")

        severity_style = {"good": "green", "warning": "yellow", "bad": "red"}
        style = severity_style.get(audit.severity, "")
        console.print(f"  Severity: [{style}]{audit.severity}[/{style}]")

        if audit.issues:
            console.print(f"\n[bold]Issues:[/bold]")
            for issue in audit.issues:
                console.print(f"  - {issue}")

        # Chapter table
        table = Table(title="Chapters", show_header=True)
        table.add_column("#", justify="right")
        table.add_column("Title", max_width=50, no_wrap=False)
        table.add_column("Words", justify="right")
        table.add_column("Flags")

        title_issue_map: dict[int, list[str]] = {}
        for ti in audit.title_issues:
            title_issue_map.setdefault(ti.chapter_number, []).append(ti.issue)

        for ch in audit.chapters:
            flags = title_issue_map.get(ch.chapter_number, [])
            word_style = ""
            if ch.word_count > 20000:
                word_style = "[red]"
            elif ch.word_count < 500:
                word_style = "[dim]"
            words_str = f"{word_style}{ch.word_count:,}"

            table.add_row(
                str(ch.chapter_number),
                ch.title,
                words_str,
                ", ".join(flags) if flags else "",
            )

        console.print()
        console.print(table)
        console.print()

    except ImportError:
        # Plain fallback
        print(f"\n{'='*60}")
        print(f"Book: {audit.title}")
        print(f"Author: {audit.author}")
        print(f"Type: {audit.book_type}")
        print(f"ID: {audit.book_id}")
        print(f"{'='*60}")
        print(f"Chapters: {audit.chapter_count}  |  Total words: {audit.total_words:,}")
        print(f"Avg: {audit.avg_words:,.0f}  |  Min: {audit.min_words:,}  |  Max: {audit.max_words:,}")
        print(f"CV: {audit.cv:.2f}  |  Severity: {audit.severity}")
        if audit.epub_toc_count is not None:
            print(f"EPUB TOC entries: {audit.epub_toc_count}")
        if audit.issues:
            print(f"\nIssues:")
            for issue in audit.issues:
                print(f"  - {issue}")
        print(f"\nChapters:")
        for ch in audit.chapters:
            flags = [ti.issue for ti in audit.title_issues if ti.chapter_number == ch.chapter_number]
            flag_str = f"  [{', '.join(flags)}]" if flags else ""
            print(f"  {ch.chapter_number:3d}. {ch.title} ({ch.word_count:,} words){flag_str}")
        print()


def print_json(audits: list[BookAudit], severity_filter: str) -> None:
    filtered = _filter_severity(audits, severity_filter)
    data = {
        "summary": {
            "total": len(audits),
            "good": sum(1 for a in audits if a.severity == "good"),
            "warning": sum(1 for a in audits if a.severity == "warning"),
            "bad": sum(1 for a in audits if a.severity == "bad"),
        },
        "books": [
            {
                "book_id": a.book_id,
                "title": a.title,
                "author": a.author,
                "book_type": a.book_type,
                "chapter_count": a.chapter_count,
                "total_words": a.total_words,
                "avg_words": round(a.avg_words, 1),
                "min_words": a.min_words,
                "max_words": a.max_words,
                "cv": round(a.cv, 2),
                "severity": a.severity,
                "issues": a.issues,
                "non_chapter_count": a.non_chapter_count,
                "epub_toc_count": a.epub_toc_count,
                "title_issues": [
                    {"chapter": ti.chapter_number, "title": ti.title, "issue": ti.issue}
                    for ti in a.title_issues
                ],
            }
            for a in filtered
        ],
    }
    print(json_module.dumps(data, indent=2))


def print_csv(audits: list[BookAudit], severity_filter: str) -> None:
    filtered = _filter_severity(audits, severity_filter)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "book_id", "title", "author", "book_type", "chapters", "total_words",
        "avg_words", "min_words", "max_words", "cv", "severity", "issues",
        "non_chapter_count", "epub_toc_count",
    ])
    for a in filtered:
        writer.writerow([
            a.book_id, a.title, a.author, a.book_type, a.chapter_count,
            a.total_words, round(a.avg_words, 1), a.min_words, a.max_words,
            round(a.cv, 2), a.severity, "; ".join(a.issues),
            a.non_chapter_count, a.epub_toc_count or "",
        ])
    print(output.getvalue(), end="")


def _filter_severity(audits: list[BookAudit], severity_filter: str) -> list[BookAudit]:
    if severity_filter == "all":
        return audits
    if severity_filter == "bad":
        return [a for a in audits if a.severity == "bad"]
    if severity_filter == "warning":
        return [a for a in audits if a.severity in ("warning", "bad")]
    return audits


# ---------------------------------------------------------------------------
# Public API (used by MCP tool)
# ---------------------------------------------------------------------------
def audit_library(
    db_path: Path | None = None,
    book_id: str | None = None,
    severity_filter: str = "all",
) -> dict:
    """Run audit and return structured results. Used by both CLI and MCP tool."""
    if db_path is None:
        db_path = get_db_path()

    audits = run_audit(db_path, book_id=book_id)
    filtered = _filter_severity(audits, severity_filter)

    summary = {
        "total": len(audits),
        "good": sum(1 for a in audits if a.severity == "good"),
        "warning": sum(1 for a in audits if a.severity == "warning"),
        "bad": sum(1 for a in audits if a.severity == "bad"),
    }

    books = []
    for a in filtered:
        entry = {
            "book_id": a.book_id,
            "title": a.title,
            "author": a.author,
            "book_type": a.book_type,
            "chapter_count": a.chapter_count,
            "total_words": a.total_words,
            "avg_words": round(a.avg_words, 1),
            "min_words": a.min_words,
            "max_words": a.max_words,
            "cv": round(a.cv, 2),
            "severity": a.severity,
            "issues": a.issues,
            "non_chapter_count": a.non_chapter_count,
            "epub_toc_count": a.epub_toc_count,
        }
        # Include chapter detail for single-book queries
        if book_id:
            entry["chapters"] = [
                {
                    "number": ch.chapter_number,
                    "title": ch.title,
                    "word_count": ch.word_count,
                    "flags": [
                        ti.issue
                        for ti in a.title_issues
                        if ti.chapter_number == ch.chapter_number
                    ],
                }
                for ch in a.chapters
            ]
            entry["title_issues"] = [
                {"chapter": ti.chapter_number, "title": ti.title, "issue": ti.issue}
                for ti in a.title_issues
            ]
        books.append(entry)

    return {"summary": summary, "books": books}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Audit chapter quality across the library")
    parser.add_argument("--format", choices=["table", "json", "csv"], default="table",
                        help="Output format (default: table)")
    parser.add_argument("--severity", choices=["all", "warning", "bad"], default="all",
                        help="Filter by minimum severity (default: all)")
    parser.add_argument("--book-id", type=str, default=None,
                        help="Audit a single book (deep dive mode)")
    args = parser.parse_args()

    db_path = get_db_path()
    logger.info(f"Database: {db_path}")

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)

    audits = run_audit(db_path, book_id=args.book_id)

    if not audits:
        logger.error("No books found.")
        sys.exit(1)

    # Single book deep dive
    if args.book_id and args.format == "table":
        print_deep_dive(audits[0])
        return

    # Multi-book output
    if args.format == "json":
        print_json(audits, args.severity)
    elif args.format == "csv":
        print_csv(audits, args.severity)
    else:
        print_table(audits, args.severity)


if __name__ == "__main__":
    main()

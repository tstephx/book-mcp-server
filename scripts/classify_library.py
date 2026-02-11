#!/usr/bin/env python3
"""
Classify all books in the library.

Self-contained script that:
1. Adds classification columns to the books table (idempotent)
2. Backfills from existing pipeline classification data (~33 books)
3. Classifies remaining books via LLM (OpenAI gpt-4o-mini or Anthropic)

Usage:
    python scripts/classify_library.py              # Classify all unclassified books
    python scripts/classify_library.py --dry-run     # Preview what would happen
    python scripts/classify_library.py --limit 10    # Classify first 10 unclassified
    python scripts/classify_library.py --provider anthropic  # Use Anthropic instead
"""

import argparse
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path so we can import agentic_pipeline
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic_pipeline.db.config import get_db_path
from agentic_pipeline.agents.classifier_types import BookProfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Base path for resolving relative chapter file_paths
BOOKS_BASE = Path(__file__).resolve().parent.parent.parent / "book-ingestion-python"

CLASSIFICATION_COLUMNS = [
    ("book_type", "TEXT"),
    ("classification_confidence", "REAL"),
    ("suggested_tags", "TEXT"),
    ("classification_reasoning", "TEXT"),
    ("classified_at", "TEXT"),
    ("classified_by", "TEXT"),
]


def ensure_columns(conn: sqlite3.Connection) -> int:
    """Add classification columns to books table if missing. Returns count added."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(books)")
    existing = {row[1] for row in cursor.fetchall()}

    added = 0
    for col_name, col_type in CLASSIFICATION_COLUMNS:
        if col_name not in existing:
            cursor.execute(f"ALTER TABLE books ADD COLUMN {col_name} {col_type}")
            logger.info(f"Added column: books.{col_name} ({col_type})")
            added += 1

    if added:
        conn.commit()
    else:
        logger.info("All classification columns already exist")

    return added


def backfill_from_pipeline(conn: sqlite3.Connection, dry_run: bool = False) -> int:
    """Copy classification data from processing_pipelines into books table.

    Joins on books.source_file = processing_pipelines.source_path.
    Only updates books that don't already have a book_type set.
    Returns number of books updated.
    """
    cursor = conn.cursor()

    cursor.execute("""
        SELECT b.id, b.title, p.book_profile
        FROM books b
        JOIN processing_pipelines p ON b.source_file = p.source_path
        WHERE p.book_profile IS NOT NULL
          AND (b.book_type IS NULL OR b.book_type = '')
    """)
    rows = cursor.fetchall()

    if not rows:
        logger.info("No pipeline data to backfill (already done or no matches)")
        return 0

    if dry_run:
        logger.info(f"[DRY RUN] Would backfill {len(rows)} books from pipeline data")
        for row in rows:
            logger.info(f"  - {row[1]}")
        return len(rows)

    updated = 0
    now = datetime.now(timezone.utc).isoformat()
    for book_id, title, profile_json in rows:
        try:
            profile = json.loads(profile_json) if isinstance(profile_json, str) else profile_json
            tags = json.dumps(profile.get("suggested_tags", []))
            cursor.execute("""
                UPDATE books SET
                    book_type = ?,
                    classification_confidence = ?,
                    suggested_tags = ?,
                    classification_reasoning = ?,
                    classified_at = ?,
                    classified_by = ?
                WHERE id = ?
            """, (
                profile.get("book_type", "unknown"),
                profile.get("confidence", 0.0),
                tags,
                profile.get("reasoning", ""),
                now,
                "pipeline_backfill",
                book_id,
            ))
            updated += 1
            logger.info(f"Backfilled: {title} -> {profile.get('book_type')}")
        except Exception as e:
            logger.warning(f"Failed to backfill {title}: {e}")

    conn.commit()
    return updated


def build_classification_text(conn: sqlite3.Connection, book_id: str, title: str, author: str) -> str:
    """Build a text sample for classification from book metadata and chapter content."""
    parts = []

    if title:
        parts.append(f"Title: {title}")
    if author:
        parts.append(f"Author: {author}")

    # Get all chapter titles
    cursor = conn.cursor()
    cursor.execute(
        "SELECT chapter_number, title, file_path FROM chapters WHERE book_id = ? ORDER BY chapter_number",
        (book_id,),
    )
    chapters = cursor.fetchall()

    if chapters:
        chapter_titles = [f"  {c[0]}. {c[1]}" for c in chapters]
        parts.append("Table of Contents:\n" + "\n".join(chapter_titles))

    # Read first chapter content (up to ~2K chars)
    if chapters:
        first_path_raw = chapters[0][2]
        if first_path_raw:
            path = Path(first_path_raw)
            if not path.is_absolute():
                path = BOOKS_BASE / first_path_raw
            try:
                content = path.read_text(encoding="utf-8", errors="replace")[:2000]
                parts.append(f"First chapter excerpt:\n{content}")
            except (OSError, IOError) as e:
                logger.debug(f"Could not read chapter file {path}: {e}")

    return "\n\n".join(parts)


def classify_books(
    conn: sqlite3.Connection,
    provider_name: str = "openai",
    limit: int | None = None,
    dry_run: bool = False,
) -> int:
    """Classify books that don't have classification data yet.

    Returns number of books classified.
    """
    cursor = conn.cursor()
    query = """
        SELECT id, title, author FROM books
        WHERE book_type IS NULL OR book_type = ''
        ORDER BY title
    """
    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query)
    books = cursor.fetchall()

    if not books:
        logger.info("All books already classified!")
        return 0

    logger.info(f"Found {len(books)} unclassified books")

    if dry_run:
        logger.info(f"[DRY RUN] Would classify {len(books)} books via {provider_name}")
        for book_id, title, author in books:
            logger.info(f"  - {title}" + (f" by {author}" if author else ""))
        return len(books)

    # Initialize provider
    if provider_name == "openai":
        from agentic_pipeline.agents.providers.openai_provider import OpenAIProvider
        provider = OpenAIProvider()
    elif provider_name == "anthropic":
        from agentic_pipeline.agents.providers.anthropic_provider import AnthropicProvider
        provider = AnthropicProvider()
    else:
        raise ValueError(f"Unknown provider: {provider_name}")

    classified = 0
    errors = 0
    now = datetime.now(timezone.utc).isoformat()

    for i, (book_id, title, author) in enumerate(books, 1):
        logger.info(f"[{i}/{len(books)}] Classifying: {title}")

        try:
            text = build_classification_text(conn, book_id, title, author)
            profile: BookProfile = provider.classify(text)
            tags = json.dumps(profile.suggested_tags)

            cursor.execute("""
                UPDATE books SET
                    book_type = ?,
                    classification_confidence = ?,
                    suggested_tags = ?,
                    classification_reasoning = ?,
                    classified_at = ?,
                    classified_by = ?
                WHERE id = ?
            """, (
                profile.book_type.value,
                profile.confidence,
                tags,
                profile.reasoning,
                now,
                f"{provider_name}:{provider.model if hasattr(provider, 'model') else 'unknown'}",
                book_id,
            ))
            conn.commit()
            classified += 1
            logger.info(f"  -> {profile.book_type.value} (confidence: {profile.confidence:.2f})")

            # Small delay to respect rate limits
            if i < len(books):
                time.sleep(0.2)

        except Exception as e:
            errors += 1
            logger.error(f"  Failed: {e}")

    logger.info(f"Classification complete: {classified} classified, {errors} errors")
    return classified


def print_summary(conn: sqlite3.Connection) -> None:
    """Print classification summary."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT book_type, COUNT(*) as cnt
        FROM books
        GROUP BY book_type
        ORDER BY cnt DESC
    """)
    rows = cursor.fetchall()

    print("\n" + "=" * 50)
    print("Classification Summary")
    print("=" * 50)
    for book_type, count in rows:
        label = book_type if book_type else "(unclassified)"
        print(f"  {label:30s} {count:4d}")

    cursor.execute("SELECT COUNT(*) FROM books")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM books WHERE book_type IS NOT NULL AND book_type != ''")
    classified = cursor.fetchone()[0]
    print(f"\n  Total: {total}  |  Classified: {classified}  |  Remaining: {total - classified}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Classify all library books")
    parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai",
                        help="LLM provider (default: openai)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max books to classify in this run")
    parser.add_argument("--skip-backfill", action="store_true",
                        help="Skip pipeline backfill step")
    args = parser.parse_args()

    db_path = get_db_path()
    logger.info(f"Database: {db_path}")

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        # Step 1: Ensure columns exist
        logger.info("Step 1: Ensuring classification columns exist...")
        ensure_columns(conn)

        # Step 2: Backfill from pipeline data
        if not args.skip_backfill:
            logger.info("Step 2: Backfilling from pipeline data...")
            backfilled = backfill_from_pipeline(conn, dry_run=args.dry_run)
            logger.info(f"  Backfilled: {backfilled} books")
        else:
            logger.info("Step 2: Skipped (--skip-backfill)")

        # Step 3: Classify remaining books via LLM
        logger.info(f"Step 3: Classifying via {args.provider}...")
        classified = classify_books(
            conn,
            provider_name=args.provider,
            limit=args.limit,
            dry_run=args.dry_run,
        )
        logger.info(f"  Classified: {classified} books")

        # Summary
        if not args.dry_run:
            print_summary(conn)

    finally:
        conn.close()


if __name__ == "__main__":
    main()

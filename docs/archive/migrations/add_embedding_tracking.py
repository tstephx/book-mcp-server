#!/usr/bin/env python3
"""
Migration: Add embedding tracking columns to chapters table

Adds columns for tracking content changes to enable incremental embedding updates:
- content_hash: SHA-256 hash of chapter content
- file_mtime: File modification time for fast change detection
- embedding_updated_at: Timestamp of last embedding generation

Usage:
    python migrations/add_embedding_tracking.py [--db-path PATH]
"""

import sys
import argparse
import hashlib
import logging
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import sqlite3

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_default_db_path() -> Path:
    """Get default database path"""
    return Path(__file__).parent.parent.parent / 'book-ingestion-python' / 'data' / 'library.db'


def column_exists(cursor: sqlite3.Cursor, table: str, column: str) -> bool:
    """Check if a column exists in a table"""
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    return column in columns


def add_tracking_columns(db_path: Path) -> dict:
    """Add tracking columns to chapters table

    Args:
        db_path: Path to SQLite database

    Returns:
        Migration result summary
    """
    logger.info(f"Running migration on: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    columns_added = []
    columns_skipped = []

    # Add content_hash column
    if not column_exists(cursor, 'chapters', 'content_hash'):
        cursor.execute("ALTER TABLE chapters ADD COLUMN content_hash TEXT")
        columns_added.append('content_hash')
        logger.info("Added column: content_hash")
    else:
        columns_skipped.append('content_hash')
        logger.info("Column already exists: content_hash")

    # Add file_mtime column
    if not column_exists(cursor, 'chapters', 'file_mtime'):
        cursor.execute("ALTER TABLE chapters ADD COLUMN file_mtime REAL")
        columns_added.append('file_mtime')
        logger.info("Added column: file_mtime")
    else:
        columns_skipped.append('file_mtime')
        logger.info("Column already exists: file_mtime")

    # Add embedding_updated_at column
    if not column_exists(cursor, 'chapters', 'embedding_updated_at'):
        cursor.execute("ALTER TABLE chapters ADD COLUMN embedding_updated_at TEXT")
        columns_added.append('embedding_updated_at')
        logger.info("Added column: embedding_updated_at")
    else:
        columns_skipped.append('embedding_updated_at')
        logger.info("Column already exists: embedding_updated_at")

    conn.commit()
    conn.close()

    return {
        'columns_added': columns_added,
        'columns_skipped': columns_skipped
    }


def backfill_existing_embeddings(db_path: Path, books_dir: Path) -> dict:
    """Backfill tracking data for chapters that already have embeddings

    For chapters with existing embeddings, compute and store:
    - Current file mtime
    - Current content hash
    - Set embedding_updated_at to now (approximate)

    Args:
        db_path: Path to SQLite database
        books_dir: Base directory for book files

    Returns:
        Backfill result summary
    """
    logger.info("Backfilling existing embeddings with tracking data...")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get chapters with embeddings but no tracking data
    cursor.execute("""
        SELECT id, file_path
        FROM chapters
        WHERE embedding IS NOT NULL
          AND (content_hash IS NULL OR file_mtime IS NULL)
    """)
    chapters = cursor.fetchall()

    if not chapters:
        logger.info("No chapters need backfilling")
        conn.close()
        return {'backfilled': 0, 'errors': 0}

    logger.info(f"Backfilling {len(chapters)} chapters...")

    backfilled = 0
    errors = 0
    now = datetime.now().isoformat()

    for chapter in chapters:
        try:
            # Resolve file path
            file_path = Path(chapter['file_path'])
            if not file_path.is_absolute():
                # Try stripping 'data/books/' prefix
                try:
                    relative = file_path.relative_to('data/books')
                    file_path = books_dir / relative
                except ValueError:
                    file_path = books_dir / file_path

            # Handle split chapters (directories)
            if file_path.suffix == '.md' and not file_path.exists():
                dir_path = file_path.with_suffix('')
                if dir_path.is_dir():
                    file_path = dir_path

            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                errors += 1
                continue

            # Get mtime
            if file_path.is_dir():
                # For split chapters, use latest mtime of parts
                mtimes = [p.stat().st_mtime for p in file_path.glob('*.md')]
                mtime = max(mtimes) if mtimes else 0
            else:
                mtime = file_path.stat().st_mtime

            # Read content and compute hash
            if file_path.is_dir():
                parts = sorted(file_path.glob('[0-9]*.md'))
                if not parts:
                    parts = sorted([p for p in file_path.glob('*.md') if not p.name.startswith('_')])
                content = '\n\n'.join(p.read_text(encoding='utf-8') for p in parts)
            else:
                content = file_path.read_text(encoding='utf-8')

            content_hash = hashlib.sha256(content.encode()).hexdigest()

            # Update database
            cursor.execute("""
                UPDATE chapters
                SET content_hash = ?,
                    file_mtime = ?,
                    embedding_updated_at = ?
                WHERE id = ?
            """, (content_hash, mtime, now, chapter['id']))

            backfilled += 1

            if backfilled % 50 == 0:
                logger.info(f"Progress: {backfilled}/{len(chapters)}")
                conn.commit()

        except Exception as e:
            logger.warning(f"Error backfilling chapter {chapter['id']}: {e}")
            errors += 1

    conn.commit()
    conn.close()

    logger.info(f"Backfill complete: {backfilled} updated, {errors} errors")

    return {'backfilled': backfilled, 'errors': errors}


def main():
    parser = argparse.ArgumentParser(
        description="Add embedding tracking columns to chapters table"
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default=None,
        help='Path to database (default: ../book-ingestion-python/data/library.db)'
    )
    parser.add_argument(
        '--books-dir',
        type=str,
        default=None,
        help='Path to books directory for backfill'
    )
    parser.add_argument(
        '--skip-backfill',
        action='store_true',
        help='Skip backfilling existing embeddings'
    )

    args = parser.parse_args()

    # Determine paths
    db_path = Path(args.db_path) if args.db_path else get_default_db_path()

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)

    # Run migration
    result = add_tracking_columns(db_path)

    if result['columns_added']:
        logger.info(f"Migration complete. Added columns: {', '.join(result['columns_added'])}")
    else:
        logger.info("No new columns needed")

    # Backfill existing embeddings
    if not args.skip_backfill:
        books_dir = Path(args.books_dir) if args.books_dir else db_path.parent / 'books'

        if books_dir.exists():
            backfill_result = backfill_existing_embeddings(db_path, books_dir)
            logger.info(f"Backfill: {backfill_result['backfilled']} chapters updated")
        else:
            logger.warning(f"Books directory not found: {books_dir}")
            logger.warning("Skipping backfill. Run with --books-dir to specify location.")

    logger.info("Done!")


if __name__ == "__main__":
    main()

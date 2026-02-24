#!/usr/bin/env python3
"""
Migration: Add FTS5 full-text search and chapter summaries tables

Creates:
- chapters_fts: FTS5 virtual table for full-text search
- chapter_summaries: Pre-generated chapter summaries

Usage:
    python migrations/add_fts_and_summaries.py [--db-path PATH] [--books-dir PATH]
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

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


def table_exists(cursor: sqlite3.Cursor, table_name: str) -> bool:
    """Check if a table exists"""
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    )
    return cursor.fetchone() is not None


def setup_fts(db_path: Path, books_dir: Path) -> dict:
    """Set up FTS5 full-text search table

    Args:
        db_path: Path to SQLite database
        books_dir: Path to books directory for reading content

    Returns:
        Migration result summary
    """
    logger.info("Setting up FTS5 full-text search...")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Check if FTS table already exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='chapters_fts'"
    )
    if cursor.fetchone():
        logger.info("FTS table already exists")
        conn.close()
        return {'fts_created': False, 'chapters_indexed': 0}

    # Create FTS5 virtual table
    # Note: We store content directly in FTS since chapters table doesn't have content
    cursor.execute("""
        CREATE VIRTUAL TABLE chapters_fts USING fts5(
            chapter_id,
            title,
            content,
            tokenize='porter unicode61'
        )
    """)
    logger.info("Created FTS5 virtual table")

    # Index all chapters
    cursor.execute("SELECT id, title, file_path FROM chapters")
    chapters = cursor.fetchall()

    indexed = 0
    errors = 0

    for chapter in chapters:
        try:
            # Read content from file
            file_path = Path(chapter['file_path'])
            if not file_path.is_absolute():
                try:
                    relative = file_path.relative_to('data/books')
                    file_path = books_dir / relative
                except ValueError:
                    file_path = books_dir / file_path

            # Handle split chapters
            if file_path.suffix == '.md' and not file_path.exists():
                dir_path = file_path.with_suffix('')
                if dir_path.is_dir():
                    file_path = dir_path

            if file_path.is_dir():
                parts = sorted(file_path.glob('[0-9]*.md'))
                if not parts:
                    parts = sorted([p for p in file_path.glob('*.md') if not p.name.startswith('_')])
                content = '\n\n'.join(p.read_text(encoding='utf-8') for p in parts)
            elif file_path.exists():
                content = file_path.read_text(encoding='utf-8')
            else:
                logger.warning(f"File not found: {file_path}")
                errors += 1
                continue

            # Insert into FTS
            cursor.execute(
                "INSERT INTO chapters_fts (chapter_id, title, content) VALUES (?, ?, ?)",
                (chapter['id'], chapter['title'] or '', content)
            )
            indexed += 1

            if indexed % 50 == 0:
                logger.info(f"Indexed {indexed} chapters...")
                conn.commit()

        except Exception as e:
            logger.warning(f"Error indexing chapter {chapter['id']}: {e}")
            errors += 1

    conn.commit()
    conn.close()

    logger.info(f"FTS setup complete: {indexed} indexed, {errors} errors")
    return {'fts_created': True, 'chapters_indexed': indexed, 'errors': errors}


def setup_summaries_table(db_path: Path) -> dict:
    """Set up chapter summaries table

    Args:
        db_path: Path to SQLite database

    Returns:
        Migration result summary
    """
    logger.info("Setting up chapter summaries table...")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if table exists
    if table_exists(cursor, 'chapter_summaries'):
        logger.info("Summaries table already exists")
        conn.close()
        return {'summaries_table_created': False}

    # Create summaries table
    cursor.execute("""
        CREATE TABLE chapter_summaries (
            chapter_id TEXT PRIMARY KEY,
            summary TEXT NOT NULL,
            summary_type TEXT DEFAULT 'extractive',
            word_count INTEGER,
            generated_at TEXT,
            FOREIGN KEY (chapter_id) REFERENCES chapters(id) ON DELETE CASCADE
        )
    """)

    # Create index for faster lookups
    cursor.execute("""
        CREATE INDEX idx_summaries_type ON chapter_summaries(summary_type)
    """)

    conn.commit()
    conn.close()

    logger.info("Summaries table created")
    return {'summaries_table_created': True}


def main():
    parser = argparse.ArgumentParser(
        description="Add FTS5 and chapter summaries tables"
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default=None,
        help='Path to database'
    )
    parser.add_argument(
        '--books-dir',
        type=str,
        default=None,
        help='Path to books directory'
    )
    parser.add_argument(
        '--skip-fts',
        action='store_true',
        help='Skip FTS setup'
    )
    parser.add_argument(
        '--skip-summaries',
        action='store_true',
        help='Skip summaries table setup'
    )

    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else get_default_db_path()
    books_dir = Path(args.books_dir) if args.books_dir else db_path.parent / 'books'

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)

    results = {}

    if not args.skip_fts:
        if books_dir.exists():
            results['fts'] = setup_fts(db_path, books_dir)
        else:
            logger.warning(f"Books directory not found: {books_dir}")
            logger.warning("Skipping FTS setup. Use --books-dir to specify location.")

    if not args.skip_summaries:
        results['summaries'] = setup_summaries_table(db_path)

    logger.info(f"Migration complete: {results}")


if __name__ == "__main__":
    main()

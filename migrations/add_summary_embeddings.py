#!/usr/bin/env python3
"""
Migration: Add embedding columns to chapter_summaries table

Adds:
- embedding BLOB: Serialized numpy embedding vector for the summary
- embedding_model TEXT: Name of the model used to generate the embedding

Usage:
    python migrations/add_summary_embeddings.py [--db-path PATH]
"""

import sys
import argparse
import logging
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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


def add_summary_embeddings(db_path: Path) -> dict:
    """Add embedding columns to chapter_summaries table

    Args:
        db_path: Path to SQLite database

    Returns:
        Migration result summary
    """
    logger.info("Adding embedding columns to chapter_summaries...")

    conn = sqlite3.connect(str(db_path), timeout=10)
    cursor = conn.cursor()

    try:
        # Check if chapter_summaries table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chapter_summaries'"
        )
        if not cursor.fetchone():
            logger.error("chapter_summaries table does not exist. Run add_fts_and_summaries.py first.")
            return {'error': 'chapter_summaries table not found', 'columns_added': 0}

        columns_added = 0

        if not column_exists(cursor, 'chapter_summaries', 'embedding'):
            cursor.execute("ALTER TABLE chapter_summaries ADD COLUMN embedding BLOB")
            logger.info("Added 'embedding' column")
            columns_added += 1
        else:
            logger.info("'embedding' column already exists")

        if not column_exists(cursor, 'chapter_summaries', 'embedding_model'):
            cursor.execute("ALTER TABLE chapter_summaries ADD COLUMN embedding_model TEXT")
            logger.info("Added 'embedding_model' column")
            columns_added += 1
        else:
            logger.info("'embedding_model' column already exists")

        conn.commit()

        logger.info(f"Migration complete: {columns_added} columns added")
        return {'columns_added': columns_added, 'status': 'success'}

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Add embedding columns to chapter_summaries table"
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default=None,
        help='Path to database'
    )

    args = parser.parse_args()
    db_path = Path(args.db_path) if args.db_path else get_default_db_path()

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)

    result = add_summary_embeddings(db_path)
    if 'error' in result:
        sys.exit(1)


if __name__ == "__main__":
    main()

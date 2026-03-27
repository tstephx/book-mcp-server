"""Migration: Fix stale chapter file_path entries in the database.

Most books were ingested via the old book-ingestion-python tool which stored
chapters at /Users/.../book-ingestion-python/data/processed/{id}/chapters/.
Those files were later moved to ~/Library/Application Support/book-library/processed/
but the DB was never updated.

This migration rewrites all stale file_path values to point to their actual
location on disk.

Usage:
    .venv/bin/python3 migrations/fix_chapter_paths.py [--dry-run]
"""

import argparse
import logging
import sqlite3
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path.home() / "Library" / "Application Support" / "book-library" / "library.db"
PROCESSED_DIR = Path.home() / "Library" / "Application Support" / "book-library" / "processed"

# Old path prefixes that need rewriting
STALE_PREFIXES = [
    "/Users/taylorstephens/_Projects/book-ingestion-python/data/processed/",
    "/Users/taylorstephens/Dev/_Projects/book-ingestion-python/data/processed/",
]


def find_stale_paths(cursor) -> list[tuple]:
    """Return (id, file_path) for chapters with stale file_path values."""
    placeholders = " OR ".join(["file_path LIKE ?"] * len(STALE_PREFIXES))
    params = [f"{prefix}%" for prefix in STALE_PREFIXES]
    cursor.execute(
        f"SELECT id, file_path FROM chapters WHERE {placeholders} ORDER BY id",
        params,
    )
    return cursor.fetchall()


def compute_new_path(old_path: str) -> str | None:
    """Rewrite an old file_path to point to the processed/ directory."""
    for prefix in STALE_PREFIXES:
        if old_path.startswith(prefix):
            # Extract {book_id}/chapters/filename.md from the old path
            relative = old_path[len(prefix) :]
            new_path = str(PROCESSED_DIR / relative)
            return new_path
    return None


def verify_path_exists(new_path: str) -> bool:
    """Check that the rewritten path actually exists on disk."""
    return Path(new_path).exists()


def run_migration(dry_run: bool = False):
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    cursor = conn.cursor()

    stale = find_stale_paths(cursor)
    logger.info(f"Found {len(stale)} chapters with stale file paths")

    if not stale:
        logger.info("Nothing to do")
        conn.close()
        return

    updated = 0
    missing = 0
    errors = []

    for chapter_id, old_path in stale:
        new_path = compute_new_path(old_path)
        if new_path is None:
            errors.append(f"  Could not rewrite: {old_path}")
            continue

        if not verify_path_exists(new_path):
            missing += 1
            if missing <= 5:
                errors.append(f"  File not found: {new_path}")
            continue

        if not dry_run:
            cursor.execute(
                "UPDATE chapters SET file_path = ? WHERE id = ?",
                (new_path, chapter_id),
            )
        updated += 1

    if not dry_run:
        conn.commit()

    logger.info(f"{'Would update' if dry_run else 'Updated'}: {updated} chapters")
    if missing:
        logger.warning(f"Skipped {missing} chapters (file not found on disk)")
    if errors:
        for e in errors:
            logger.warning(e)

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix stale chapter file paths")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    args = parser.parse_args()
    run_migration(dry_run=args.dry_run)

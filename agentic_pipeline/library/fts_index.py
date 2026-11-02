"""Keep the chapters_fts keyword index (behind `text_search`) in step with chapters."""

from __future__ import annotations

import logging
from pathlib import Path

from agentic_pipeline.db.connection import get_pipeline_db
from agentic_pipeline.library.chapter_reader import read_chapter_content

logger = logging.getLogger(__name__)


def sync_book_fts(db_path: Path | str, book_id: str) -> int:
    """Index one book's chapters into chapters_fts. Idempotent (rows are replaced).

    Returns the number of chapters indexed. Never raises: FTS is a search aid
    and must not fail an approval; problems are logged and skipped.
    """
    from src.utils.fts_search import fts_table_exists, sync_fts_chapter

    db_path = Path(db_path)
    if not fts_table_exists(db_path):
        logger.warning("chapters_fts missing in %s; skipping FTS sync for %s", db_path, book_id)
        return 0

    indexed = 0
    try:
        with get_pipeline_db(str(db_path)) as conn:
            chapters = conn.execute(
                "SELECT id, title, file_path FROM chapters WHERE book_id = ?", (book_id,)
            ).fetchall()
        books_dir = db_path.parent / "books"
        for ch in chapters:
            try:
                content = read_chapter_content(ch["file_path"], books_dir)
            except Exception as e:
                logger.warning("FTS sync: cannot read chapter %s: %s", ch["id"], e)
                continue
            if sync_fts_chapter(ch["id"], content, ch["title"] or "", db_path=db_path):
                indexed += 1
    except Exception as e:
        logger.warning("FTS sync failed for %s: %s", book_id, e)
    return indexed

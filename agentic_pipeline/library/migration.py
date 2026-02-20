"""Library migration helpers for chunking and re-embedding the book library."""

import hashlib
import io
import logging
import sqlite3
from pathlib import Path

import numpy as np

from agentic_pipeline.db.connection import get_pipeline_db
from agentic_pipeline.library.chapter_reader import read_chapter_content

logger = logging.getLogger(__name__)


def chunk_all_books(db_path: str, dry_run: bool = False) -> dict:
    """Chunk all chapters that don't yet have chunks.

    Args:
        db_path: Path to the library database
        dry_run: If True, count what would be chunked without modifying DB

    Returns:
        Dict with books, chapters, chunks_created counts
    """
    from src.utils.chunker import chunk_chapter

    books_dir = Path(db_path).parent / "books"

    with get_pipeline_db(db_path) as conn:
        cursor = conn.cursor()

        # Find chapters without chunks
        cursor.execute(
            """SELECT c.id, c.book_id, c.file_path, b.title
            FROM chapters c
            JOIN books b ON c.book_id = b.id
            WHERE NOT EXISTS (
                SELECT 1 FROM chunks k WHERE k.chapter_id = c.id
            )
            ORDER BY b.title, c.id"""
        )
        chapters = cursor.fetchall()

        if not chapters:
            return {"books": 0, "chapters": 0, "chunks_created": 0}

        book_ids = set()
        chapters_processed = 0
        total_chunks = 0

        for ch in chapters:
            try:
                content = read_chapter_content(ch["file_path"], books_dir)
                if not content.strip():
                    continue

                chunks = chunk_chapter(content)
                book_ids.add(ch["book_id"])
                chapters_processed += 1

                if not dry_run:
                    for chunk in chunks:
                        chunk_id = f"{ch['id']}:{chunk['chunk_index']}"
                        content_hash = hashlib.sha256(
                            chunk["content"].encode()
                        ).hexdigest()
                        cursor.execute(
                            """INSERT OR REPLACE INTO chunks
                            (id, chapter_id, book_id, chunk_index, content, word_count, content_hash)
                            VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            (chunk_id, ch["id"], ch["book_id"],
                             chunk["chunk_index"], chunk["content"],
                             chunk["word_count"], content_hash),
                        )
                        total_chunks += 1
                else:
                    total_chunks += len(chunks)

            except Exception as e:
                logger.warning(f"Cannot chunk chapter {ch['id']}: {e}")

        if not dry_run:
            conn.commit()

        return {
            "books": len(book_ids),
            "chapters": chapters_processed,
            "chunks_created": total_chunks,
        }


def embed_all_chunks(db_path: str, dry_run: bool = False,
                     batch_size: int = 30) -> dict:
    """Embed all chunks that don't yet have embeddings.

    Args:
        db_path: Path to the library database
        dry_run: If True, count what would be embedded without calling API
        batch_size: Number of chunks per API call

    Returns:
        Dict with chunks_embedded, total_chunks counts
    """
    from src.utils.openai_embeddings import OpenAIEmbeddingGenerator

    with get_pipeline_db(db_path) as conn:
        cursor = conn.cursor()

        # Count total chunks and chunks needing embeddings
        cursor.execute("SELECT COUNT(*) FROM chunks")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL")
        needs_embedding = cursor.fetchone()[0]

        if needs_embedding == 0 or dry_run:
            return {
                "chunks_embedded": 0,
                "total_chunks": total,
                "needs_embedding": needs_embedding,
            }

        generator = OpenAIEmbeddingGenerator()

        # Fetch chunks without embeddings
        cursor.execute(
            "SELECT id, content FROM chunks WHERE embedding IS NULL"
        )
        rows = cursor.fetchall()

        chunks_embedded = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            texts = [r["content"] for r in batch]
            ids = [r["id"] for r in batch]

            embeddings = generator.generate_batch(texts)

            for cid, emb in zip(ids, embeddings):
                buf = io.BytesIO()
                np.save(buf, emb)
                cursor.execute(
                    """UPDATE chunks SET embedding = ?, embedding_model = ?
                    WHERE id = ?""",
                    (buf.getvalue(), "text-embedding-3-large", cid),
                )
                chunks_embedded += 1

            conn.commit()
            logger.info(f"Embedded {chunks_embedded}/{needs_embedding} chunks")

        return {
            "chunks_embedded": chunks_embedded,
            "total_chunks": total,
            "needs_embedding": 0,
        }

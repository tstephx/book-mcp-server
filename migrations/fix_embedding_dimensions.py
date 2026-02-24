"""Migration: Re-embed chunks with wrong model dimension.

Finds all chunks with embedding_model != 'text-embedding-3-large' and
re-embeds them using the correct model (3072 dims).

Usage:
    .venv/bin/python3 migrations/fix_embedding_dimensions.py [--dry-run]
"""

import argparse
import io
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_MODEL = "text-embedding-3-large"
TARGET_DIM = 3072
DB_PATH = Path(__file__).parent.parent.parent / "book-ingestion-python" / "data" / "library.db"
BATCH_SIZE = 30


def find_mismatched_chunks(cursor) -> list[tuple]:
    """Return (id, content) for all chunks not embedded with TARGET_MODEL."""
    cursor.execute(
        """
        SELECT id, content FROM chunks
        WHERE embedding_model != ? OR embedding_model IS NULL
        ORDER BY id
        """,
        (TARGET_MODEL,),
    )
    return cursor.fetchall()


def run(db_path: Path, dry_run: bool = False) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    rows = find_mismatched_chunks(cursor)
    total = len(rows)

    if total == 0:
        logger.info("No mismatched chunks found — database is consistent.")
        conn.close()
        return

    logger.info(f"Found {total} chunks with wrong/missing embedding model.")

    if dry_run:
        cursor.execute(
            "SELECT embedding_model, COUNT(*) FROM chunks WHERE embedding_model != ? OR embedding_model IS NULL GROUP BY embedding_model",
            (TARGET_MODEL,),
        )
        for row in cursor.fetchall():
            logger.info(f"  model={row[0]!r}: {row[1]} chunks")
        logger.info("Dry run — no changes made.")
        conn.close()
        return

    # Import here so dry-run works without OpenAI creds
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.utils.openai_embeddings import OpenAIEmbeddingGenerator

    generator = OpenAIEmbeddingGenerator()
    embedded = 0

    for i in range(0, total, BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        texts = [r["content"] for r in batch]
        ids = [r["id"] for r in batch]

        embeddings = generator.generate_batch(texts)

        for cid, emb in zip(ids, embeddings):
            assert emb.shape == (TARGET_DIM,), f"Unexpected shape {emb.shape} for chunk {cid}"
            buf = io.BytesIO()
            np.save(buf, emb)
            cursor.execute(
                "UPDATE chunks SET embedding = ?, embedding_model = ? WHERE id = ?",
                (buf.getvalue(), TARGET_MODEL, cid),
            )
            embedded += 1

        conn.commit()
        logger.info(f"Re-embedded {embedded}/{total} chunks")

    logger.info(f"Done. Re-embedded {embedded} chunks → all now use {TARGET_MODEL} ({TARGET_DIM} dims).")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report without making changes")
    parser.add_argument("--db", default=str(DB_PATH), help="Path to library.db")
    args = parser.parse_args()

    run(Path(args.db), dry_run=args.dry_run)

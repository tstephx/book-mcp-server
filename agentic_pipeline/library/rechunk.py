"""Staged re-chunking of the chunks table (spec: 2026-07-15-rechunking-design).

`stage_all` re-chunks every chapter into chunks_staging, reconciling by
content_hash so embeddings survive reruns (decision 5: no stale-embedding
state is representable). Production `chunks` is only written by `swap()`
(Task 10), inside one transaction, behind the eval gate.
"""

import io
import json
import logging
import uuid
from pathlib import Path

import numpy as np

from src.utils.chunker import chunk_chapter
from src.utils.embedding_sync import compute_content_hash
from src.utils.file_utils import read_chapter_content

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-large"
# $0.13 / 1M tokens for text-embedding-3-large; ~1.35 tokens/word for book prose
_USD_PER_MTOKEN = 0.13
_TOKENS_PER_WORD = 1.35

_STAGING_DDL = """
    CREATE TABLE IF NOT EXISTS chunks_staging (
        id TEXT PRIMARY KEY,
        chapter_id TEXT NOT NULL,
        book_id TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        content TEXT NOT NULL,
        word_count INTEGER NOT NULL,
        embedding BLOB,
        embedding_model TEXT,
        content_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(chapter_id, chunk_index)
    )
"""


def ensure_staging(conn) -> None:
    conn.execute(_STAGING_DDL)
    conn.commit()


def drop_staging(conn) -> None:
    conn.execute("DROP TABLE IF EXISTS chunks_staging")
    conn.commit()


def stage_chapter(conn, chapter_row, chunk_fn) -> tuple[int, int]:
    """Re-chunk one chapter into staging, reusing embeddings by content_hash.

    Returns (staged_count, reused_embedding_count).
    """
    text = read_chapter_content(chapter_row["file_path"])
    desired = chunk_fn(text)

    # harvest existing embeddings for this chapter before replacing its rows
    existing = {}
    for row in conn.execute(
        "SELECT content_hash, embedding, embedding_model FROM chunks_staging "
        "WHERE chapter_id = ? AND embedding IS NOT NULL",
        (chapter_row["id"],),
    ):
        existing.setdefault(row["content_hash"], []).append((row["embedding"], row["embedding_model"]))

    conn.execute("DELETE FROM chunks_staging WHERE chapter_id = ?", (chapter_row["id"],))

    reused = 0
    for chunk in desired:
        chash = compute_content_hash(chunk["content"])
        embedding, model = None, None
        if existing.get(chash):
            embedding, model = existing[chash].pop()
            reused += 1
        conn.execute(
            "INSERT INTO chunks_staging "
            "(id, chapter_id, book_id, chunk_index, content, word_count, "
            " embedding, embedding_model, content_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                chapter_row["id"],
                chapter_row["book_id"],
                chunk["chunk_index"],
                chunk["content"],
                chunk["word_count"],
                embedding,
                model,
                chash,
            ),
        )
    return len(desired), reused


def _carry_live_chunks(conn, chapter_id: str) -> int:
    """Source file gone: carry the chapter's live chunks into staging as-is."""
    conn.execute("DELETE FROM chunks_staging WHERE chapter_id = ?", (chapter_id,))
    cur = conn.execute(
        "INSERT INTO chunks_staging "
        "(id, chapter_id, book_id, chunk_index, content, word_count, "
        " embedding, embedding_model, content_hash) "
        "SELECT id, chapter_id, book_id, chunk_index, content, word_count, "
        "       embedding, embedding_model, COALESCE(content_hash, '') "
        "FROM chunks WHERE chapter_id = ?",
        (chapter_id,),
    )
    return cur.rowcount


def stage_all(conn, chunk_fn=None) -> dict:
    """Re-chunk every chapter into staging. Never writes `chunks`."""
    if chunk_fn is None:
        chunk_fn = chunk_chapter
    ensure_staging(conn)

    chapters = conn.execute(
        "SELECT id, book_id, file_path FROM chapters WHERE file_path IS NOT NULL ORDER BY id"
    ).fetchall()

    staged = reused = 0
    carried: list[str] = []
    skipped: list[str] = []
    for ch in chapters:
        try:
            s, r = stage_chapter(conn, ch, chunk_fn)
            staged += s
            reused += r
        except (FileNotFoundError, IOError):
            n = _carry_live_chunks(conn, ch["id"])
            if n:
                staged += n
                reused += n  # carried rows keep their embeddings
                carried.append(ch["id"])
            else:
                skipped.append(ch["id"])
                logger.warning(f"Chapter {ch['id']}: source gone and no live chunks — skipped")
    conn.commit()

    pending = conn.execute("SELECT COUNT(*) FROM chunks_staging WHERE embedding IS NULL").fetchone()[0]

    report = {
        "chapters": len(chapters),
        "staged_chunks": staged,
        "reused_embeddings": reused,
        "pending_embeddings": pending,
        "carried_chapters": carried,
        "skipped": skipped,
    }
    logger.info(f"stage_all: {report}")
    return report


def estimate_embedding_cost(conn) -> dict:
    row = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(word_count), 0) FROM chunks_staging WHERE embedding IS NULL"
    ).fetchone()
    pending, words = row[0], row[1]
    est_tokens = int(words * _TOKENS_PER_WORD)
    return {
        "pending": pending,
        "words": words,
        "est_tokens": est_tokens,
        "est_usd": est_tokens / 1_000_000 * _USD_PER_MTOKEN,
    }


def embed_pending(conn, generator=None, batch_size: int = 256) -> int:
    """Embed staged rows lacking embeddings. Resumable by construction."""
    if generator is None:
        from src.utils.openai_embeddings import OpenAIEmbeddingGenerator

        generator = OpenAIEmbeddingGenerator()

    total = 0
    while True:
        rows = conn.execute(
            "SELECT id, content FROM chunks_staging WHERE embedding IS NULL ORDER BY id LIMIT ?",
            (batch_size,),
        ).fetchall()
        if not rows:
            break
        vectors = generator.generate_batch([r["content"] for r in rows])
        if len(vectors) != len(rows):
            # zip would silently drop the tail, leaving rows NULL forever and
            # re-selecting them next iteration — an infinite loop
            raise RuntimeError(f"generate_batch returned {len(vectors)} vectors for {len(rows)} rows")
        for row, vec in zip(rows, vectors):
            buf = io.BytesIO()
            np.save(buf, np.asarray(vec, dtype=np.float32))
            conn.execute(
                "UPDATE chunks_staging SET embedding = ?, embedding_model = ? WHERE id = ?",
                (buf.getvalue(), EMBEDDING_MODEL, row["id"]),
            )
        conn.commit()  # commit per batch: interruption loses at most one batch
        total += len(rows)
        logger.info(f"embed_pending: {total} embedded so far")
    return total


def snapshot_marker(conn) -> dict:
    """Live-chunks marker for the swap-time staleness check (decision 2)."""
    row = conn.execute("SELECT COUNT(*), COALESCE(MAX(created_at), '') FROM chunks").fetchone()
    return {"chunk_count": row[0], "max_created_at": row[1]}


# Gold files live in the repo (spec: only runtime state lives beside the DB)
_REPO_ROOT = Path(__file__).resolve().parents[2]
GOLD_PATHS = [
    _REPO_ROOT / "eval" / "gold-queries.json",
    _REPO_ROOT / "eval" / "gold-queries-manual.json",
]


def _verdict_path(db_path) -> Path:
    return Path(db_path).parent / "rechunk" / "last-verdict.json"


def gate_pass(baseline: dict, staged: dict) -> bool:
    """Spec acceptance gate, semantic mode only (decision 1), verbatim:

    auto hit@5 >= baseline AND auto MRR >= baseline AND manual hit@5 > baseline.
    """
    b_auto = baseline["auto"]["semantic"]
    s_auto = staged["auto"]["semantic"]
    b_manual = baseline["manual"]["semantic"]
    s_manual = staged["manual"]["semantic"]
    return (
        s_auto["hit_at_5"] >= b_auto["hit_at_5"]
        and s_auto["mrr"] >= b_auto["mrr"]
        and s_manual["hit_at_5"] > b_manual["hit_at_5"]
    )


def run_gate_eval(db_path, gold_paths=None, embedder=None) -> dict:
    """Score live vs staged, decide the gate, persist the verdict."""
    import sqlite3 as _sqlite3

    from src.utils.retrieval_eval import run_eval

    if gold_paths is None:
        gold_paths = GOLD_PATHS
    if embedder is None:
        from src.utils.openai_embeddings import OpenAIEmbeddingGenerator

        embedder = OpenAIEmbeddingGenerator()

    baseline = run_eval(db_path, gold_paths, table="chunks", embedder=embedder)
    staged = run_eval(db_path, gold_paths, table="chunks_staging", embedder=embedder)

    if baseline["auto"]["semantic"]["n"] == 0:
        raise RuntimeError("gold set has zero 'auto' entries — gate cannot evaluate the auto arm")
    if baseline["manual"]["semantic"]["n"] == 0:
        raise RuntimeError("gold set has zero 'manual' entries — gate cannot evaluate the manual arm")

    conn = _sqlite3.connect(str(db_path))
    try:
        snapshot = snapshot_marker(conn)
    finally:
        conn.close()

    verdict = {
        "baseline": baseline,
        "staged": staged,
        "pass": gate_pass(baseline, staged),
        "snapshot": snapshot,
        "gold_counts": {
            "auto": baseline["auto"]["semantic"]["n"],
            "manual": baseline["manual"]["semantic"]["n"],
        },
    }
    path = _verdict_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(verdict, indent=2))
    logger.info(f"Verdict ({'PASS' if verdict['pass'] else 'FAIL'}) -> {path}")
    return verdict


def load_verdict(db_path):
    path = _verdict_path(db_path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


class SwapRefused(RuntimeError):
    """Swap preconditions not met (no PASS verdict, unembedded rows, ...)."""


def _pre_commit_hook() -> None:
    """Test seam: raised-from here simulates mid-transaction failure."""


def _stage_delta(conn, marker: dict, generator=None) -> int:
    """Stage chapters whose live chunks postdate the marker (decision 2)."""
    rows = conn.execute(
        "SELECT DISTINCT c.id, c.book_id, c.file_path FROM chapters c "
        "JOIN chunks k ON k.chapter_id = c.id "
        "WHERE k.created_at > ? "
        "   OR c.id NOT IN (SELECT DISTINCT chapter_id FROM chunks_staging)",
        (marker.get("max_created_at", ""),),
    ).fetchall()
    for ch in rows:
        try:
            stage_chapter(conn, ch, chunk_chapter)
        except (FileNotFoundError, IOError):
            _carry_live_chunks(conn, ch["id"])
    conn.commit()
    if rows:
        embed_pending(conn, generator=generator)
    return len(rows)


def swap(db_path, generator=None) -> dict:
    """Apply staged chunks to production. Refuses without a PASS verdict."""
    import sqlite3 as _sqlite3

    from agentic_pipeline.health.doctor import create_backup

    verdict = load_verdict(db_path)
    if verdict is None:
        raise SwapRefused("no verdict found — run `agentic-pipeline rechunk` first")
    if not verdict.get("pass"):
        raise SwapRefused("last eval verdict is FAIL — refusing to swap")

    conn = _sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = _sqlite3.Row
    # autocommit mode: python's implicit-transaction management would fight
    # the explicit BEGIN IMMEDIATE below ("cannot start a transaction within
    # a transaction" when a DML statement auto-begins first)
    conn.isolation_level = None
    try:
        staging_exists = conn.execute("SELECT name FROM sqlite_master WHERE name = 'chunks_staging'").fetchone()
        if not staging_exists:
            raise SwapRefused("chunks_staging does not exist — run `agentic-pipeline rechunk` first")

        delta = _stage_delta(conn, verdict.get("snapshot", {}), generator=generator)

        unembedded = conn.execute("SELECT COUNT(*) FROM chunks_staging WHERE embedding IS NULL").fetchone()[0]
        if unembedded:
            raise SwapRefused(f"{unembedded} staged chunks lack embeddings")

        staged_n = conn.execute("SELECT COUNT(*) FROM chunks_staging").fetchone()[0]
        if staged_n == 0:
            raise SwapRefused("staging is empty")

        backup = create_backup(db_path)

        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM chunks")
            conn.execute(
                "INSERT INTO chunks (id, chapter_id, book_id, chunk_index, content, "
                " word_count, embedding, embedding_model, content_hash) "
                "SELECT id, chapter_id, book_id, chunk_index, content, "
                " word_count, embedding, embedding_model, content_hash "
                "FROM chunks_staging"
            )
            conn.execute("DROP TABLE chunks_staging")
            conn.execute("UPDATE library_meta SET data_version = data_version + 1")
            _pre_commit_hook()
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        live_n = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        nulls = conn.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL").fetchone()[0]
        if live_n != staged_n or nulls:
            raise RuntimeError(
                f"post-swap verification failed (count {live_n} vs {staged_n}, "
                f"{nulls} NULL embeddings) — restore backup: {backup}"
            )
        version = conn.execute("SELECT data_version FROM library_meta WHERE id = 1").fetchone()[0]
    finally:
        conn.close()

    logger.info(f"swap complete: {live_n} chunks live, data_version={version}")
    return {
        "chunks": live_n,
        "backup": str(backup),
        "data_version": version,
        "delta_chapters": delta,
    }

"""Retrieval eval: gold-query scoring of chunk-embedding schemes.

Scores hit@5 and MRR at CHAPTER level so results are comparable across
chunking schemes. Semantic mode is the acceptance-gate number; hybrid
mode (production RRF path) is reported for user-facing truth but never
gates (spec decision 1).
"""

import io
import json
import logging
import sqlite3
from pathlib import Path

import numpy as np

from .fts_search import full_text_search
from .hybrid_search import reciprocal_rank_fusion
from .vector_store import find_top_k

logger = logging.getLogger(__name__)

_ALLOWED_TABLES = {"chunks", "chunks_staging"}
_FETCH_K = 50  # chunk-level over-fetch before chapter aggregation


def load_gold(paths: list) -> list[dict]:
    """Load and merge gold-query files; missing files are skipped."""
    golds: list[dict] = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            logger.warning(f"Gold file missing, skipping: {p}")
            continue
        golds.extend(json.loads(p.read_text()))
    return golds


def load_matrix(db_path, table: str = "chunks"):
    """Load (matrix, metadata) for one chunk table from an explicit DB path."""
    if table not in _ALLOWED_TABLES:
        raise ValueError(f"table must be one of {_ALLOWED_TABLES}, got {table!r}")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"SELECT id, chapter_id, book_id, embedding FROM {table} WHERE embedding IS NOT NULL ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return None, []
    embeddings = [np.load(io.BytesIO(r["embedding"])) for r in rows]
    meta = [{"chunk_id": r["id"], "chapter_id": r["chapter_id"], "book_id": r["book_id"]} for r in rows]
    return np.vstack(embeddings), meta


def rank_chapters_semantic(query_vec, matrix, meta, k: int = 10) -> list[dict]:
    """Top-k DISTINCT chapters by best-chunk cosine similarity."""
    top = find_top_k(query_vec, matrix, k=min(_FETCH_K, matrix.shape[0]), min_similarity=0.0)
    seen: set = set()
    chapters: list[dict] = []
    for idx, sim in top:
        cid = meta[idx]["chapter_id"]
        if cid in seen:
            continue
        seen.add(cid)
        chapters.append(
            {
                "chapter_id": cid,
                "book_id": meta[idx]["book_id"],
                "similarity": float(sim),
            }
        )
        if len(chapters) >= k:
            break
    return chapters


def rank_chapters_hybrid(query: str, query_vec, matrix, meta, k: int = 10) -> list[dict]:
    """Production-shaped ranking: chapter FTS + chunk semantic fused by RRF.

    Uses full_text_search() (reads BOOK_DB_PATH env) for the FTS arm — the
    live chapters_fts index is chapter-level and identical for both chunk
    schemes, so it cancels out in before/after comparison.
    """
    semantic = rank_chapters_semantic(query_vec, matrix, meta, k=_FETCH_K)
    fts = full_text_search(query, limit=10).get("results", [])
    fused = reciprocal_rank_fusion(fts, semantic)
    return fused[:k]


def _is_hit(gold: dict, item: dict) -> bool:
    if gold.get("gold_chapter_id"):
        return item.get("chapter_id") == gold["gold_chapter_id"]
    return item.get("book_id") == gold["gold_book_id"]


def evaluate(golds: list[dict], ranked_lists: list[list[dict]], k: int = 5) -> dict:
    """hit@k and MRR over parallel (gold, ranked) lists.

    Manual gold entries may carry gold_chapter_id=None — any chapter of
    the gold book counts (book-level hit).
    """
    if len(golds) != len(ranked_lists):
        raise ValueError(f"golds and ranked_lists must be the same length, got {len(golds)} and {len(ranked_lists)}")
    hits = 0
    rr = 0.0
    for gold, ranked in zip(golds, ranked_lists):
        rank = next((i + 1 for i, item in enumerate(ranked) if _is_hit(gold, item)), None)
        if rank is not None and rank <= k:
            hits += 1
        if rank is not None:
            rr += 1.0 / rank
    n = len(golds)
    return {
        "hit_at_5": hits / n if n else 0.0,
        "mrr": rr / n if n else 0.0,
        "n": n,
    }


def run_eval(db_path, gold_paths: list, table: str, embedder) -> dict:
    """Score one chunk table against the gold set, semantic + hybrid.

    embedder: object with .generate(text) -> np.ndarray (injectable for tests).
    Returns {"auto": {"semantic": {...}, "hybrid": {...}}, "manual": {...}}.
    """
    golds = load_gold(gold_paths)
    matrix, meta = load_matrix(db_path, table=table)
    if matrix is None:
        raise RuntimeError(f"No embedded rows in {table}")

    report: dict = {}
    for source in ("auto", "manual"):
        subset = [g for g in golds if g.get("source") == source]
        sem_lists, hyb_lists = [], []
        for g in subset:
            qv = embedder.generate(g["query"])
            sem_lists.append(rank_chapters_semantic(qv, matrix, meta, k=10))
            hyb_lists.append(rank_chapters_hybrid(g["query"], qv, matrix, meta, k=10))
        report[source] = {
            "semantic": evaluate(subset, sem_lists, k=5),
            "hybrid": evaluate(subset, hyb_lists, k=5),
        }
    return report

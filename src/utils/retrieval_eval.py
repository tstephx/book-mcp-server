"""Retrieval eval: gold-query scoring of chunk-embedding schemes.

Scores hit@5 and MRR at CHAPTER level so results are comparable across
chunking schemes. Semantic mode is the acceptance-gate number; hybrid
mode (production RRF path) is reported for user-facing truth but never
gates (spec decision 1).
"""

import io
import json
import logging
import random
import re
import sqlite3
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from .fts_search import full_text_search
from .hybrid_search import reciprocal_rank_fusion
from .vector_store import find_top_k

logger = logging.getLogger(__name__)

_ALLOWED_TABLES = {"chunks", "chunks_staging"}
_FETCH_K = 50  # chunk-level over-fetch before chapter aggregation

_GOLD_PROMPT = (
    "Below is a passage from a book. Write ONE specific question that this "
    "passage uniquely answers — concrete enough that only this passage (not "
    "general knowledge) answers it. Respond with ONLY a JSON object: "
    '{"query": "<the question>"}\n\nPASSAGE:\n'
)


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


def _parse_fenced_json(text: str) -> Optional[dict]:
    """Parse a JSON object from raw or ```json-fenced output."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else None
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _claude_generate(passage: str, timeout: int = 120) -> Optional[str]:
    """One gold question via `claude -p` (subscription billing, decision 7)."""
    try:
        proc = subprocess.run(
            ["claude", "-p", _GOLD_PROMPT + passage],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning(f"claude -p failed: {e}")
        return None
    if proc.returncode != 0:
        logger.warning(f"claude -p exit {proc.returncode}: {proc.stderr[:200]}")
        return None
    parsed = _parse_fenced_json(proc.stdout)
    if not parsed or not parsed.get("query"):
        logger.warning(f"Unparseable gold output: {proc.stdout[:200]}")
        return None
    return str(parsed["query"]).strip()


def build_gold(
    db_path,
    out_path,
    n: int = 60,
    seed: int = 42,
    min_words: int = 300,
    passage_words: int = 400,
    generator=None,
) -> int:
    """Sample chapters (NOT chunks — decision 6), excerpt a seeded passage,
    generate one question each, write gold records. Returns records written.
    """
    from .file_utils import read_chapter_content

    if generator is None:
        generator = _claude_generate

    rng = random.Random(seed)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        chapters = conn.execute(
            "SELECT c.id, c.book_id, c.file_path FROM chapters c "
            "JOIN books b ON b.id = c.book_id "
            "WHERE c.word_count >= ? AND c.file_path IS NOT NULL "
            "ORDER BY c.id",
            (min_words,),
        ).fetchall()
    finally:
        conn.close()

    # Stratify: round-robin across shuffled books so no book dominates
    by_book: dict = defaultdict(list)
    for ch in chapters:
        by_book[ch["book_id"]].append(ch)
    books = sorted(by_book)
    rng.shuffle(books)
    for b in books:
        rng.shuffle(by_book[b])

    picks = []
    round_idx = 0
    while len(picks) < n:
        advanced = False
        for b in books:
            if round_idx < len(by_book[b]):
                picks.append(by_book[b][round_idx])
                advanced = True
                if len(picks) >= n:
                    break
        if not advanced:
            break  # corpus smaller than n
        round_idx += 1

    records = []
    for ch in picks:
        try:
            text = read_chapter_content(ch["file_path"])
        except (FileNotFoundError, OSError) as e:
            logger.warning(f"Skipping {ch['id']}: {e}")
            continue
        words = text.split()
        if len(words) < min_words:
            continue
        start = rng.randrange(0, max(1, len(words) - passage_words))
        passage = " ".join(words[start : start + passage_words])
        query = generator(passage)
        if not query:
            logger.warning(f"Generator failed for chapter {ch['id']} — skipped")
            continue
        records.append(
            {
                "query": query,
                "gold_chapter_id": ch["id"],
                "gold_book_id": ch["book_id"],
                "source": "auto",
            }
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(records, indent=2))
    logger.info(f"Wrote {len(records)} gold records to {out_path}")
    return len(records)

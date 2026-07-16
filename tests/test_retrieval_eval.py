"""Retrieval eval core: matrix loading, ranking, hit@5/MRR scoring."""

import io
import json
import sqlite3

import numpy as np
import pytest

from src.utils.retrieval_eval import (
    evaluate,
    load_gold,
    load_matrix,
    rank_chapters_semantic,
)


def _blob(vec):
    buf = io.BytesIO()
    np.save(buf, np.asarray(vec, dtype=np.float32))
    return buf.getvalue()


@pytest.fixture
def eval_db(tmp_path):
    db = tmp_path / "library.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE chunks (
            id TEXT PRIMARY KEY, chapter_id TEXT NOT NULL, book_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL, content TEXT NOT NULL,
            word_count INTEGER NOT NULL, embedding BLOB, embedding_model TEXT,
            content_hash TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
    )
    rows = [
        ("k1", "ch1", "b1", 0, "alpha", 1, _blob([1.0, 0.0, 0.0])),
        ("k2", "ch1", "b1", 1, "beta", 1, _blob([0.9, 0.1, 0.0])),  # same chapter as k1
        ("k3", "ch2", "b1", 0, "gamma", 1, _blob([0.0, 1.0, 0.0])),
        ("k4", "ch3", "b2", 0, "delta", 1, _blob([0.0, 0.0, 1.0])),
    ]
    conn.executemany(
        "INSERT INTO chunks (id, chapter_id, book_id, chunk_index, content, word_count, embedding) VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()
    return db


class TestLoadMatrix:
    def test_loads_only_embedded_rows(self, eval_db):
        matrix, meta = load_matrix(eval_db, table="chunks")
        assert matrix.shape == (4, 3)
        assert [m["chunk_id"] for m in meta] == ["k1", "k2", "k3", "k4"]

    def test_rejects_unknown_table(self, eval_db):
        with pytest.raises(ValueError):
            load_matrix(eval_db, table="chunks; DROP TABLE chunks")


class TestSemanticRanking:
    def test_distinct_chapters_and_ordering(self, eval_db):
        matrix, meta = load_matrix(eval_db, table="chunks")
        q = np.array([1.0, 0.05, 0.0], dtype=np.float32)
        ranked = rank_chapters_semantic(q, matrix, meta, k=10)
        ids = [r["chapter_id"] for r in ranked]
        assert ids[0] == "ch1"
        assert len(ids) == len(set(ids)), "chapters must be distinct"
        # k1 and k2 are both ch1 — aggregation collapses them
        assert ids.count("ch1") == 1


class TestEvaluate:
    def test_hit_and_mrr_chapter_gold(self):
        golds = [{"query": "q", "gold_chapter_id": "ch2", "gold_book_id": "b1"}]
        ranked = [
            [
                {"chapter_id": "ch9", "book_id": "b9"},
                {"chapter_id": "ch2", "book_id": "b1"},
            ]
        ]
        r = evaluate(golds, ranked, k=5)
        assert r["hit_at_5"] == 1.0
        assert r["mrr"] == pytest.approx(0.5)

    def test_book_level_fallback_when_no_gold_chapter(self):
        golds = [{"query": "q", "gold_chapter_id": None, "gold_book_id": "b2"}]
        ranked = [
            [
                {"chapter_id": "chX", "book_id": "b1"},
                {"chapter_id": "chY", "book_id": "b2"},
            ]
        ]
        r = evaluate(golds, ranked, k=5)
        assert r["hit_at_5"] == 1.0
        assert r["mrr"] == pytest.approx(0.5)

    def test_miss_beyond_k(self):
        golds = [{"query": "q", "gold_chapter_id": "ch1", "gold_book_id": "b1"}]
        ranked = [
            [{"chapter_id": f"c{i}", "book_id": "b"} for i in range(5)] + [{"chapter_id": "ch1", "book_id": "b1"}]
        ]
        r = evaluate(golds, ranked, k=5)
        assert r["hit_at_5"] == 0.0
        assert r["mrr"] == pytest.approx(1 / 6)


class TestLoadGold:
    def test_merges_files_and_tags_source(self, tmp_path):
        a = tmp_path / "auto.json"
        b = tmp_path / "manual.json"
        a.write_text(json.dumps([{"query": "x", "gold_chapter_id": "c", "gold_book_id": "b", "source": "auto"}]))
        b.write_text(json.dumps([{"query": "y", "gold_chapter_id": None, "gold_book_id": "b", "source": "manual"}]))
        golds = load_gold([a, b])
        assert {g["source"] for g in golds} == {"auto", "manual"}

    def test_missing_file_ok(self, tmp_path):
        a = tmp_path / "auto.json"
        a.write_text(json.dumps([]))
        assert load_gold([a, tmp_path / "nope.json"]) == []

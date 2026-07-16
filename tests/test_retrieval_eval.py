"""Retrieval eval core: matrix loading, ranking, hit@5/MRR scoring."""

import io
import json
import sqlite3

import numpy as np
import pytest

from src.utils.retrieval_eval import (
    _parse_fenced_json,
    build_gold,
    evaluate,
    load_gold,
    load_matrix,
    rank_chapters_semantic,
    run_eval,
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

    def test_mismatched_lengths_raises(self):
        golds = [
            {"query": "q1", "gold_chapter_id": "ch1", "gold_book_id": "b1"},
            {"query": "q2", "gold_chapter_id": "ch2", "gold_book_id": "b1"},
        ]
        ranked = [[{"chapter_id": "ch1", "book_id": "b1"}]]
        with pytest.raises(ValueError, match=r"2.*1|1.*2"):
            evaluate(golds, ranked, k=5)


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


class FakeEmbedder:
    def generate(self, text):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


class TestRunEval:
    def test_report_shape_and_empty_table_raises(self, eval_db, tmp_path, monkeypatch):
        conn = sqlite3.connect(eval_db)
        conn.execute(
            """CREATE TABLE chunks_staging (
                id TEXT PRIMARY KEY, chapter_id TEXT NOT NULL, book_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL, content TEXT NOT NULL,
                word_count INTEGER NOT NULL, embedding BLOB, embedding_model TEXT,
                content_hash TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
        )
        conn.commit()
        conn.close()

        gold_path = tmp_path / "gold.json"
        gold_path.write_text(
            json.dumps(
                [
                    {"query": "alpha query", "gold_chapter_id": "ch1", "gold_book_id": "b1", "source": "auto"},
                    {"query": "beta query", "gold_chapter_id": None, "gold_book_id": "b1", "source": "manual"},
                ]
            )
        )

        monkeypatch.setattr("src.utils.retrieval_eval.full_text_search", lambda *a, **k: {"results": []})

        report = run_eval(eval_db, [gold_path], table="chunks", embedder=FakeEmbedder())
        assert set(report) == {"auto", "manual"}
        for source in report:
            assert set(report[source]) == {"semantic", "hybrid"}
            for mode in report[source]:
                assert set(report[source][mode]) == {"hit_at_5", "mrr", "n"}

        with pytest.raises(RuntimeError):
            run_eval(eval_db, [gold_path], table="chunks_staging", embedder=FakeEmbedder())


@pytest.fixture
def gold_db(tmp_path):
    db = tmp_path / "library.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE books (id TEXT PRIMARY KEY, title TEXT)")
    conn.execute(
        """CREATE TABLE chapters (
            id TEXT PRIMARY KEY, book_id TEXT NOT NULL, title TEXT,
            chapter_number INTEGER, file_path TEXT, word_count INTEGER)"""
    )
    # 3 books x 3 chapters, each with a real content file
    for b in range(3):
        conn.execute("INSERT INTO books VALUES (?, ?)", (f"b{b}", f"Book {b}"))
        for c in range(3):
            path = tmp_path / f"b{b}-c{c}.md"
            path.write_text(" ".join(f"b{b}c{c}word{i}." for i in range(600)))
            conn.execute(
                "INSERT INTO chapters VALUES (?,?,?,?,?,?)",
                (f"b{b}ch{c}", f"b{b}", f"Ch {c}", c, str(path), 600),
            )
    conn.commit()
    conn.close()
    return db


class TestParseFencedJson:
    def test_plain_json(self):
        assert _parse_fenced_json('{"query": "q"}') == {"query": "q"}

    def test_fenced_json(self):
        out = _parse_fenced_json('Here:\n```json\n{"query": "q"}\n```\ndone')
        assert out == {"query": "q"}

    def test_garbage_returns_none(self):
        assert _parse_fenced_json("no json here") is None


class TestBuildGold:
    def test_deterministic_sampling_and_records(self, gold_db, tmp_path):
        calls = []

        def fake_generator(passage):
            calls.append(passage)
            return f"question {len(calls)}?"

        out = tmp_path / "gold.json"
        n1 = build_gold(gold_db, out, n=6, seed=42, min_words=100, passage_words=50, generator=fake_generator)
        first_passages = list(calls)
        records = json.loads(out.read_text())
        assert n1 == 6 == len(records)
        for r in records:
            assert set(r) == {"query", "gold_chapter_id", "gold_book_id", "source"}
            assert r["source"] == "auto"

        # rerun with same seed -> identical passages sampled
        calls.clear()
        build_gold(gold_db, out, n=6, seed=42, min_words=100, passage_words=50, generator=fake_generator)
        assert calls == first_passages

    def test_stratifies_across_books(self, gold_db, tmp_path):
        out = tmp_path / "gold.json"
        build_gold(gold_db, out, n=3, seed=1, min_words=100, passage_words=50, generator=lambda p: "q?")
        records = json.loads(out.read_text())
        assert len({r["gold_book_id"] for r in records}) == 3

    def test_generator_failure_skips_and_logs(self, gold_db, tmp_path):
        def flaky(passage):
            return None  # simulates claude -p failure / unparseable output

        out = tmp_path / "gold.json"
        n = build_gold(gold_db, out, n=4, seed=7, min_words=100, passage_words=50, generator=flaky)
        assert n == 0
        assert json.loads(out.read_text()) == []

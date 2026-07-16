"""rechunk staging: hash-keyed reconcile, embedding, snapshot marker."""

import io
import sqlite3

import numpy as np
import pytest

from agentic_pipeline.library import rechunk as rc


def _blob(vec):
    buf = io.BytesIO()
    np.save(buf, np.asarray(vec, dtype=np.float32))
    return buf.getvalue()


class FakeGenerator:
    """Deterministic stand-in for OpenAIEmbeddingGenerator."""

    def __init__(self):
        self.calls = []

    def generate_batch(self, texts):
        self.calls.append(list(texts))
        return np.stack([np.full(4, float(len(t)), dtype=np.float32) for t in texts])


@pytest.fixture
def staged_db(tmp_path, monkeypatch):
    db = tmp_path / "library.db"
    monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(db))
    monkeypatch.setenv("BOOK_DB_PATH", str(db))
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE books (id TEXT PRIMARY KEY, title TEXT)")
    conn.execute(
        """CREATE TABLE chapters (
            id TEXT PRIMARY KEY, book_id TEXT NOT NULL, title TEXT,
            chapter_number INTEGER, file_path TEXT, word_count INTEGER)"""
    )
    conn.execute(
        """CREATE TABLE chunks (
            id TEXT PRIMARY KEY, chapter_id TEXT NOT NULL, book_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL, content TEXT NOT NULL,
            word_count INTEGER NOT NULL, embedding BLOB, embedding_model TEXT,
            content_hash TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
    )
    conn.execute("CREATE UNIQUE INDEX idx_chunks_chapter_index ON chunks(chapter_id, chunk_index)")
    conn.execute("INSERT INTO books VALUES ('b1', 'Book One')")
    # chapter with a readable file (wall of text, will produce multiple windows)
    ch_file = tmp_path / "ch1.md"
    ch_file.write_text(" ".join(f"Sentence number {i} has five words." for i in range(400)))
    conn.execute("INSERT INTO chapters VALUES ('ch1', 'b1', 'One', 1, ?, 2400)", (str(ch_file),))
    # chapter whose source file is GONE, but has live chunks (carried case)
    conn.execute(
        "INSERT INTO chapters VALUES ('ch2', 'b1', 'Two', 2, ?, 100)",
        (str(tmp_path / "missing.md"),),
    )
    conn.execute(
        "INSERT INTO chunks VALUES ('old1', 'ch2', 'b1', 0, 'legacy content', 2, ?, 'text-embedding-3-large', 'oldhash', '2026-01-01')",
        (_blob([1, 2, 3, 4]),),
    )
    conn.commit()
    yield conn, db
    conn.close()


class TestStaging:
    def test_stage_all_populates_staging_and_never_touches_chunks(self, staged_db):
        conn, db = staged_db
        before = conn.execute("SELECT COUNT(*), COALESCE(SUM(LENGTH(content)),0) FROM chunks").fetchone()
        rc.ensure_staging(conn)
        report = rc.stage_all(conn)
        after = conn.execute("SELECT COUNT(*), COALESCE(SUM(LENGTH(content)),0) FROM chunks").fetchone()
        assert tuple(before) == tuple(after), "stage must not mutate chunks"
        n = conn.execute("SELECT COUNT(*) FROM chunks_staging").fetchone()[0]
        assert n >= 3  # ch1 windows + carried ch2 row
        assert report["carried_chapters"] == ["ch2"]

    def test_carried_chapter_keeps_live_rows_and_embeddings(self, staged_db):
        conn, db = staged_db
        rc.ensure_staging(conn)
        rc.stage_all(conn)
        row = conn.execute("SELECT content, embedding FROM chunks_staging WHERE chapter_id = 'ch2'").fetchone()
        assert row["content"] == "legacy content"
        assert row["embedding"] is not None

    def test_hash_reconcile_preserves_embeddings_on_rerun(self, staged_db):
        conn, db = staged_db
        rc.ensure_staging(conn)
        rc.stage_all(conn)
        gen = FakeGenerator()
        embedded = rc.embed_pending(conn, generator=gen)
        assert embedded > 0
        # rerun stage: identical chunker output -> all embeddings reused.
        # reused_embeddings also includes ch2's carried row (always re-carried
        # since its source file stays missing), so the invariant is "at least
        # as many reused as were embedded last run", not exact equality.
        report2 = rc.stage_all(conn)
        assert report2["pending_embeddings"] == 0
        assert report2["reused_embeddings"] >= embedded
        gen2 = FakeGenerator()
        assert rc.embed_pending(conn, generator=gen2) == 0
        assert gen2.calls == []

    def test_param_change_reembeds_only_changed(self, staged_db):
        conn, db = staged_db
        rc.ensure_staging(conn)
        rc.stage_all(conn)
        rc.embed_pending(conn, generator=FakeGenerator())

        # different chunker output -> old rows replaced, need embedding again
        def tiny_chunker(text, **kw):
            words = text.split()
            half = len(words) // 2
            return [
                {"chunk_index": 0, "content": " ".join(words[:half]), "word_count": half, "token_count": half},
                {
                    "chunk_index": 1,
                    "content": " ".join(words[half:]),
                    "word_count": len(words) - half,
                    "token_count": len(words) - half,
                },
            ]

        report = rc.stage_all(conn, chunk_fn=tiny_chunker)
        assert report["pending_embeddings"] == 2  # ch1 re-chunked; ch2 carried untouched
        assert report["reused_embeddings"] >= 1  # ch2's carried row

    def test_embed_pending_writes_numpy_blobs_and_model(self, staged_db):
        conn, db = staged_db
        rc.ensure_staging(conn)
        rc.stage_all(conn)
        rc.embed_pending(conn, generator=FakeGenerator())
        row = conn.execute(
            "SELECT embedding, embedding_model FROM chunks_staging WHERE chapter_id='ch1' LIMIT 1"
        ).fetchone()
        vec = np.load(io.BytesIO(row["embedding"]))
        assert vec.shape == (4,)
        assert row["embedding_model"] == "text-embedding-3-large"

    def test_snapshot_marker(self, staged_db):
        conn, db = staged_db
        marker = rc.snapshot_marker(conn)
        assert marker["chunk_count"] == 1
        assert marker["max_created_at"] == "2026-01-01"

    def test_estimate_cost_counts_pending_only(self, staged_db):
        conn, db = staged_db
        rc.ensure_staging(conn)
        rc.stage_all(conn)
        est = rc.estimate_embedding_cost(conn)
        assert est["pending"] > 0
        assert est["est_usd"] > 0
        rc.embed_pending(conn, generator=FakeGenerator())
        assert rc.estimate_embedding_cost(conn)["pending"] == 0


class ShortGenerator:
    """Returns one vector fewer than requested — must raise, never loop."""

    def generate_batch(self, texts):
        return np.stack([np.full(4, 1.0, dtype=np.float32) for _ in texts[:-1]]) if len(texts) > 1 else np.empty((0, 4), dtype=np.float32)


class TestEmbedPendingGuard:
    def test_short_generator_batch_raises(self, staged_db):
        conn, db = staged_db
        rc.ensure_staging(conn)
        rc.stage_all(conn)
        with pytest.raises(RuntimeError, match="vectors for"):
            rc.embed_pending(conn, generator=ShortGenerator())

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
        return (
            np.stack([np.full(4, 1.0, dtype=np.float32) for _ in texts[:-1]])
            if len(texts) > 1
            else np.empty((0, 4), dtype=np.float32)
        )


class TestEmbedPendingGuard:
    def test_short_generator_batch_raises(self, staged_db):
        conn, db = staged_db
        rc.ensure_staging(conn)
        rc.stage_all(conn)
        with pytest.raises(RuntimeError, match="vectors for"):
            rc.embed_pending(conn, generator=ShortGenerator())


class TestGate:
    BASE = {
        "auto": {
            "semantic": {"hit_at_5": 0.60, "mrr": 0.40, "n": 60},
            "hybrid": {"hit_at_5": 0.70, "mrr": 0.50, "n": 60},
        },
        "manual": {
            "semantic": {"hit_at_5": 0.20, "mrr": 0.15, "n": 10},
            "hybrid": {"hit_at_5": 0.30, "mrr": 0.20, "n": 10},
        },
    }

    def _staged(self, auto_hit, auto_mrr, manual_hit):
        return {
            "auto": {
                "semantic": {"hit_at_5": auto_hit, "mrr": auto_mrr, "n": 60},
                "hybrid": {"hit_at_5": 0.0, "mrr": 0.0, "n": 60},
            },
            "manual": {
                "semantic": {"hit_at_5": manual_hit, "mrr": 0.5, "n": 10},
                "hybrid": {"hit_at_5": 0.0, "mrr": 0.0, "n": 10},
            },
        }

    def test_pass_requires_all_three_arms(self):
        assert rc.gate_pass(self.BASE, self._staged(0.60, 0.40, 0.30)) is True

    def test_auto_hit_regression_fails(self):
        assert rc.gate_pass(self.BASE, self._staged(0.59, 0.40, 0.30)) is False

    def test_auto_mrr_regression_fails(self):
        assert rc.gate_pass(self.BASE, self._staged(0.60, 0.39, 0.30)) is False

    def test_manual_equal_is_not_strict_improvement(self):
        assert rc.gate_pass(self.BASE, self._staged(0.60, 0.40, 0.20)) is False

    def test_hybrid_numbers_never_gate(self):
        # staged hybrid is 0.0 everywhere; gate still passes on semantic arms
        assert rc.gate_pass(self.BASE, self._staged(0.65, 0.45, 0.40)) is True


class TestVerdictPersistence:
    def test_run_gate_eval_persists_verdict(self, staged_db, tmp_path, monkeypatch):
        conn, db = staged_db
        rc.ensure_staging(conn)
        rc.stage_all(conn)
        rc.embed_pending(conn, generator=FakeGenerator())

        gold = tmp_path / "gold.json"
        gold.write_text(
            '[{"query": "q", "gold_chapter_id": "ch1", "gold_book_id": "b1", "source": "auto"},'
            ' {"query": "q2", "gold_chapter_id": "ch1", "gold_book_id": "b1", "source": "manual"}]'
        )

        class FakeEmbedder:
            def generate(self, text):
                return np.full(4, 5.0, dtype=np.float32)

        # avoid the FTS arm in unit tests (no chapters_fts table in fixture)
        monkeypatch.setattr(
            "src.utils.retrieval_eval.full_text_search",
            lambda q, limit=10: {"results": []},
        )
        verdict = rc.run_gate_eval(db, [gold], embedder=FakeEmbedder())
        assert "baseline" in verdict and "staged" in verdict
        assert isinstance(verdict["pass"], bool)
        assert verdict["snapshot"]["chunk_count"] == 1

        loaded = rc.load_verdict(db)
        assert loaded == verdict

    def test_load_verdict_none_when_absent(self, tmp_path):
        assert rc.load_verdict(tmp_path / "library.db") is None

    def test_run_gate_eval_refuses_empty_manual_gold_subset(self, staged_db, tmp_path, monkeypatch):
        conn, db = staged_db
        rc.ensure_staging(conn)
        rc.stage_all(conn)
        rc.embed_pending(conn, generator=FakeGenerator())

        gold = tmp_path / "gold.json"
        gold.write_text('[{"query": "q", "gold_chapter_id": "ch1", "gold_book_id": "b1", "source": "auto"}]')

        class FakeEmbedder:
            def generate(self, text):
                return np.full(4, 5.0, dtype=np.float32)

        monkeypatch.setattr(
            "src.utils.retrieval_eval.full_text_search",
            lambda q, limit=10: {"results": []},
        )
        with pytest.raises(RuntimeError, match="manual"):
            rc.run_gate_eval(db, [gold], embedder=FakeEmbedder())

    def test_run_gate_eval_refuses_empty_auto_gold_subset(self, staged_db, tmp_path, monkeypatch):
        conn, db = staged_db
        rc.ensure_staging(conn)
        rc.stage_all(conn)
        rc.embed_pending(conn, generator=FakeGenerator())

        gold = tmp_path / "gold.json"
        gold.write_text('[{"query": "q", "gold_chapter_id": "ch1", "gold_book_id": "b1", "source": "manual"}]')

        class FakeEmbedder:
            def generate(self, text):
                return np.full(4, 5.0, dtype=np.float32)

        monkeypatch.setattr(
            "src.utils.retrieval_eval.full_text_search",
            lambda q, limit=10: {"results": []},
        )
        with pytest.raises(RuntimeError, match="auto"):
            rc.run_gate_eval(db, [gold], embedder=FakeEmbedder())


def _prep_passing_swap(conn, db, tmp_path):
    """Stage + embed + write a PASS verdict with a current snapshot."""
    import json as _json

    rc.ensure_staging(conn)
    rc.stage_all(conn)
    rc.embed_pending(conn, generator=FakeGenerator())
    # library_meta must exist for the version bump
    conn.execute(
        "CREATE TABLE IF NOT EXISTS library_meta "
        "(id INTEGER PRIMARY KEY CHECK (id = 1), data_version INTEGER NOT NULL DEFAULT 1)"
    )
    conn.execute("INSERT OR IGNORE INTO library_meta (id, data_version) VALUES (1, 1)")
    conn.commit()
    verdict_path = db.parent / "rechunk" / "last-verdict.json"
    verdict_path.parent.mkdir(parents=True, exist_ok=True)
    verdict_path.write_text(
        _json.dumps(
            {
                "pass": True,
                "baseline": {},
                "staged": {},
                "snapshot": rc.snapshot_marker(conn),
            }
        )
    )


class TestSwap:
    def test_swap_refused_without_pass_verdict(self, staged_db):
        conn, db = staged_db
        rc.ensure_staging(conn)
        with pytest.raises(rc.SwapRefused):
            rc.swap(db)

    def test_swap_refused_with_fail_verdict(self, staged_db):
        import json as _json

        conn, db = staged_db
        rc.ensure_staging(conn)
        p = db.parent / "rechunk" / "last-verdict.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_json.dumps({"pass": False, "snapshot": {}}))
        with pytest.raises(rc.SwapRefused):
            rc.swap(db)

    def test_swap_refused_with_unembedded_staging(self, staged_db, tmp_path):
        conn, db = staged_db
        _prep_passing_swap(conn, db, tmp_path)
        conn.execute("UPDATE chunks_staging SET embedding = NULL WHERE rowid = 1")
        conn.commit()
        with pytest.raises(rc.SwapRefused):
            rc.swap(db)

    def test_swap_refused_when_staging_missing(self, staged_db, tmp_path):
        conn, db = staged_db
        _prep_passing_swap(conn, db, tmp_path)
        conn.execute("DROP TABLE chunks_staging")
        conn.commit()
        with pytest.raises(rc.SwapRefused, match="does not exist"):
            rc.swap(db)

    def test_swap_refused_when_staging_empty(self, staged_db, tmp_path, monkeypatch):
        conn, db = staged_db
        _prep_passing_swap(conn, db, tmp_path)
        conn.execute("DELETE FROM chunks_staging")
        conn.commit()
        # ch2's live chunk would otherwise get re-staged by the delta step
        # (it's still joinable against `chunks`); stub it out so this test
        # isolates the "staging is empty" precondition itself.
        monkeypatch.setattr(rc, "_stage_delta", lambda conn, marker, generator=None: 0)
        with pytest.raises(rc.SwapRefused, match="empty"):
            rc.swap(db)

    def test_swap_refused_without_library_meta_row(self, staged_db, tmp_path):
        conn, db = staged_db
        _prep_passing_swap(conn, db, tmp_path)
        conn.execute("DELETE FROM library_meta")
        conn.commit()
        with pytest.raises(rc.SwapRefused, match="library_meta"):
            rc.swap(db)

    def test_swap_replaces_chunks_bumps_version_drops_staging(self, staged_db, tmp_path):
        conn, db = staged_db
        _prep_passing_swap(conn, db, tmp_path)
        staged_n = conn.execute("SELECT COUNT(*) FROM chunks_staging").fetchone()[0]

        report = rc.swap(db)

        assert report["chunks"] == staged_n
        live = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert live == staged_n
        nulls = conn.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL").fetchone()[0]
        assert nulls == 0
        version = conn.execute("SELECT data_version FROM library_meta").fetchone()[0]
        assert version == 2 == report["data_version"]
        remaining = conn.execute("SELECT name FROM sqlite_master WHERE name = 'chunks_staging'").fetchone()
        assert remaining is None
        assert (db.parent / db.name).exists()
        backups = list(db.parent.glob(f"{db.name}.backup-doctor-*"))
        assert backups, "swap must take a backup first"

    def test_swap_stages_delta_for_books_added_after_marker(self, staged_db, tmp_path):
        conn, db = staged_db
        _prep_passing_swap(conn, db, tmp_path)
        # simulate the worker approving a new book AFTER the marker
        ch_file = tmp_path / "ch3.md"
        ch_file.write_text(" ".join(f"Delta sentence {i} five words." for i in range(200)))
        conn.execute("INSERT INTO chapters VALUES ('ch3', 'b1', 'Three', 3, ?, 1000)", (str(ch_file),))
        conn.execute(
            "INSERT INTO chunks (id, chapter_id, book_id, chunk_index, content, word_count, embedding, created_at) "
            "VALUES ('new1', 'ch3', 'b1', 0, 'delta live', 2, ?, '2026-12-31')",
            (_blob([9, 9, 9, 9]),),
        )
        conn.commit()

        report = rc.swap(db, generator=FakeGenerator())

        assert report["delta_chapters"] == 1
        survived = conn.execute("SELECT COUNT(*) FROM chunks WHERE chapter_id = 'ch3'").fetchone()[0]
        assert survived >= 1, "book approved mid-run must survive the swap"

    def test_mid_transaction_failure_leaves_chunks_and_version_untouched(self, staged_db, tmp_path, monkeypatch):
        conn, db = staged_db
        _prep_passing_swap(conn, db, tmp_path)
        before_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

        def boom():
            raise RuntimeError("injected failure")

        monkeypatch.setattr(rc, "_pre_commit_hook", boom)
        with pytest.raises(RuntimeError, match="injected failure"):
            rc.swap(db)

        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == before_chunks
        assert conn.execute("SELECT data_version FROM library_meta").fetchone()[0] == 1
        assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'chunks_staging'").fetchone() is not None


class TestRechunkCli:
    def test_stage_flow_reports_and_never_swaps(self, staged_db, tmp_path, monkeypatch):
        from click.testing import CliRunner
        from agentic_pipeline.cli import main

        conn, db = staged_db
        gold = tmp_path / "gold.json"
        gold.write_text("[]")
        monkeypatch.setattr("agentic_pipeline.library.rechunk.GOLD_PATHS", [gold])
        monkeypatch.setattr(
            "agentic_pipeline.library.rechunk.embed_pending",
            lambda conn, generator=None, batch_size=256: 0,
        )
        monkeypatch.setattr(
            "agentic_pipeline.library.rechunk.run_gate_eval",
            lambda db_path, gold_paths=None, embedder=None: {
                "pass": False,
                "baseline": {},
                "staged": {},
                "snapshot": {},
            },
        )
        runner = CliRunner()
        result = runner.invoke(main, ["rechunk", "--yes"])
        assert result.exit_code in (0, 1)  # FAIL verdict exits 1; either way no crash
        before = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert before == 1, "CLI stage flow must not touch chunks"

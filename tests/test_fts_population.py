"""chapters_fts must be filled by code: on approve/embed and via `fts-rebuild`."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from agentic_pipeline.db.migrations import run_migrations


def _make_db(tmp_path: Path, chapters: int = 2) -> Path:
    db_path = tmp_path / "library.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE books (id TEXT PRIMARY KEY, title TEXT, author TEXT)")
    conn.execute(
        "CREATE TABLE chapters (id TEXT PRIMARY KEY, book_id TEXT, title TEXT, chapter_number INTEGER, file_path TEXT)"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE chapters_fts USING fts5(chapter_id, title, content, tokenize='porter unicode61')"
    )
    conn.execute("INSERT INTO books VALUES ('b1', 'Test Book', 'Author')")
    books_dir = tmp_path / "books"
    books_dir.mkdir()
    for i in range(1, chapters + 1):
        f = books_dir / f"ch{i}.md"
        f.write_text(f"Chapter {i} discusses zebrafoo{i} and kubernetes.")
        conn.execute("INSERT INTO chapters VALUES (?, 'b1', ?, ?, ?)", (f"c{i}", f"Title {i}", i, str(f)))
    conn.commit()
    conn.close()
    run_migrations(db_path)
    return db_path


def _fts_ids(db_path: Path) -> list[str]:
    conn = sqlite3.connect(db_path)
    try:
        return sorted(r[0] for r in conn.execute("SELECT chapter_id FROM chapters_fts"))
    finally:
        conn.close()


def _match(db_path: Path, term: str) -> list[str]:
    conn = sqlite3.connect(db_path)
    try:
        return [r[0] for r in conn.execute("SELECT chapter_id FROM chapters_fts WHERE chapters_fts MATCH ?", (term,))]
    finally:
        conn.close()


class TestSyncFts:
    def test_sync_fts_indexes_book_chapters(self, tmp_path):
        from agentic_pipeline.library.fts_index import sync_book_fts

        db_path = _make_db(tmp_path)
        assert sync_book_fts(db_path, "b1") == 2
        assert _fts_ids(db_path) == ["c1", "c2"]
        assert _match(db_path, "zebrafoo2") == ["c2"]

    def test_sync_fts_is_idempotent(self, tmp_path):
        from agentic_pipeline.library.fts_index import sync_book_fts

        db_path = _make_db(tmp_path)
        sync_book_fts(db_path, "b1")
        sync_book_fts(db_path, "b1")
        assert _fts_ids(db_path) == ["c1", "c2"]

    def test_sync_fts_skips_when_table_missing(self, tmp_path):
        from agentic_pipeline.library.fts_index import sync_book_fts

        db_path = _make_db(tmp_path)
        conn = sqlite3.connect(db_path)
        conn.execute("DROP TABLE chapters_fts")
        conn.commit()
        conn.close()
        assert sync_book_fts(db_path, "b1") == 0

    def test_unreadable_chapter_is_skipped_not_fatal(self, tmp_path):
        from agentic_pipeline.library.fts_index import sync_book_fts

        db_path = _make_db(tmp_path)
        (tmp_path / "books" / "ch1.md").unlink()
        assert sync_book_fts(db_path, "b1") == 1
        assert _fts_ids(db_path) == ["c2"]


class TestApproveIndexesFts:
    def test_complete_approved_syncs_fts_before_complete(self, tmp_path):
        from agentic_pipeline.approval.actions import _complete_approved
        from agentic_pipeline.db.pipelines import PipelineRepository
        from agentic_pipeline.pipeline.states import PipelineState

        db_path = _make_db(tmp_path)
        repo = PipelineRepository(db_path)
        pid = repo.create("/book.epub", "hash_fts")
        for st in [
            PipelineState.HASHING,
            PipelineState.CLASSIFYING,
            PipelineState.SELECTING_STRATEGY,
            PipelineState.PROCESSING,
            PipelineState.VALIDATING,
            PipelineState.PENDING_APPROVAL,
            PipelineState.APPROVED,
        ]:
            repo.update_state(pid, st)

        ok = MagicMock(success=True, chapters_processed=2, error=None)
        fake_mod = MagicMock()
        fake_mod.ProcessingAdapter.return_value.generate_embeddings.return_value = ok
        with patch.dict("sys.modules", {"agentic_pipeline.adapters.processing_adapter": fake_mod}):
            result = _complete_approved(db_path, pid, {"processing_result": {"book_id": "b1"}})

        assert result["state"] == PipelineState.COMPLETE.value
        assert _fts_ids(db_path) == ["c1", "c2"]


class TestRebuild:
    def test_rebuild_is_idempotent_no_duplicates(self, tmp_path):
        from src.utils.fts_search import rebuild_fts_index

        db_path = _make_db(tmp_path)

        def reader(fp):
            return Path(fp).read_text()

        first = rebuild_fts_index(db_path=db_path, reader=reader)
        second = rebuild_fts_index(db_path=db_path, reader=reader)
        assert first["entries_after"] == 2 and second["entries_after"] == 2
        assert second["entries_before"] == 2
        assert _fts_ids(db_path) == ["c1", "c2"]

    def test_full_text_search_finds_synced_chapter(self, tmp_path):
        """full_text_search (the text_search backend) returns hits once populated."""
        from src.utils import fts_search
        from agentic_pipeline.library.fts_index import sync_book_fts

        db_path = _make_db(tmp_path)
        sync_book_fts(db_path, "b1")

        from contextlib import contextmanager

        @contextmanager
        def conn_ctx():
            c = sqlite3.connect(db_path)
            c.row_factory = sqlite3.Row
            try:
                yield c
            finally:
                c.close()

        with patch.object(fts_search, "get_db_connection", conn_ctx):
            result = fts_search.full_text_search("zebrafoo1")
        assert "error" not in result, result
        assert result["total_found"] >= 1


class TestFtsRebuildCli:
    def test_cli_rebuilds_index(self, tmp_path, monkeypatch):
        from agentic_pipeline.cli import main

        db_path = _make_db(tmp_path)
        monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(db_path))
        result = CliRunner().invoke(main, ["fts-rebuild"])
        assert result.exit_code == 0, result.output
        assert _fts_ids(db_path) == ["c1", "c2"]

    def test_cli_exits_1_when_table_missing(self, tmp_path, monkeypatch):
        from agentic_pipeline.cli import main

        db_path = _make_db(tmp_path)
        conn = sqlite3.connect(db_path)
        conn.execute("DROP TABLE chapters_fts")
        conn.commit()
        conn.close()
        monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(db_path))
        result = CliRunner().invoke(main, ["fts-rebuild"])
        assert result.exit_code == 1

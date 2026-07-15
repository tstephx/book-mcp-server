"""Tests for approval actions."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


@pytest.fixture
def db_path():
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    yield path
    path.unlink(missing_ok=True)


def _mock_embedding_result(success=True, chapters_processed=5, error=None):
    """Create a mock EmbeddingResult."""
    from agentic_pipeline.adapters.processing_adapter import EmbeddingResult

    return EmbeddingResult(
        success=success,
        chapters_processed=chapters_processed,
        error=error,
    )


def _create_pending_pipeline(db_path, processing_result=None):
    """Create a pipeline in PENDING_APPROVAL state via valid transitions."""
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "hash123")
    # Walk through valid transitions to reach PENDING_APPROVAL
    for state in [
        PipelineState.HASHING,
        PipelineState.CLASSIFYING,
        PipelineState.SELECTING_STRATEGY,
        PipelineState.PROCESSING,
        PipelineState.VALIDATING,
        PipelineState.PENDING_APPROVAL,
    ]:
        repo.update_state(pid, state)
    if processing_result is not None:
        repo.update_processing_result(pid, processing_result)
    return pid


# ---------------------------------------------------------------------------
# approve_book() — tests the new non-blocking contract
# ---------------------------------------------------------------------------


def test_approve_book(db_path):
    """approve_book() returns immediately with state=approved, embedding=queued."""
    from agentic_pipeline.approval.actions import approve_book
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    with patch("agentic_pipeline.approval.actions._run_embedding_background"):
        pid = _create_pending_pipeline(db_path)
        result = approve_book(db_path, pid, actor="human:taylor")

    assert result["success"] is True
    assert result["state"] == PipelineState.APPROVED.value
    assert result["embedding"] == "queued"

    pipeline = PipelineRepository(db_path).get(pid)
    assert pipeline["state"] == PipelineState.APPROVED.value
    assert pipeline["approved_by"] == "human:taylor"


class TestResolveSourceFile:
    """Find a book whose record predates the archive-tracking fix."""

    def test_prefers_the_recorded_path(self, tmp_path):
        from agentic_pipeline.approval.actions import resolve_source_file

        book = tmp_path / "b.epub"
        book.write_bytes(b"x")

        assert resolve_source_file(str(book)) == book

    def test_falls_back_to_the_processed_dir(self, tmp_path):
        """142 of 307 completed books point at a path auto-archive emptied."""
        from agentic_pipeline.approval import actions

        processed = tmp_path / "processed"
        processed.mkdir()
        archived = processed / "b.epub"
        archived.write_bytes(b"x")
        stale = tmp_path / "watch" / "b.epub"  # never existed / moved away

        with patch.object(actions, "_get_processed_dir", return_value=processed):
            assert actions.resolve_source_file(str(stale)) == archived

    def test_returns_none_when_the_file_is_gone(self, tmp_path):
        from agentic_pipeline.approval import actions

        processed = tmp_path / "processed"
        processed.mkdir()

        with patch.object(actions, "_get_processed_dir", return_value=processed):
            assert actions.resolve_source_file(str(tmp_path / "nope.epub")) is None

    def test_handles_no_processed_dir_configured(self, tmp_path):
        from agentic_pipeline.approval import actions

        with patch.object(actions, "_get_processed_dir", return_value=None):
            assert actions.resolve_source_file(str(tmp_path / "nope.epub")) is None

    def test_handles_empty_source_path(self):
        from agentic_pipeline.approval.actions import resolve_source_file

        assert resolve_source_file(None) is None
        assert resolve_source_file("") is None


class TestArchiveRecordsNewLocation:
    """Archiving moves the file; the record must learn where it went.

    Regression: _complete_approved() called _archive_source_file() and discarded
    its return, so source_path kept pointing at a file that no longer existed.
    reingest checks that path and refuses — broken for 142 of 307 completed books
    (46%). Collisions make it worse: the archive renames to 'stem_1.epub', so
    even guessing by basename fails.
    """

    @patch("agentic_pipeline.adapters.processing_adapter.ProcessingAdapter", autospec=True)
    def test_source_path_points_at_the_archived_file(self, MockAdapter, db_path, tmp_path):
        from agentic_pipeline.approval import actions
        from agentic_pipeline.db.pipelines import PipelineRepository

        MockAdapter.return_value.generate_embeddings.return_value = _mock_embedding_result()

        watch, processed = tmp_path / "watch", tmp_path / "processed"
        watch.mkdir()
        processed.mkdir()
        book = watch / "b.epub"
        book.write_bytes(b"epub")

        pid, pipeline = _setup_approved_pipeline(db_path)
        PipelineRepository(db_path).update_source_path(pid, str(book))
        pipeline = PipelineRepository(db_path).get(pid)

        with patch.object(actions, "_get_processed_dir", return_value=processed):
            actions._complete_approved(db_path, pid, pipeline)

        assert PipelineRepository(db_path).get(pid)["source_path"] == str(processed / "b.epub")

    @patch("agentic_pipeline.adapters.processing_adapter.ProcessingAdapter", autospec=True)
    def test_records_the_collision_renamed_filename(self, MockAdapter, db_path, tmp_path):
        """A colliding archive becomes stem_1.epub — record that, not the original."""
        from agentic_pipeline.approval import actions
        from agentic_pipeline.db.pipelines import PipelineRepository

        MockAdapter.return_value.generate_embeddings.return_value = _mock_embedding_result()

        watch, processed = tmp_path / "watch", tmp_path / "processed"
        watch.mkdir()
        processed.mkdir()
        (processed / "b.epub").write_bytes(b"older book, same name")
        book = watch / "b.epub"
        book.write_bytes(b"epub")

        pid, pipeline = _setup_approved_pipeline(db_path)
        PipelineRepository(db_path).update_source_path(pid, str(book))
        pipeline = PipelineRepository(db_path).get(pid)

        with patch.object(actions, "_get_processed_dir", return_value=processed):
            actions._complete_approved(db_path, pid, pipeline)

        assert PipelineRepository(db_path).get(pid)["source_path"] == str(processed / "b_1.epub")


class TestApproveBookForeground:
    """A caller that exits on return must embed before returning.

    Regression: approve_book() spawns embedding on a daemon thread, which dies
    with the process. That is correct for the long-lived MCP server but silently
    wrong for `agentic-pipeline approve`, which prints and exits — the book was
    left in APPROVED with no chunks and no error. It only ever completed because
    a background worker happened to sweep up APPROVED books.
    """

    def test_foreground_embeds_before_returning(self, db_path):
        from agentic_pipeline.approval import actions
        from agentic_pipeline.pipeline.states import PipelineState

        calls = []

        def fake_complete(db, pid, pipeline):
            calls.append(pid)
            return {"state": PipelineState.COMPLETE.value, "chapters_embedded": 7}

        with patch.object(actions, "_complete_approved", fake_complete):
            pid = _create_pending_pipeline(db_path)
            result = actions.approve_book(db_path, pid, actor="human:cli", background=False)

        assert calls == [pid], "embedding must run before approve_book returns"
        assert result["success"] is True
        assert result["state"] == PipelineState.COMPLETE.value
        assert result["chapters_embedded"] == 7

    def test_foreground_surfaces_embedding_failure(self, db_path):
        from agentic_pipeline.approval import actions
        from agentic_pipeline.pipeline.states import PipelineState

        def fake_complete(db, pid, pipeline):
            return {"state": PipelineState.NEEDS_RETRY.value, "embedding_error": "quota exceeded"}

        with patch.object(actions, "_complete_approved", fake_complete):
            pid = _create_pending_pipeline(db_path)
            result = actions.approve_book(db_path, pid, actor="human:cli", background=False)

        assert result["state"] == PipelineState.NEEDS_RETRY.value
        assert result["embedding_error"] == "quota exceeded"

    def test_foreground_does_not_spawn_a_thread(self, db_path):
        from agentic_pipeline.approval import actions
        from agentic_pipeline.pipeline.states import PipelineState

        def fake_complete(db, pid, pipeline):
            return {"state": PipelineState.COMPLETE.value, "chapters_embedded": 1}

        with patch.object(actions, "_complete_approved", fake_complete):
            with patch.object(actions.threading, "Thread") as MockThread:
                pid = _create_pending_pipeline(db_path)
                actions.approve_book(db_path, pid, actor="human:cli", background=False)

        MockThread.assert_not_called()

    def test_background_remains_the_default(self, db_path):
        """The MCP server must keep its non-blocking behaviour."""
        from agentic_pipeline.approval import actions
        from agentic_pipeline.pipeline.states import PipelineState

        with patch.object(actions, "_run_embedding_background"):
            pid = _create_pending_pipeline(db_path)
            result = actions.approve_book(db_path, pid, actor="mcp")

        assert result["state"] == PipelineState.APPROVED.value
        assert result["embedding"] == "queued"


def test_reject_book(db_path):
    from agentic_pipeline.approval.actions import reject_book
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    pid = _create_pending_pipeline(db_path)
    result = reject_book(db_path, pid, reason="Poor quality extraction", actor="human:taylor")

    assert result["success"] is True

    pipeline = PipelineRepository(db_path).get(pid)
    assert pipeline["state"] == PipelineState.REJECTED.value


def test_approve_creates_audit_record(db_path):
    import sqlite3

    from agentic_pipeline.approval.actions import approve_book

    with patch("agentic_pipeline.approval.actions._run_embedding_background"):
        pid = _create_pending_pipeline(db_path)
        approve_book(db_path, pid, actor="human:taylor")

    # Audit is written before the thread starts — always present
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM approval_audit WHERE pipeline_id = ?", (pid,))
    audit = cursor.fetchone()
    conn.close()

    assert audit is not None
    assert audit["action"] == "approved"
    assert audit["actor"] == "human:taylor"


def test_audit_record_reflects_actual_autonomy_mode(db_path):
    """Audit trail must record the real autonomy mode, not hardcoded 'supervised'."""
    import sqlite3

    from agentic_pipeline.approval.actions import approve_book
    from agentic_pipeline.autonomy.config import AutonomyConfig

    AutonomyConfig(db_path).set_mode("partial")

    with patch("agentic_pipeline.approval.actions._run_embedding_background"):
        pid = _create_pending_pipeline(db_path)
        approve_book(db_path, pid, actor="human:taylor")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT autonomy_mode FROM approval_audit WHERE pipeline_id = ?", (pid,))
    audit = cursor.fetchone()
    conn.close()

    assert audit["autonomy_mode"] == "partial", (
        f"Expected 'partial' but got '{audit['autonomy_mode']}' — hardcoded 'supervised' still in place"
    )


# ---------------------------------------------------------------------------
# _complete_approved() — tests embedding behaviour (runs in background thread)
# ---------------------------------------------------------------------------


def _setup_approved_pipeline(db_path, processing_result=None):
    """Create a pipeline in APPROVED state, return (pipeline_id, pipeline_dict)."""
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    pid = _create_pending_pipeline(db_path, processing_result=processing_result)

    repo = PipelineRepository(db_path)
    with patch("agentic_pipeline.approval.actions._run_embedding_background"):
        from agentic_pipeline.approval.actions import approve_book

        approve_book(db_path, pid, actor="human:taylor")

    pipeline = repo.get(pid)
    return pid, pipeline


@patch("agentic_pipeline.adapters.processing_adapter.ProcessingAdapter", autospec=True)
def test_complete_approved_succeeds(MockAdapter, db_path):
    """_complete_approved() drives APPROVED → EMBEDDING → COMPLETE on success."""
    from agentic_pipeline.approval.actions import _complete_approved
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    mock_instance = MockAdapter.return_value
    mock_instance.generate_embeddings.return_value = _mock_embedding_result()

    pid, pipeline = _setup_approved_pipeline(db_path)
    result = _complete_approved(db_path, pid, pipeline)

    assert result["state"] == PipelineState.COMPLETE.value
    assert result["chapters_embedded"] == 5

    assert PipelineRepository(db_path).get(pid)["state"] == PipelineState.COMPLETE.value


@patch("agentic_pipeline.adapters.processing_adapter.ProcessingAdapter", autospec=True)
def test_complete_approved_embedding_failure_goes_to_needs_retry(MockAdapter, db_path):
    """_complete_approved() transitions to NEEDS_RETRY when embedding returns failure."""
    from agentic_pipeline.approval.actions import _complete_approved
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    mock_instance = MockAdapter.return_value
    mock_instance.generate_embeddings.return_value = _mock_embedding_result(
        success=False, chapters_processed=0, error="OpenAI rate limit"
    )

    pid, pipeline = _setup_approved_pipeline(db_path)
    result = _complete_approved(db_path, pid, pipeline)

    assert result["state"] == PipelineState.NEEDS_RETRY.value
    assert "OpenAI rate limit" in result["embedding_error"]
    assert PipelineRepository(db_path).get(pid)["state"] == PipelineState.NEEDS_RETRY.value


@patch("agentic_pipeline.adapters.processing_adapter.ProcessingAdapter", autospec=True)
def test_complete_approved_embedding_exception_goes_to_needs_retry(MockAdapter, db_path):
    """_complete_approved() transitions to NEEDS_RETRY when embedding raises."""
    from agentic_pipeline.approval.actions import _complete_approved
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    mock_instance = MockAdapter.return_value
    mock_instance.generate_embeddings.side_effect = RuntimeError("Connection refused")

    pid, pipeline = _setup_approved_pipeline(db_path)
    result = _complete_approved(db_path, pid, pipeline)

    assert result["state"] == PipelineState.NEEDS_RETRY.value
    assert "Connection refused" in result["embedding_error"]
    assert PipelineRepository(db_path).get(pid)["state"] == PipelineState.NEEDS_RETRY.value


@patch("agentic_pipeline.adapters.processing_adapter.ProcessingAdapter", autospec=True)
def test_complete_approved_no_processing_result_uses_pipeline_id(MockAdapter, db_path):
    """_complete_approved() falls back to pipeline_id as book_id when no processing_result."""
    from agentic_pipeline.approval.actions import _complete_approved

    mock_instance = MockAdapter.return_value
    mock_instance.generate_embeddings.return_value = _mock_embedding_result()

    pid, pipeline = _setup_approved_pipeline(db_path, processing_result=None)
    _complete_approved(db_path, pid, pipeline)

    mock_instance.generate_embeddings.assert_called_once_with(book_id=pid)


@patch("agentic_pipeline.adapters.processing_adapter.ProcessingAdapter", autospec=True)
def test_complete_approved_uses_book_id_from_processing_result(MockAdapter, db_path):
    """_complete_approved() uses book_id from processing_result when present."""
    from agentic_pipeline.approval.actions import _complete_approved

    mock_instance = MockAdapter.return_value
    mock_instance.generate_embeddings.return_value = _mock_embedding_result()

    pid, pipeline = _setup_approved_pipeline(db_path, processing_result={"book_id": "my-book-uuid"})
    _complete_approved(db_path, pid, pipeline)

    mock_instance.generate_embeddings.assert_called_once_with(book_id="my-book-uuid")

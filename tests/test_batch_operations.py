# tests/test_batch_operations.py
"""Tests for batch operations."""

import pytest
import tempfile
from pathlib import Path

from conftest import transition_to


@pytest.fixture
def db_path():
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    yield path
    path.unlink(missing_ok=True)


def test_batch_approve(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.batch import BatchOperations, BatchFilter
    from agentic_pipeline.audit import AuditTrail

    repo = PipelineRepository(db_path)

    # Create pending approval books
    id1 = repo.create("/book1.epub", "hash1")
    transition_to(repo, id1, PipelineState.PENDING_APPROVAL)
    repo.update_book_profile(id1, {"book_type": "technical_tutorial", "confidence": 0.95})

    id2 = repo.create("/book2.epub", "hash2")
    transition_to(repo, id2, PipelineState.PENDING_APPROVAL)
    repo.update_book_profile(id2, {"book_type": "technical_tutorial", "confidence": 0.92})

    ops = BatchOperations(db_path)
    filter = BatchFilter(min_confidence=0.9)

    result = ops.approve(filter, actor="human:taylor", execute=True)

    assert result["approved"] == 2

    # Check audit trail
    trail = AuditTrail(db_path)
    entries = trail.query(action="BATCH_APPROVED")
    assert len(entries) == 1
    assert entries[0]["filter_used"]["min_confidence"] == 0.9


class TestBatchToleratesConcurrentClaims:
    """One contended book must not abort the batch or lose the audit trail.

    update_state now raises ConcurrentModificationError on a lost claim. The
    batch loop called it unguarded, so a single book claimed between the filter
    query and the loop reaching it would abort the whole run — leaving books
    already flipped and audit.log() (called after the loop) never reached.
    """

    def _pending(self, repo, path, hash_, conf):
        from agentic_pipeline.pipeline.states import PipelineState

        pid = repo.create(path, hash_)
        transition_to(repo, pid, PipelineState.PENDING_APPROVAL)
        repo.update_book_profile(pid, {"book_type": "technical_tutorial", "confidence": conf})
        return pid

    def test_a_claimed_book_is_skipped_not_fatal(self, db_path, monkeypatch):
        from agentic_pipeline.db.pipelines import PipelineRepository
        from agentic_pipeline.pipeline.states import PipelineState
        from agentic_pipeline.batch import BatchOperations, BatchFilter
        from agentic_pipeline.audit import AuditTrail
        import agentic_pipeline.approval.actions as actions

        from agentic_pipeline.db.pipelines import ConcurrentModificationError

        repo = PipelineRepository(db_path)
        id1 = self._pending(repo, "/book1.epub", "hash1", 0.95)
        id2 = self._pending(repo, "/book2.epub", "hash2", 0.92)

        monkeypatch.setattr(
            actions,
            "_complete_approved",
            lambda db, pid, pipeline: {"state": PipelineState.COMPLETE.value, "chapters_embedded": 1},
        )

        ops = BatchOperations(db_path)
        real_update = ops.repo.update_state

        def lose_the_claim_on_book1(pipeline_id, new_state, **kw):
            # A rival claims book1 between filter.apply() and the loop reaching it.
            if pipeline_id == id1 and new_state == PipelineState.APPROVED:
                raise ConcurrentModificationError(f"{id1} claimed by another actor")
            return real_update(pipeline_id, new_state, **kw)

        ops.repo.update_state = lose_the_claim_on_book1

        result = ops.approve(BatchFilter(min_confidence=0.9), actor="human:taylor", execute=True)

        # book2 still got approved; book1 was reported, not fatal.
        assert repo.get(id2)["state"] in (PipelineState.COMPLETE.value, PipelineState.APPROVED.value)
        assert result["approved"] >= 1
        assert any(id1 in str(s) for s in result.get("skipped", [])), result

        # The audit entry must exist despite the contended book.
        assert len(AuditTrail(db_path).query(action="BATCH_APPROVED")) == 1

    def test_reject_skips_a_claimed_book_and_still_audits(self, db_path):
        from agentic_pipeline.db.pipelines import PipelineRepository
        from agentic_pipeline.pipeline.states import PipelineState
        from agentic_pipeline.batch import BatchOperations, BatchFilter
        from agentic_pipeline.audit import AuditTrail

        from agentic_pipeline.db.pipelines import ConcurrentModificationError

        repo = PipelineRepository(db_path)
        id1 = self._pending(repo, "/book1.epub", "hash1", 0.95)
        id2 = self._pending(repo, "/book2.epub", "hash2", 0.92)

        ops = BatchOperations(db_path)
        real_update = ops.repo.update_state

        def lose_the_claim_on_book1(pipeline_id, new_state, **kw):
            if pipeline_id == id1 and new_state == PipelineState.REJECTED:
                raise ConcurrentModificationError(f"{id1} claimed by another actor")
            return real_update(pipeline_id, new_state, **kw)

        ops.repo.update_state = lose_the_claim_on_book1

        result = ops.reject(BatchFilter(min_confidence=0.9), reason="bad", actor="human:taylor", execute=True)

        assert repo.get(id2)["state"] == PipelineState.REJECTED.value
        assert any(id1 in str(s) for s in result.get("skipped", [])), result
        assert len(AuditTrail(db_path).query(action="BATCH_REJECTED")) == 1


def test_batch_approve_dry_run(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.batch import BatchOperations, BatchFilter

    repo = PipelineRepository(db_path)

    id1 = repo.create("/book1.epub", "hash1")
    transition_to(repo, id1, PipelineState.PENDING_APPROVAL)

    ops = BatchOperations(db_path)
    filter = BatchFilter()

    result = ops.approve(filter, actor="human:taylor", execute=False)

    assert result["would_approve"] == 1
    assert result["approved"] == 0

    # State should not change
    pipeline = repo.get(id1)
    assert pipeline["state"] == PipelineState.PENDING_APPROVAL.value


def test_batch_reject(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.batch import BatchOperations, BatchFilter

    repo = PipelineRepository(db_path)

    id1 = repo.create("/book1.epub", "hash1")
    transition_to(repo, id1, PipelineState.PENDING_APPROVAL)
    repo.update_book_profile(id1, {"book_type": "newspaper", "confidence": 0.9})

    ops = BatchOperations(db_path)
    filter = BatchFilter(book_type="newspaper")

    result = ops.reject(filter, reason="Not ingesting periodicals", actor="human:taylor", execute=True)

    assert result["rejected"] == 1

    # Check state changed
    pipeline = repo.get(id1)
    assert pipeline["state"] == PipelineState.REJECTED.value


def test_batch_set_priority(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.batch import BatchOperations, BatchFilter

    repo = PipelineRepository(db_path)

    id1 = repo.create("/book1.epub", "hash1", priority=5)

    ops = BatchOperations(db_path)
    filter = BatchFilter(state="detected")

    result = ops.set_priority(filter, priority=1, actor="human:taylor", execute=True)

    assert result["updated"] == 1

    pipeline = repo.get(id1)
    assert pipeline["priority"] == 1

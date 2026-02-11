"""Approval actions - approve, reject, rollback."""

import logging
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from agentic_pipeline.db.pipelines import PipelineRepository
from agentic_pipeline.pipeline.states import PipelineState

logger = logging.getLogger(__name__)


def _extract_book_id(pipeline: dict) -> Optional[str]:
    """Extract book_id from the processing_result JSON in a pipeline record."""
    processing_result = pipeline.get("processing_result")
    if processing_result and isinstance(processing_result, str):
        try:
            processing_result = json.loads(processing_result)
        except (json.JSONDecodeError, TypeError):
            return None
    if isinstance(processing_result, dict):
        return processing_result.get("book_id")
    return None


def _record_audit(
    db_path: Path,
    pipeline_id: str,
    book_id: Optional[str],
    action: str,
    actor: str,
    reason: Optional[str] = None,
    before_state: Optional[dict] = None,
    after_state: Optional[dict] = None,
    adjustments: Optional[dict] = None,
    confidence: Optional[float] = None,
) -> None:
    """Record an action in the audit trail."""
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO approval_audit
            (book_id, pipeline_id, action, actor, reason, before_state, after_state,
             adjustments, confidence_at_decision, autonomy_mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                book_id or "",
                pipeline_id,
                action,
                actor,
                reason,
                json.dumps(before_state) if before_state else None,
                json.dumps(after_state) if after_state else None,
                json.dumps(adjustments) if adjustments else None,
                confidence,
                "supervised",  # TODO: get from autonomy_config
            )
        )
        conn.commit()
    finally:
        conn.close()


def _complete_approved(db_path: Path, pipeline_id: str, pipeline: dict) -> dict:
    """Run embedding for an approved book and transition to COMPLETE.

    Returns a dict with state and embedding details. On failure, transitions
    to NEEDS_RETRY instead.
    """
    repo = PipelineRepository(db_path)

    # Extract book_id from processing_result (same logic as orchestrator)
    processing_result = pipeline.get("processing_result")
    if processing_result and isinstance(processing_result, str):
        processing_result = json.loads(processing_result)

    book_id = (
        processing_result.get("book_id", pipeline_id)
        if processing_result
        else pipeline_id
    )

    # Transition to EMBEDDING
    repo.update_state(pipeline_id, PipelineState.EMBEDDING)

    try:
        # Lazy import — book_ingestion may not be installed in test envs
        from agentic_pipeline.adapters.processing_adapter import ProcessingAdapter

        adapter = ProcessingAdapter(db_path=db_path)
        result = adapter.generate_embeddings(book_id=book_id)

        if not result.success:
            logger.warning(
                "Embedding failed for %s: %s", pipeline_id, result.error
            )
            repo.update_state(pipeline_id, PipelineState.NEEDS_RETRY)
            return {
                "state": PipelineState.NEEDS_RETRY.value,
                "embedding_error": result.error,
            }

        repo.update_state(pipeline_id, PipelineState.COMPLETE)
        return {
            "state": PipelineState.COMPLETE.value,
            "chapters_embedded": result.chapters_processed,
        }

    except Exception as e:
        logger.error("Embedding exception for %s: %s", pipeline_id, e)
        repo.update_state(pipeline_id, PipelineState.NEEDS_RETRY)
        return {
            "state": PipelineState.NEEDS_RETRY.value,
            "embedding_error": str(e),
        }


def approve_book(
    db_path: Path,
    pipeline_id: str,
    actor: str,
    adjustments: Optional[dict] = None,
) -> dict:
    """Approve a book and generate embeddings inline."""
    repo = PipelineRepository(db_path)
    pipeline = repo.get(pipeline_id)

    if not pipeline:
        return {"success": False, "error": f"Pipeline not found: {pipeline_id}"}

    if pipeline["state"] != PipelineState.PENDING_APPROVAL.value:
        return {"success": False, "error": f"Pipeline not in pending state: {pipeline['state']}"}

    # Get confidence from profile
    profile = json.loads(pipeline.get("book_profile") or "{}")
    confidence = profile.get("confidence")

    before_state = {"state": pipeline["state"]}

    # Transition state, then set approval metadata
    repo.update_state(pipeline_id, PipelineState.APPROVED)
    repo.mark_approved(pipeline_id, approved_by=actor, confidence=confidence)

    # Run embedding inline (APPROVED → EMBEDDING → COMPLETE)
    embed_result = _complete_approved(db_path, pipeline_id, pipeline)

    after_state = {"state": embed_result["state"]}

    # Record audit
    _record_audit(
        db_path,
        pipeline_id,
        _extract_book_id(pipeline),
        action="approved",
        actor=actor,
        before_state=before_state,
        after_state=after_state,
        adjustments=adjustments,
        confidence=confidence,
    )

    return {
        "success": True,
        "pipeline_id": pipeline_id,
        **embed_result,
    }


def reject_book(
    db_path: Path,
    pipeline_id: str,
    reason: str,
    actor: str,
    retry: bool = False,
) -> dict:
    """Reject a book."""
    repo = PipelineRepository(db_path)
    pipeline = repo.get(pipeline_id)

    if not pipeline:
        return {"success": False, "error": f"Pipeline not found: {pipeline_id}"}

    before_state = {"state": pipeline["state"]}

    if retry:
        new_state = PipelineState.NEEDS_RETRY
    else:
        new_state = PipelineState.REJECTED

    repo.update_state(pipeline_id, new_state)

    after_state = {"state": new_state.value}

    # Record audit
    _record_audit(
        db_path,
        pipeline_id,
        _extract_book_id(pipeline),
        action="rejected",
        actor=actor,
        reason=reason,
        before_state=before_state,
        after_state=after_state,
    )

    return {
        "success": True,
        "pipeline_id": pipeline_id,
        "state": new_state.value,
        "retry_queued": retry,
    }


def rollback_book(
    db_path: Path,
    pipeline_id: str,
    reason: str,
    actor: str,
) -> dict:
    """Rollback an approved/completed book."""
    repo = PipelineRepository(db_path)
    pipeline = repo.get(pipeline_id)

    if not pipeline:
        return {"success": False, "error": f"Pipeline not found: {pipeline_id}"}

    before_state = {"state": pipeline["state"]}

    repo.update_state(pipeline_id, PipelineState.ARCHIVED)

    after_state = {"state": PipelineState.ARCHIVED.value}

    # Record audit
    _record_audit(
        db_path,
        pipeline_id,
        _extract_book_id(pipeline),
        action="rollback",
        actor=actor,
        reason=reason,
        before_state=before_state,
        after_state=after_state,
    )

    return {
        "success": True,
        "pipeline_id": pipeline_id,
        "state": PipelineState.ARCHIVED.value,
        "reason": reason,
    }

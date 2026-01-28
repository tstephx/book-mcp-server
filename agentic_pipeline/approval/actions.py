"""Approval actions - approve, reject, rollback."""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from agentic_pipeline.db.pipelines import PipelineRepository
from agentic_pipeline.pipeline.states import PipelineState


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
    conn = sqlite3.connect(db_path)
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
    conn.close()


def approve_book(
    db_path: Path,
    pipeline_id: str,
    actor: str,
    adjustments: Optional[dict] = None,
) -> dict:
    """Approve a book for ingestion."""
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

    # Mark as approved
    repo.mark_approved(pipeline_id, approved_by=actor, confidence=confidence)

    after_state = {"state": PipelineState.APPROVED.value}

    # Record audit
    _record_audit(
        db_path,
        pipeline_id,
        pipeline.get("book_id"),
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
        "state": PipelineState.APPROVED.value,
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
        pipeline.get("book_id"),
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
        pipeline.get("book_id"),
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

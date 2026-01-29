"""MCP server with approval tools."""

from pathlib import Path
from typing import Optional

from agentic_pipeline.approval.queue import ApprovalQueue
from agentic_pipeline.approval.actions import approve_book, reject_book, rollback_book
from agentic_pipeline.db.config import get_db_path


def review_pending_books(db_path: Optional[str] = None, sort_by: str = "priority") -> dict:
    """
    Get all books pending approval.

    Returns queue of books awaiting review with stats.
    """
    path = Path(db_path) if db_path else get_db_path()
    queue = ApprovalQueue(path)
    return queue.get_pending(sort_by=sort_by)


def approve_book_tool(
    db_path: Optional[str] = None,
    pipeline_id: str = "",
    actor: str = "human:unknown",
    adjustments: Optional[dict] = None,
) -> dict:
    """
    Approve a book for ingestion into the library.

    Args:
        pipeline_id: The pipeline ID to approve
        actor: Who is approving (e.g., "human:taylor", "auto:confident")
        adjustments: Optional adjustments to apply before ingestion
    """
    path = Path(db_path) if db_path else get_db_path()
    return approve_book(path, pipeline_id, actor, adjustments)


def reject_book_tool(
    db_path: Optional[str] = None,
    pipeline_id: str = "",
    reason: str = "",
    actor: str = "human:unknown",
    retry: bool = False,
) -> dict:
    """
    Reject a book.

    Args:
        pipeline_id: The pipeline ID to reject
        reason: Why the book is being rejected
        actor: Who is rejecting
        retry: If True, queue for retry with adjustments instead of permanent rejection
    """
    path = Path(db_path) if db_path else get_db_path()
    return reject_book(path, pipeline_id, reason, actor, retry)


def rollback_book_tool(
    db_path: Optional[str] = None,
    pipeline_id: str = "",
    reason: str = "",
    actor: str = "human:unknown",
) -> dict:
    """
    Rollback an approved/completed book from the library.

    Args:
        pipeline_id: The pipeline ID to rollback
        reason: Why the book is being rolled back
        actor: Who is performing the rollback
    """
    path = Path(db_path) if db_path else get_db_path()
    return rollback_book(path, pipeline_id, reason, actor)


def process_book(path: str) -> dict:
    """
    Process a book through the pipeline.

    Args:
        path: Path to the book file (epub, pdf, etc.)

    Returns:
        Processing result with pipeline_id, state, book_type, and confidence
    """
    from agentic_pipeline.config import OrchestratorConfig
    from agentic_pipeline.orchestrator import Orchestrator

    config = OrchestratorConfig.from_env()
    orchestrator = Orchestrator(config)

    result = orchestrator.process_one(path)
    return result


def get_pipeline_status(pipeline_id: str) -> dict:
    """
    Get the status of a pipeline run.

    Args:
        pipeline_id: The pipeline ID to check

    Returns:
        Pipeline details including state, book_type, confidence, retries
    """
    import json
    from agentic_pipeline.db.pipelines import PipelineRepository

    db_path = get_db_path()
    repo = PipelineRepository(db_path)

    pipeline = repo.get(pipeline_id)

    if not pipeline:
        return {"error": f"Pipeline not found: {pipeline_id}"}

    result = {
        "pipeline_id": pipeline_id,
        "state": pipeline["state"],
        "source_path": pipeline["source_path"],
        "retry_count": pipeline.get("retry_count", 0),
        "created_at": pipeline.get("created_at"),
    }

    if pipeline.get("book_profile"):
        profile = json.loads(pipeline["book_profile"]) if isinstance(pipeline["book_profile"], str) else pipeline["book_profile"]
        result["book_type"] = profile.get("book_type")
        result["confidence"] = profile.get("confidence")
        result["suggested_tags"] = profile.get("suggested_tags")

    if pipeline.get("approved_by"):
        result["approved_by"] = pipeline["approved_by"]

    return result


# Phase 4: Production Hardening Tools

def get_pipeline_health() -> dict:
    """
    Get current pipeline health status.

    Returns health metrics including active, queued, stuck counts and alerts.
    """
    from agentic_pipeline.health import HealthMonitor, StuckDetector

    db_path = get_db_path()
    monitor = HealthMonitor(db_path)
    detector = StuckDetector(db_path)

    report = monitor.get_health()
    report["stuck"] = detector.detect()

    return report


def get_stuck_pipelines() -> list[dict]:
    """
    Get list of pipelines that appear to be stuck.

    Returns pipelines that have been in the same state longer than expected.
    """
    from agentic_pipeline.health import StuckDetector

    db_path = get_db_path()
    detector = StuckDetector(db_path)

    return detector.detect()


def batch_approve_tool(
    min_confidence: float = None,
    book_type: str = None,
    max_count: int = 50,
    execute: bool = False,
) -> dict:
    """
    Approve books matching filters.

    Args:
        min_confidence: Minimum confidence threshold
        book_type: Filter by book type
        max_count: Maximum books to approve
        execute: Set True to actually approve (otherwise preview)

    Returns:
        Count of approved/would_approve books and list of affected books.
    """
    from agentic_pipeline.batch import BatchOperations, BatchFilter

    db_path = get_db_path()
    ops = BatchOperations(db_path)
    filter = BatchFilter(
        min_confidence=min_confidence,
        book_type=book_type,
        max_count=max_count,
    )

    return ops.approve(filter, actor="mcp:batch", execute=execute)


def batch_reject_tool(
    book_type: str = None,
    max_confidence: float = None,
    reason: str = "",
    max_count: int = 50,
    execute: bool = False,
) -> dict:
    """
    Reject books matching filters.

    Args:
        book_type: Filter by book type
        max_confidence: Maximum confidence threshold
        reason: Rejection reason (required for execute)
        max_count: Maximum books to reject
        execute: Set True to actually reject (otherwise preview)
    """
    from agentic_pipeline.batch import BatchOperations, BatchFilter

    db_path = get_db_path()
    ops = BatchOperations(db_path)
    filter = BatchFilter(
        book_type=book_type,
        max_confidence=max_confidence,
        max_count=max_count,
    )

    return ops.reject(filter, reason=reason, actor="mcp:batch", execute=execute)


def get_audit_log(
    book_id: str = None,
    actor: str = None,
    action: str = None,
    last_days: int = 7,
    limit: int = 100,
) -> list[dict]:
    """
    Query the audit trail.

    Args:
        book_id: Filter by book ID
        actor: Filter by actor
        action: Filter by action type
        last_days: Only return entries from last N days
        limit: Maximum entries to return
    """
    from agentic_pipeline.audit import AuditTrail

    db_path = get_db_path()
    trail = AuditTrail(db_path)

    return trail.query(
        book_id=book_id,
        actor=actor,
        action=action,
        last_days=last_days,
        limit=limit,
    )

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

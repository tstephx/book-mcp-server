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

    Returns health metrics including active, queued, stuck counts,
    alerts, and number of untracked library books.
    """
    from agentic_pipeline.health import HealthMonitor, StuckDetector
    from agentic_pipeline.backfill import BackfillManager

    db_path = get_db_path()
    monitor = HealthMonitor(db_path)
    detector = StuckDetector(db_path)

    report = monitor.get_health()
    report["stuck"] = detector.detect()

    # Include untracked books count
    try:
        manager = BackfillManager(db_path)
        untracked = manager.find_untracked()
        report["untracked_books"] = len(untracked)
    except Exception:
        report["untracked_books"] = None

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


# Phase 5: Autonomy Tools

def get_autonomy_status() -> dict:
    """
    Get current autonomy mode, thresholds, and metrics summary.
    """
    from agentic_pipeline.autonomy import AutonomyConfig, MetricsCollector

    db_path = get_db_path()
    config = AutonomyConfig(db_path)
    collector = MetricsCollector(db_path)

    mode = config.get_mode()
    metrics = collector.get_metrics(days=30)

    return {
        "mode": mode,
        "escape_hatch_active": config.is_escape_hatch_active(),
        "metrics_30d": metrics,
    }


def set_autonomy_mode(mode: str) -> dict:
    """
    Change autonomy mode.

    Args:
        mode: One of "supervised", "partial", "confident"
    """
    from agentic_pipeline.autonomy import AutonomyConfig

    db_path = get_db_path()
    config = AutonomyConfig(db_path)

    if config.is_escape_hatch_active():
        return {"error": "Cannot change mode while escape hatch is active"}

    config.set_mode(mode)
    return {"mode": mode, "success": True}


def activate_escape_hatch_tool(reason: str) -> dict:
    """
    Immediately revert to supervised mode.

    Args:
        reason: Why the escape hatch is being activated
    """
    from agentic_pipeline.autonomy import AutonomyConfig

    db_path = get_db_path()
    config = AutonomyConfig(db_path)
    config.activate_escape_hatch(reason)

    return {
        "success": True,
        "message": "Escape hatch activated. All books now require human review.",
        "reason": reason,
    }


def get_autonomy_readiness() -> dict:
    """
    Check if the system is ready to advance to the next autonomy mode.
    """
    from agentic_pipeline.autonomy import AutonomyConfig, MetricsCollector, CalibrationEngine

    db_path = get_db_path()
    config = AutonomyConfig(db_path)
    collector = MetricsCollector(db_path)
    engine = CalibrationEngine(db_path)

    mode = config.get_mode()
    metrics = collector.get_metrics(days=90)

    # Calculate override rate
    total = metrics["total_processed"]
    overrides = metrics["human_rejected"] + metrics["human_adjusted"]
    override_rate = overrides / total if total > 0 else None

    # Get thresholds
    thresholds = engine.update_thresholds()

    ready_for_partial = total >= 100 and (override_rate or 1) < 0.15
    ready_for_confident = total >= 500 and (override_rate or 1) < 0.05

    return {
        "current_mode": mode,
        "total_processed": total,
        "override_rate": override_rate,
        "thresholds": thresholds,
        "ready_for_partial": ready_for_partial,
        "ready_for_confident": ready_for_confident,
        "recommendation": "confident" if ready_for_confident else ("partial" if ready_for_partial else "supervised"),
    }


# Phase 6: Backfill & Validation Tools

def backfill_library(
    db_path: Optional[str] = None,
    dry_run: bool = True,
) -> dict:
    """
    Register legacy library books in the pipeline.

    Creates pipeline records for books ingested via the raw CLI that have
    no audit trail. Safe and non-destructive.

    Args:
        dry_run: If True (default), preview what would be backfilled without changes.
                 Set to False to actually create pipeline records.

    Returns:
        Dict with backfilled/would_backfill count and per-book details.
    """
    from agentic_pipeline.backfill import BackfillManager

    path = Path(db_path) if db_path else get_db_path()
    manager = BackfillManager(path)
    return manager.run(dry_run=dry_run)


def validate_library(db_path: Optional[str] = None) -> dict:
    """
    Check all library books for quality issues.

    Scans for: missing chapters, missing embeddings, low word count.

    Returns:
        Dict with issue_count and list of issues per book.
    """
    from agentic_pipeline.backfill import LibraryValidator

    path = Path(db_path) if db_path else get_db_path()
    validator = LibraryValidator(path)
    issues = validator.validate()
    return {
        "issue_count": len(issues),
        "issues": issues,
    }


def reingest_book_tool(
    db_path: Optional[str] = None,
    book_id: str = "",
) -> dict:
    """
    Reprocess a book through the full pipeline.

    Archives the existing pipeline record and creates a new one.
    The book goes through classification, processing, validation,
    and approval again. Requires the original source file to exist.

    Args:
        book_id: The book/pipeline ID to reingest.

    Returns:
        Dict with new pipeline_id and resulting state.
    """
    from agentic_pipeline.db.pipelines import PipelineRepository

    path = Path(db_path) if db_path else get_db_path()
    repo = PipelineRepository(path)

    record = repo.get(book_id)
    if not record:
        return {"error": f"Pipeline record not found: {book_id}"}

    source_path = record["source_path"]
    if not source_path or not Path(source_path).exists():
        return {"error": f"Source file not found: {source_path}"}

    new_pid = repo.prepare_reingest(book_id)

    from agentic_pipeline.config import OrchestratorConfig
    from agentic_pipeline.orchestrator import Orchestrator

    config = OrchestratorConfig.from_env()
    orchestrator = Orchestrator(config)
    result = orchestrator._process_book(new_pid, source_path, record["content_hash"])
    result["new_pipeline_id"] = new_pid
    result["old_pipeline_id"] = book_id
    return result

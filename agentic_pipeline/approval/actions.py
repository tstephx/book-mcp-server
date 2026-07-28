"""Approval actions - approve, reject, rollback."""

import hashlib
import logging
import json
import os
import shutil
import threading
from pathlib import Path
from typing import Optional

from agentic_pipeline.autonomy.config import AutonomyConfig
from agentic_pipeline.autonomy.metrics import MetricsCollector
from agentic_pipeline.db.connection import get_pipeline_db
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
    connection=None,
) -> None:
    """Record an action in the audit trail."""

    def insert(conn) -> None:
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
                AutonomyConfig(db_path).get_mode(),
            ),
        )

    if connection is not None:
        insert(connection)
        return

    with get_pipeline_db(str(db_path)) as conn:
        insert(conn)
        conn.commit()


def _get_processed_dir() -> Optional[Path]:
    """Get processed directory from environment."""
    processed_dir_str = os.environ.get("PROCESSED_DIR")
    return Path(processed_dir_str).resolve() if processed_dir_str else None


def _hash_file(path: Path) -> str:
    """SHA-256 of file contents. Mirrors Orchestrator._compute_hash."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_source_file(source_path: Optional[str], expected_hash: Optional[str] = None) -> Optional[Path]:
    """Locate a book's source file, following it into the archive if it moved.

    Records written before archiving tracked the move still name the original
    watch-dir path, which auto-archive emptied. We can look for the same
    basename under processed_dir — but that is a *guess*, not a lookup:
    _archive_source_file() renames collisions to stem_1.epub, so two books can
    share an original filename and the basename will resolve to whichever one
    got archived first. Handing back the wrong book is worse than finding
    nothing: reingest wipes the book's chapters and refills them from the file
    we return.

    So the fallback is only taken when expected_hash is supplied and the
    candidate's contents match it. Without a hash we cannot tell the right book
    from a namesake, so we fail closed and return None.

    Args:
        expected_hash: the record's content_hash. Required to use the archive
            fallback; ignored when the file is still at source_path, which is a
            recorded location rather than a guess.

    Returns:
        The file, or None if it cannot be found or cannot be proven to be the
        right one.
    """
    if not source_path:
        return None

    src = Path(source_path)
    if src.exists():
        return src

    processed_dir = _get_processed_dir()
    if not processed_dir:
        return None

    archived = Path(processed_dir) / src.name
    if not archived.exists():
        return None

    if not expected_hash:
        logger.warning(
            "Found %s in the archive but no expected hash was given; refusing to "
            "assume it is the right book (collisions share basenames)",
            src.name,
        )
        return None

    try:
        actual = _hash_file(archived)
    except OSError as e:
        logger.warning("Could not hash archived candidate %s: %s", archived.name, e)
        return None

    if actual != expected_hash:
        logger.warning(
            "Archived %s is a different book (hash %s != %s) — most likely a "
            "filename collision renamed the real file to stem_N",
            src.name,
            actual[:12],
            expected_hash[:12],
        )
        return None

    return archived


def _archive_source_file(
    source_path: str,
    processed_dir: Optional[Path] = None,
) -> Optional[str]:
    """Move source file to processed directory after successful completion.

    Args:
        source_path: Path to the source file to archive.
        processed_dir: Target directory. Falls back to PROCESSED_DIR env var.

    Returns:
        Destination filename if archived, None otherwise.
    """
    if processed_dir is None:
        processed_dir = _get_processed_dir()
    if not processed_dir:
        return None

    src = Path(source_path)
    if not src.exists():
        logger.info("Archive skipped, file missing: %s", src.name)
        return None

    processed_dir.mkdir(parents=True, exist_ok=True)
    dest = processed_dir / src.name

    # Handle name collision
    if dest.exists():
        stem, suffix = dest.stem, dest.suffix
        counter = 1
        while dest.exists():
            dest = processed_dir / f"{stem}_{counter}{suffix}"
            counter += 1

    try:
        shutil.move(str(src), str(dest))
        logger.info("Archived %s to %s", src.name, dest.name)
        return dest.name
    except OSError as e:
        logger.error("Failed to archive %s: %s", src.name, e)
        return None


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

    book_id = processing_result.get("book_id", pipeline_id) if processing_result else pipeline_id

    # Transition to EMBEDDING
    repo.update_state(pipeline_id, PipelineState.EMBEDDING)

    try:
        # Lazy import — book_ingestion may not be installed in test envs
        from agentic_pipeline.adapters.processing_adapter import ProcessingAdapter

        adapter = ProcessingAdapter(db_path=db_path)
        result = adapter.generate_embeddings(book_id=book_id)

        if not result.success:
            logger.warning("Embedding failed for %s: %s", pipeline_id, result.error)
            repo.update_state(
                pipeline_id,
                PipelineState.NEEDS_RETRY,
                error_details={"embedding_error": result.error, "failed_in": "embedding"},
            )
            return {
                "state": PipelineState.NEEDS_RETRY.value,
                "embedding_error": result.error,
            }

        repo.update_state(pipeline_id, PipelineState.COMPLETE)

        # Archive source file if configured, and follow it in the record —
        # otherwise source_path names a file that is no longer there and
        # reingest refuses to run.
        source_path = pipeline.get("source_path")
        if source_path:
            archived_name = _archive_source_file(source_path)
            processed_dir = _get_processed_dir()
            if archived_name and processed_dir:
                # archived_name, not the original: collisions rename to stem_1.
                repo.update_source_path(pipeline_id, str(Path(processed_dir) / archived_name))

        return {
            "state": PipelineState.COMPLETE.value,
            "chapters_embedded": result.chapters_processed,
        }

    except Exception as e:
        logger.error("Embedding exception for %s: %s", pipeline_id, e)
        repo.update_state(
            pipeline_id,
            PipelineState.NEEDS_RETRY,
            error_details={"embedding_error": str(e), "failed_in": "embedding"},
        )
        return {
            "state": PipelineState.NEEDS_RETRY.value,
            "embedding_error": str(e),
        }


def _run_embedding_background(db_path: Path, pipeline_id: str, pipeline: dict) -> None:
    """Run embedding in a daemon thread — called by approve_book() fire-and-forget."""
    _complete_approved(db_path, pipeline_id, pipeline)


def approve_book(
    db_path: Path,
    pipeline_id: str,
    actor: str,
    adjustments: Optional[dict] = None,
    background: bool = True,
    confidence: Optional[float] = None,
    decision_metadata: Optional[dict] = None,
) -> dict:
    """Approve a book, driving PENDING_APPROVAL → APPROVED → EMBEDDING → COMPLETE.

    Args:
        background: When True (default), embedding runs on a daemon thread and
            this returns as soon as the book is APPROVED — right for the MCP
            server, which outlives the thread. Callers that exit on return (the
            CLI) MUST pass False: a daemon thread dies with the process, leaving
            the book APPROVED with no chunks and no error reported.

    Returns:
        With background=True: state=APPROVED, embedding="queued".
        With background=False: the embedding outcome — state=COMPLETE with
        chapters_embedded, or state=NEEDS_RETRY with embedding_error.
    """
    repo = PipelineRepository(db_path)
    pipeline = repo.get(pipeline_id)

    if not pipeline:
        return {"success": False, "error": f"Pipeline not found: {pipeline_id}"}

    if pipeline["state"] != PipelineState.PENDING_APPROVAL.value:
        return {"success": False, "error": f"Pipeline not in pending state: {pipeline['state']}"}

    # Get confidence from profile
    profile = json.loads(pipeline.get("book_profile") or "{}")
    approval_confidence = confidence if confidence is not None else profile.get("confidence")

    if actor.startswith("auto:"):
        processing_result = pipeline.get("processing_result")
        if isinstance(processing_result, str):
            try:
                processing_result = json.loads(processing_result)
            except (json.JSONDecodeError, TypeError):
                processing_result = {}
        if not isinstance(processing_result, dict):
            processing_result = {}

        validation_result = pipeline.get("validation_result")
        if isinstance(validation_result, str):
            try:
                validation_result = json.loads(validation_result)
            except (json.JSONDecodeError, TypeError):
                validation_result = {}
        if not isinstance(validation_result, dict):
            validation_result = {}

        try:
            fresh_decision = AutonomyConfig(db_path).evaluate_auto_approval(
                profile.get("book_type", "unknown"),
                approval_confidence,
                validation_passed=validation_result.get("passed") is True,
                needs_review=processing_result.get("needs_review", True),
            )
        except Exception as exc:
            logger.error(
                "Auto-approval policy recheck failed for %s: %s",
                pipeline_id,
                exc,
            )
            return {
                "success": False,
                "pipeline_id": pipeline_id,
                "state": PipelineState.PENDING_APPROVAL.value,
                "error": "auto_approval_denied",
                "approval_reason": "policy_error",
                "autonomy_mode": "supervised",
                "approval_threshold": None,
            }
        if not fresh_decision.should_auto_approve:
            return {
                "success": False,
                "pipeline_id": pipeline_id,
                "state": PipelineState.PENDING_APPROVAL.value,
                "error": "auto_approval_denied",
                "approval_reason": fresh_decision.reason,
                "autonomy_mode": fresh_decision.mode,
                "approval_threshold": fresh_decision.threshold,
            }
        actor = f"auto:{fresh_decision.mode}"
        decision_metadata = {
            "mode": fresh_decision.mode,
            "reason": fresh_decision.reason,
            "threshold": fresh_decision.threshold,
        }

    audit_adjustments = dict(adjustments or {})
    if decision_metadata:
        audit_adjustments["autonomy_decision"] = decision_metadata

    before_state = {"state": pipeline["state"]}
    after_state = {"state": PipelineState.APPROVED.value}

    # Transition to APPROVED immediately
    repo.update_state(pipeline_id, PipelineState.APPROVED)
    repo.mark_approved(pipeline_id, approved_by=actor, confidence=approval_confidence)

    # Governance records must exist before embedding can publish the book.
    try:
        with get_pipeline_db(str(db_path)) as governance_conn:
            _record_audit(
                db_path,
                pipeline_id,
                _extract_book_id(pipeline),
                action="approved",
                actor=actor,
                before_state=before_state,
                after_state=after_state,
                adjustments=audit_adjustments or None,
                confidence=approval_confidence,
                connection=governance_conn,
            )
            MetricsCollector(db_path).record_decision(
                book_id=_extract_book_id(pipeline) or pipeline_id,
                pipeline_id=pipeline_id,
                book_type=profile.get("book_type", "unknown"),
                confidence=approval_confidence if approval_confidence is not None else 0.0,
                decision="approved",
                actor=actor,
                adjustments=audit_adjustments or None,
                connection=governance_conn,
            )
            governance_conn.commit()
    except Exception as exc:
        logger.error("Approval governance recording failed for %s: %s", pipeline_id, exc)
        repo.update_state(
            pipeline_id,
            PipelineState.NEEDS_RETRY,
            error_details={
                "failed_in": "approval_governance",
                "approval_governance_error": str(exc),
            },
        )
        return {
            "success": False,
            "pipeline_id": pipeline_id,
            "state": PipelineState.NEEDS_RETRY.value,
            "error": "approval_governance_failed",
        }

    if not background:
        # Caller exits on return — embed here or the work never happens.
        return {"success": True, "pipeline_id": pipeline_id, **_complete_approved(db_path, pipeline_id, pipeline)}

    # Fire-and-forget: embedding runs in background, MCP caller is not blocked
    thread = threading.Thread(
        target=_run_embedding_background,
        args=(db_path, pipeline_id, pipeline),
        daemon=True,
        name=f"embed-{pipeline_id[:8]}",
    )
    thread.start()

    return {
        "success": True,
        "pipeline_id": pipeline_id,
        "state": PipelineState.APPROVED.value,
        "embedding": "queued",
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

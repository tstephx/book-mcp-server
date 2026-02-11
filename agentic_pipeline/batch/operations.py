"""Batch operations for processing multiple books."""

from pathlib import Path

from agentic_pipeline.audit import AuditTrail
from agentic_pipeline.batch.filters import BatchFilter
from agentic_pipeline.db.pipelines import PipelineRepository
from agentic_pipeline.pipeline.states import PipelineState


class BatchOperations:
    """Execute operations on multiple pipelines."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.repo = PipelineRepository(db_path)
        self.audit = AuditTrail(db_path)

    def approve(
        self,
        filter: BatchFilter,
        actor: str,
        execute: bool = False,
    ) -> dict:
        """Approve books matching filter."""
        # Override filter to only match pending_approval
        filter.state = "pending_approval"
        matches = filter.apply(self.db_path)

        if not execute:
            return {
                "would_approve": len(matches),
                "approved": 0,
                "books": [{"id": m["id"], "source_path": m["source_path"]} for m in matches]
            }

        from agentic_pipeline.approval.actions import _complete_approved

        embedded = 0
        embedding_failures = []

        for pipeline in matches:
            self.repo.update_state(pipeline["id"], PipelineState.APPROVED)
            self.repo.mark_approved(
                pipeline["id"],
                approved_by=f"batch:{actor}",
                confidence=None
            )
            embed_result = _complete_approved(self.db_path, pipeline["id"], pipeline)
            if embed_result["state"] == PipelineState.COMPLETE.value:
                embedded += 1
            else:
                embedding_failures.append({
                    "id": pipeline["id"],
                    "error": embed_result.get("embedding_error", "unknown"),
                })

        # Log batch operation to audit
        self.audit.log(
            book_id=f"batch:{len(matches)}_books",
            action="BATCH_APPROVED",
            actor=actor,
            filter_used=filter.to_dict(),
        )

        return {
            "approved": len(matches),
            "would_approve": len(matches),
            "embedded": embedded,
            "embedding_failures": embedding_failures,
            "books": [{"id": m["id"], "source_path": m["source_path"]} for m in matches]
        }

    def reject(
        self,
        filter: BatchFilter,
        reason: str,
        actor: str,
        execute: bool = False,
    ) -> dict:
        """Reject books matching filter."""
        filter.state = "pending_approval"
        matches = filter.apply(self.db_path)

        if not execute:
            return {
                "would_reject": len(matches),
                "rejected": 0,
                "books": [{"id": m["id"], "source_path": m["source_path"]} for m in matches]
            }

        for pipeline in matches:
            self.repo.update_state(
                pipeline["id"],
                PipelineState.REJECTED,
                error_details={"reason": reason, "actor": actor}
            )

        self.audit.log(
            book_id=f"batch:{len(matches)}_books",
            action="BATCH_REJECTED",
            actor=actor,
            reason=reason,
            filter_used=filter.to_dict(),
        )

        return {
            "rejected": len(matches),
            "would_reject": len(matches),
            "books": [{"id": m["id"], "source_path": m["source_path"]} for m in matches]
        }

    def set_priority(
        self,
        filter: BatchFilter,
        priority: int,
        actor: str,
        execute: bool = False,
    ) -> dict:
        """Set priority for books matching filter."""
        matches = filter.apply(self.db_path)

        if not execute:
            return {
                "would_update": len(matches),
                "updated": 0,
                "books": [{"id": m["id"], "source_path": m["source_path"]} for m in matches]
            }

        for pipeline in matches:
            self.repo.update_priority(pipeline["id"], priority)

        self.audit.log(
            book_id=f"batch:{len(matches)}_books",
            action="BATCH_PRIORITY_CHANGED",
            actor=actor,
            filter_used=filter.to_dict(),
            adjustments={"new_priority": priority}
        )

        return {
            "updated": len(matches),
            "would_update": len(matches),
            "books": [{"id": m["id"], "source_path": m["source_path"]} for m in matches]
        }

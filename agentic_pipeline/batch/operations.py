"""Batch operations for processing multiple books."""

from pathlib import Path

from agentic_pipeline.audit import AuditTrail
from agentic_pipeline.batch.filters import BatchFilter
from agentic_pipeline.db.pipelines import ConcurrentModificationError, PipelineRepository
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
                "books": [{"id": m["id"], "source_path": m["source_path"]} for m in matches],
            }

        from agentic_pipeline.approval.actions import _complete_approved

        embedded = 0
        embedding_failures = []
        skipped = []
        approved = []

        for pipeline in matches:
            try:
                self.repo.update_state(pipeline["id"], PipelineState.APPROVED)
            except ConcurrentModificationError as e:
                # Claimed between filter.apply() and now. One contended book must
                # not abort the batch, nor cost us the audit entry below.
                skipped.append({"id": pipeline["id"], "reason": str(e)})
                continue

            approved.append(pipeline)
            self.repo.mark_approved(pipeline["id"], approved_by=f"batch:{actor}", confidence=None)
            embed_result = _complete_approved(self.db_path, pipeline["id"], pipeline)
            if embed_result["state"] == PipelineState.COMPLETE.value:
                embedded += 1
            else:
                embedding_failures.append(
                    {
                        "id": pipeline["id"],
                        "error": embed_result.get("embedding_error", "unknown"),
                    }
                )

        # Log batch operation to audit — always, even on a partial run.
        self.audit.log(
            book_id=f"batch:{len(approved)}_books",
            action="BATCH_APPROVED",
            actor=actor,
            filter_used=filter.to_dict(),
        )

        return {
            "approved": len(approved),
            "would_approve": len(matches),
            "embedded": embedded,
            "embedding_failures": embedding_failures,
            "skipped": skipped,
            "books": [{"id": m["id"], "source_path": m["source_path"]} for m in approved],
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
                "books": [{"id": m["id"], "source_path": m["source_path"]} for m in matches],
            }

        rejected = []
        skipped = []

        for pipeline in matches:
            try:
                self.repo.update_state(
                    pipeline["id"], PipelineState.REJECTED, error_details={"reason": reason, "actor": actor}
                )
            except ConcurrentModificationError as e:
                skipped.append({"id": pipeline["id"], "reason": str(e)})
                continue
            rejected.append(pipeline)

        # Log batch operation to audit — always, even on a partial run.
        self.audit.log(
            book_id=f"batch:{len(rejected)}_books",
            action="BATCH_REJECTED",
            actor=actor,
            reason=reason,
            filter_used=filter.to_dict(),
        )

        return {
            "rejected": len(rejected),
            "would_reject": len(matches),
            "skipped": skipped,
            "books": [{"id": m["id"], "source_path": m["source_path"]} for m in rejected],
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
                "books": [{"id": m["id"], "source_path": m["source_path"]} for m in matches],
            }

        for pipeline in matches:
            self.repo.update_priority(pipeline["id"], priority)

        self.audit.log(
            book_id=f"batch:{len(matches)}_books",
            action="BATCH_PRIORITY_CHANGED",
            actor=actor,
            filter_used=filter.to_dict(),
            adjustments={"new_priority": priority},
        )

        return {
            "updated": len(matches),
            "would_update": len(matches),
            "books": [{"id": m["id"], "source_path": m["source_path"]} for m in matches],
        }

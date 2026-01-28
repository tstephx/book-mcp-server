"""Approval queue management."""

import json
from pathlib import Path
from typing import Optional

from agentic_pipeline.db.pipelines import PipelineRepository
from agentic_pipeline.pipeline.states import PipelineState


class ApprovalQueue:
    """Manages the queue of books pending approval."""

    def __init__(self, db_path: Path):
        self.repo = PipelineRepository(db_path)

    def get_pending(self, sort_by: str = "priority") -> dict:
        """Get all books pending approval with stats."""
        pipelines = self.repo.list_pending_approval()

        # Calculate stats
        high_confidence = 0
        needs_attention = 0
        total_confidence = 0

        books = []
        for p in pipelines:
            profile = json.loads(p.get("book_profile") or "{}")
            confidence = profile.get("confidence", 0)

            if confidence >= 0.9:
                high_confidence += 1
            elif confidence < 0.8:
                needs_attention += 1

            total_confidence += confidence

            books.append({
                "id": p["id"],
                "source_path": p["source_path"],
                "content_hash": p["content_hash"],
                "book_type": profile.get("book_type", "unknown"),
                "confidence": confidence,
                "suggested_tags": profile.get("suggested_tags", []),
                "created_at": p["created_at"],
                "priority": p["priority"],
            })

        avg_confidence = total_confidence / len(pipelines) if pipelines else 0

        return {
            "pending_count": len(pipelines),
            "stats": {
                "avg_confidence": round(avg_confidence, 2),
                "high_confidence": high_confidence,
                "needs_attention": needs_attention,
            },
            "books": books,
        }

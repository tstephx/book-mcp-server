"""Autonomy metrics collection."""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

from agentic_pipeline.db.connection import get_pipeline_db


class MetricsCollector:
    """Collects and aggregates autonomy metrics."""

    def __init__(self, db_path: Path):
        self.db_path = str(db_path)

    def record_decision(
        self,
        book_id: str,
        book_type: str,
        confidence: float,
        decision: str,
        actor: str,
        pipeline_id: str = None,
        adjustments: dict = None,
    ) -> None:
        """Record a decision for metrics tracking."""
        with get_pipeline_db(self.db_path) as conn:
            cursor = conn.cursor()

            # Determine original decision type
            original_decision = "auto_approved" if actor.startswith("auto:") else "human_review"

            cursor.execute("""
                INSERT INTO autonomy_feedback
                (book_id, pipeline_id, original_decision, original_confidence, original_book_type,
                 human_decision, human_adjustments, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                book_id,
                pipeline_id,
                original_decision,
                confidence,
                book_type,
                decision,
                json.dumps(adjustments) if adjustments else None,
                datetime.now(timezone.utc).isoformat()
            ))

            conn.commit()

    def get_metrics(self, days: int = 30) -> dict:
        """Get aggregated metrics for the specified period."""
        with get_pipeline_db(self.db_path) as conn:
            cursor = conn.cursor()

            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN original_decision = 'auto_approved' AND human_decision = 'approved' THEN 1 ELSE 0 END) as auto_approved,
                    SUM(CASE WHEN original_decision = 'human_review' AND human_decision = 'approved' THEN 1 ELSE 0 END) as human_approved,
                    SUM(CASE WHEN human_decision = 'rejected' THEN 1 ELSE 0 END) as human_rejected,
                    SUM(CASE WHEN human_adjustments IS NOT NULL THEN 1 ELSE 0 END) as human_adjusted,
                    AVG(CASE WHEN original_decision = 'auto_approved' THEN original_confidence END) as avg_conf_auto,
                    AVG(CASE WHEN original_decision = 'human_review' THEN original_confidence END) as avg_conf_human
                FROM autonomy_feedback
                WHERE created_at > ?
            """, (cutoff,))

            row = cursor.fetchone()

            return {
                "total_processed": row["total"] or 0,
                "auto_approved": row["auto_approved"] or 0,
                "human_approved": row["human_approved"] or 0,
                "human_rejected": row["human_rejected"] or 0,
                "human_adjusted": row["human_adjusted"] or 0,
                "avg_confidence_auto": row["avg_conf_auto"],
                "avg_confidence_human": row["avg_conf_human"],
            }

    def get_accuracy_by_type(self, book_type: str, days: int = 90) -> dict:
        """Get accuracy metrics for a specific book type."""
        with get_pipeline_db(self.db_path) as conn:
            cursor = conn.cursor()

            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN human_decision = 'approved' THEN 1 ELSE 0 END) as correct
                FROM autonomy_feedback
                WHERE original_book_type = ?
                AND created_at > ?
            """, (book_type, cutoff))

            row = cursor.fetchone()

            total = row["total"] or 0
            correct = row["correct"] or 0

            return {
                "book_type": book_type,
                "sample_count": total,
                "accuracy": correct / total if total > 0 else None,
            }

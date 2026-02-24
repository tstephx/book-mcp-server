"""Spot-check system for ongoing verification."""

import random
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from agentic_pipeline.db.connection import get_pipeline_db


class SpotCheckManager:
    """Manages spot-check selection and results."""

    def __init__(self, db_path: Path, sample_rate: float = 0.10):
        self.db_path = str(db_path)
        self.sample_rate = sample_rate

    def get_all_unreviewed(self, days: int = 7) -> list[dict]:
        """Return all auto-approved books not yet spot-checked (no sampling)."""
        with get_pipeline_db(self.db_path) as conn:
            cursor = conn.cursor()
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            cursor.execute("""
                SELECT f.book_id, f.original_book_type, f.original_confidence, f.created_at
                FROM autonomy_feedback f
                LEFT JOIN spot_checks s ON f.book_id = s.book_id
                WHERE f.original_decision = 'auto_approved'
                AND f.human_decision = 'approved'
                AND f.created_at > ?
                AND s.id IS NULL
                ORDER BY f.created_at DESC
            """, (cutoff,))
            return [dict(row) for row in cursor.fetchall()]

    def select_for_review(self, days: int = 7) -> list[dict]:
        """Select auto-approved books for spot-check review."""
        with get_pipeline_db(self.db_path) as conn:
            cursor = conn.cursor()

            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

            # Get auto-approved books not yet spot-checked
            cursor.execute("""
                SELECT f.book_id, f.original_book_type, f.original_confidence, f.created_at
                FROM autonomy_feedback f
                LEFT JOIN spot_checks s ON f.book_id = s.book_id
                WHERE f.original_decision = 'auto_approved'
                AND f.human_decision = 'approved'
                AND f.created_at > ?
                AND s.id IS NULL
            """, (cutoff,))

            candidates = [dict(row) for row in cursor.fetchall()]

            if not candidates:
                return []

            # Calculate sample size
            sample_size = max(1, int(len(candidates) * self.sample_rate))

            # Random sample
            return random.sample(candidates, min(sample_size, len(candidates)))

    def submit_result(
        self,
        book_id: str,
        classification_correct: bool,
        quality_acceptable: bool,
        reviewer: str,
        notes: str = None,
        pipeline_id: str = None,
    ) -> None:
        """Submit a spot-check review result."""
        with get_pipeline_db(self.db_path) as conn:
            cursor = conn.cursor()

            # Get original info
            cursor.execute("""
                SELECT original_book_type, original_confidence, created_at
                FROM autonomy_feedback
                WHERE book_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (book_id,))
            original = cursor.fetchone()

            cursor.execute("""
                INSERT INTO spot_checks
                (book_id, pipeline_id, original_classification, original_confidence,
                 auto_approved_at, classification_correct, quality_acceptable, reviewer, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                book_id,
                pipeline_id,
                original["original_book_type"] if original else None,
                original["original_confidence"] if original else None,
                original["created_at"] if original else None,
                classification_correct,
                quality_acceptable,
                reviewer,
                notes,
            ))

            conn.commit()

    def get_results(self, days: int = 30) -> list[dict]:
        """Get spot-check results for the specified period."""
        with get_pipeline_db(self.db_path) as conn:
            cursor = conn.cursor()

            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

            cursor.execute("""
                SELECT * FROM spot_checks
                WHERE checked_at > ?
                ORDER BY checked_at DESC
            """, (cutoff,))

            return [dict(row) for row in cursor.fetchall()]

    def get_accuracy_rate(self, days: int = 30) -> Optional[float]:
        """Get the accuracy rate from spot-checks."""
        with get_pipeline_db(self.db_path) as conn:
            cursor = conn.cursor()

            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN classification_correct = 1 THEN 1 ELSE 0 END) as correct
                FROM spot_checks
                WHERE checked_at > ?
            """, (cutoff,))

            row = cursor.fetchone()

            total = row["total"] or 0
            if total == 0:
                return None

            return row["correct"] / total

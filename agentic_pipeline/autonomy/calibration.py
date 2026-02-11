"""Calibration engine for measuring accuracy vs confidence."""

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from agentic_pipeline.db.connection import get_pipeline_db


class CalibrationEngine:
    """Calculates calibration metrics and thresholds."""

    def __init__(
        self,
        db_path: Path,
        min_samples: int = 50,
        target_accuracy: float = 0.95,
    ):
        self.db_path = str(db_path)
        self.min_samples = min_samples
        self.target_accuracy = target_accuracy

    def calculate_calibration(self, book_type: str, days: int = 90) -> Optional[dict]:
        """Calculate calibration metrics for a book type."""
        with get_pipeline_db(self.db_path) as conn:
            cursor = conn.cursor()

            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN human_decision = 'approved' THEN 1 ELSE 0 END) as correct,
                    AVG(original_confidence) as avg_confidence
                FROM autonomy_feedback
                WHERE original_book_type = ?
                AND created_at > ?
            """, (book_type, cutoff))

            row = cursor.fetchone()

            total = row["total"] or 0

            if total < self.min_samples:
                return None

            correct = row["correct"] or 0
            accuracy = correct / total

            return {
                "book_type": book_type,
                "sample_count": total,
                "accuracy": accuracy,
                "avg_confidence": row["avg_confidence"],
            }

    def calculate_threshold(self, book_type: str) -> Optional[float]:
        """Calculate the safe auto-approve threshold for a book type."""
        with get_pipeline_db(self.db_path) as conn:
            cursor = conn.cursor()

            cutoff = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()

            # Try progressively lower thresholds
            for threshold in [0.95, 0.92, 0.90, 0.88, 0.85, 0.82, 0.80]:
                cursor.execute("""
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN human_decision = 'approved' THEN 1 ELSE 0 END) as correct
                    FROM autonomy_feedback
                    WHERE original_book_type = ?
                    AND original_confidence >= ?
                    AND created_at > ?
                """, (book_type, threshold, cutoff))

                row = cursor.fetchone()
                total = row["total"] or 0

                if total < 10:  # Need at least 10 samples at this threshold
                    continue

                correct = row["correct"] or 0
                accuracy = correct / total

                if accuracy >= self.target_accuracy:
                    return threshold

            return None

    def update_thresholds(self) -> dict:
        """Recalculate and update all thresholds."""
        with get_pipeline_db(self.db_path) as conn:
            cursor = conn.cursor()

            # Get all book types with data
            cursor.execute("""
                SELECT DISTINCT original_book_type FROM autonomy_feedback
                WHERE original_book_type IS NOT NULL
            """)
            book_types = [row[0] for row in cursor.fetchall()]

            results = {}
            for book_type in book_types:
                calibration = self.calculate_calibration(book_type)
                threshold = self.calculate_threshold(book_type)

                if calibration:
                    cursor.execute("""
                        INSERT OR REPLACE INTO autonomy_thresholds
                        (book_type, auto_approve_threshold, sample_count, measured_accuracy, last_calculated)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        book_type,
                        threshold,
                        calibration["sample_count"],
                        calibration["accuracy"],
                        datetime.now(timezone.utc).isoformat()
                    ))

                    results[book_type] = {
                        "threshold": threshold,
                        "sample_count": calibration["sample_count"],
                        "accuracy": calibration["accuracy"],
                    }

            conn.commit()
            return results

"""Autonomy configuration manager."""

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agentic_pipeline.db.connection import get_pipeline_db


@dataclass(frozen=True)
class AutoApprovalDecision:
    """Explain a fail-closed automatic approval decision."""

    should_auto_approve: bool
    mode: str
    reason: str
    book_type: str
    confidence: float
    threshold: Optional[float] = None


class AutonomyConfig:
    """Manages autonomy mode and settings."""

    def __init__(self, db_path: Path):
        self.db_path = str(db_path)

    def get_mode(self) -> str:
        """Get current autonomy mode."""
        with get_pipeline_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT current_mode, escape_hatch_active FROM autonomy_config WHERE id = 1")
            row = cursor.fetchone()

            if row and row["escape_hatch_active"]:
                return "supervised"
            return row["current_mode"] if row else "supervised"

    def set_mode(self, mode: str) -> None:
        """Set autonomy mode."""
        if mode not in ("supervised", "partial", "confident"):
            raise ValueError(f"Invalid mode: {mode}")

        with get_pipeline_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE autonomy_config SET current_mode = ?, updated_at = ? WHERE id = 1",
                (mode, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()

    def activate_escape_hatch(self, reason: str) -> None:
        """Activate escape hatch - immediately revert to supervised."""
        with get_pipeline_db(self.db_path) as conn:
            cursor = conn.cursor()
            now = datetime.now(timezone.utc).isoformat()
            cursor.execute(
                """
                UPDATE autonomy_config SET
                    escape_hatch_active = TRUE,
                    escape_hatch_activated_at = ?,
                    escape_hatch_reason = ?,
                    updated_at = ?
                WHERE id = 1
            """,
                (now, reason, now),
            )
            conn.commit()

    def deactivate_escape_hatch(self) -> None:
        """Deactivate escape hatch."""
        with get_pipeline_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE autonomy_config SET
                    escape_hatch_active = FALSE,
                    updated_at = ?
                WHERE id = 1
            """,
                (datetime.now(timezone.utc).isoformat(),),
            )
            conn.commit()

    def is_escape_hatch_active(self) -> bool:
        """Check if escape hatch is active."""
        with get_pipeline_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT escape_hatch_active FROM autonomy_config WHERE id = 1")
            row = cursor.fetchone()
            return bool(row and row["escape_hatch_active"])

    def get_threshold(self, book_type: str) -> Optional[float]:
        """Get auto-approve threshold for a book type."""
        with get_pipeline_db(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT auto_approve_threshold, manual_override FROM autonomy_thresholds WHERE book_type = ?",
                (book_type,),
            )
            row = cursor.fetchone()

            if not row:
                return None
            return row["manual_override"] if row["manual_override"] is not None else row["auto_approve_threshold"]

    def _get_settings(self) -> dict:
        with get_pipeline_db(self.db_path) as conn:
            row = conn.execute("SELECT * FROM autonomy_config WHERE id = 1").fetchone()
        return (
            dict(row)
            if row
            else {
                "current_mode": "supervised",
                "escape_hatch_active": True,
            }
        )

    def _auto_approvals_today(self) -> int:
        with get_pipeline_db(self.db_path) as conn:
            row = conn.execute(
                """SELECT COUNT(*) AS total
                   FROM approval_audit
                   WHERE action = 'approved'
                   AND actor LIKE 'auto:%'
                   AND date(performed_at) = date('now')"""
            ).fetchone()
        return int(row["total"]) if row else 0

    def evaluate_auto_approval(
        self,
        book_type: str,
        confidence: float,
        *,
        validation_passed: bool = True,
        needs_review: bool = False,
    ) -> AutoApprovalDecision:
        """Evaluate automatic approval using the stored autonomy controls."""
        settings = self._get_settings()
        stored_mode = settings.get("current_mode") or "supervised"
        escape_hatch_active = bool(settings.get("escape_hatch_active"))
        mode = "supervised" if escape_hatch_active else stored_mode

        def deny(reason: str, threshold: Optional[float] = None) -> AutoApprovalDecision:
            return AutoApprovalDecision(
                should_auto_approve=False,
                mode=mode,
                reason=reason,
                book_type=book_type,
                confidence=confidence,
                threshold=threshold,
            )

        if escape_hatch_active:
            return deny("escape_hatch_active")
        if mode == "supervised":
            return deny("supervised_mode")
        if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            return deny("invalid_confidence")
        if not book_type or book_type == "unknown":
            return deny("unknown_book_type")
        if not validation_passed:
            return deny("validation_failed")
        if needs_review:
            return deny("needs_review")

        daily_limit = settings.get("max_auto_approvals_per_day")
        if daily_limit is not None and self._auto_approvals_today() >= int(daily_limit):
            return deny("daily_limit_reached")

        if mode == "partial":
            threshold = settings.get("auto_approve_threshold")
        elif mode == "confident":
            threshold = self.get_threshold(book_type)
        else:
            return deny("invalid_mode")

        if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            return deny("threshold_unavailable")
        if confidence < threshold:
            return deny("below_threshold", threshold)

        return AutoApprovalDecision(
            should_auto_approve=True,
            mode=mode,
            reason="approved",
            book_type=book_type,
            confidence=confidence,
            threshold=threshold,
        )

    def should_auto_approve(self, book_type: str, confidence: float) -> bool:
        """Compatibility wrapper for callers that only need a boolean."""
        return self.evaluate_auto_approval(book_type, confidence).should_auto_approve

"""Autonomy configuration manager."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class AutonomyConfig:
    """Manages autonomy mode and settings."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_mode(self) -> str:
        """Get current autonomy mode."""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("SELECT current_mode, escape_hatch_active FROM autonomy_config WHERE id = 1")
        row = cursor.fetchone()
        conn.close()

        if row and row["escape_hatch_active"]:
            return "supervised"
        return row["current_mode"] if row else "supervised"

    def set_mode(self, mode: str) -> None:
        """Set autonomy mode."""
        if mode not in ("supervised", "partial", "confident"):
            raise ValueError(f"Invalid mode: {mode}")

        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE autonomy_config SET current_mode = ?, updated_at = ? WHERE id = 1",
            (mode, datetime.now(timezone.utc).isoformat())
        )
        conn.commit()
        conn.close()

    def activate_escape_hatch(self, reason: str) -> None:
        """Activate escape hatch - immediately revert to supervised."""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE autonomy_config SET
                escape_hatch_active = TRUE,
                escape_hatch_activated_at = ?,
                escape_hatch_reason = ?,
                updated_at = ?
            WHERE id = 1
        """, (
            datetime.now(timezone.utc).isoformat(),
            reason,
            datetime.now(timezone.utc).isoformat()
        ))
        conn.commit()
        conn.close()

    def deactivate_escape_hatch(self) -> None:
        """Deactivate escape hatch."""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE autonomy_config SET
                escape_hatch_active = FALSE,
                updated_at = ?
            WHERE id = 1
        """, (datetime.now(timezone.utc).isoformat(),))
        conn.commit()
        conn.close()

    def is_escape_hatch_active(self) -> bool:
        """Check if escape hatch is active."""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("SELECT escape_hatch_active FROM autonomy_config WHERE id = 1")
        row = cursor.fetchone()
        conn.close()
        return bool(row and row["escape_hatch_active"])

    def get_threshold(self, book_type: str) -> Optional[float]:
        """Get auto-approve threshold for a book type."""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT auto_approve_threshold, manual_override FROM autonomy_thresholds WHERE book_type = ?",
            (book_type,)
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None
        return row["manual_override"] if row["manual_override"] else row["auto_approve_threshold"]

    def should_auto_approve(self, book_type: str, confidence: float) -> bool:
        """Determine if a book should be auto-approved."""
        mode = self.get_mode()

        if mode == "supervised":
            return False

        threshold = self.get_threshold(book_type)
        if threshold is None:
            return False

        return confidence >= threshold

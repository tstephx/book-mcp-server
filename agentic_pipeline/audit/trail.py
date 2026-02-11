"""Audit trail for tracking all approval decisions."""

import json
import sqlite3
from pathlib import Path
from typing import Optional


class AuditTrail:
    """Immutable append-only log of approval decisions."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def log(
        self,
        book_id: str,
        action: str,
        actor: str,
        pipeline_id: str = None,
        reason: str = None,
        before_state: dict = None,
        after_state: dict = None,
        adjustments: dict = None,
        filter_used: dict = None,
        confidence: float = None,
        autonomy_mode: str = None,
        session_id: str = None,
    ) -> int:
        """Log an audit entry. Returns the entry ID."""
        conn = self._connect()
        try:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO approval_audit
                (book_id, pipeline_id, action, actor, reason, before_state, after_state,
                 adjustments, filter_used, confidence_at_decision, autonomy_mode, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                book_id,
                pipeline_id,
                action,
                actor,
                reason,
                json.dumps(before_state) if before_state else None,
                json.dumps(after_state) if after_state else None,
                json.dumps(adjustments) if adjustments else None,
                json.dumps(filter_used) if filter_used else None,
                confidence,
                autonomy_mode,
                session_id,
            ))

            entry_id = cursor.lastrowid
            conn.commit()
            return entry_id
        finally:
            conn.close()

    def query(
        self,
        book_id: str = None,
        actor: str = None,
        action: str = None,
        last_days: int = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query audit entries."""
        conn = self._connect()
        try:
            cursor = conn.cursor()

            conditions = []
            params = []

            if book_id:
                conditions.append("book_id = ?")
                params.append(book_id)

            if actor:
                conditions.append("actor = ?")
                params.append(actor)

            if action:
                conditions.append("action = ?")
                params.append(action)

            if last_days:
                conditions.append("performed_at > datetime('now', ?)")
                params.append(f"-{last_days} days")

            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

            cursor.execute(f"""
                SELECT * FROM approval_audit
                {where}
                ORDER BY performed_at DESC
                LIMIT ?
            """, params + [limit])

            rows = cursor.fetchall()

            results = []
            for row in rows:
                entry = dict(row)
                # Parse JSON fields
                for field in ["before_state", "after_state", "adjustments", "filter_used"]:
                    if entry.get(field):
                        entry[field] = json.loads(entry[field])
                results.append(entry)

            return results
        finally:
            conn.close()

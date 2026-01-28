"""Pipeline repository for CRUD operations."""

import sqlite3
import uuid
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from agentic_pipeline.pipeline.states import PipelineState


class PipelineRepository:
    """Repository for pipeline records."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def create(
        self,
        source_path: str,
        content_hash: str,
        priority: int = 5
    ) -> str:
        """Create a new pipeline record."""
        pipeline_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO processing_pipelines
            (id, source_path, content_hash, state, priority, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (pipeline_id, source_path, content_hash, PipelineState.DETECTED.value, priority, now, now)
        )
        conn.commit()
        conn.close()

        return pipeline_id

    def get(self, pipeline_id: str) -> Optional[dict]:
        """Get a pipeline by ID."""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM processing_pipelines WHERE id = ?",
            (pipeline_id,)
        )
        row = cursor.fetchone()
        conn.close()

        return dict(row) if row else None

    def find_by_hash(self, content_hash: str) -> Optional[dict]:
        """Find a pipeline by content hash."""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM processing_pipelines WHERE content_hash = ?",
            (content_hash,)
        )
        row = cursor.fetchone()
        conn.close()

        return dict(row) if row else None

    def update_state(
        self,
        pipeline_id: str,
        new_state: PipelineState,
        agent_output: Optional[dict] = None,
        error_details: Optional[dict] = None
    ) -> None:
        """Update pipeline state and record history."""
        conn = self._connect()
        cursor = conn.cursor()

        # Get current state
        cursor.execute(
            "SELECT state, updated_at FROM processing_pipelines WHERE id = ?",
            (pipeline_id,)
        )
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise ValueError(f"Pipeline not found: {pipeline_id}")

        old_state = row["state"]
        old_updated = row["updated_at"]
        now = datetime.utcnow().isoformat()

        # Calculate duration
        duration_ms = None
        if old_updated:
            try:
                old_dt = datetime.fromisoformat(old_updated)
                duration_ms = int((datetime.utcnow() - old_dt).total_seconds() * 1000)
            except (ValueError, TypeError):
                pass

        # Update pipeline state
        cursor.execute(
            """
            UPDATE processing_pipelines
            SET state = ?, updated_at = ?
            WHERE id = ?
            """,
            (new_state.value, now, pipeline_id)
        )

        # Record state history
        cursor.execute(
            """
            INSERT INTO pipeline_state_history
            (pipeline_id, from_state, to_state, duration_ms, agent_output, error_details)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                pipeline_id,
                old_state,
                new_state.value,
                duration_ms,
                json.dumps(agent_output) if agent_output else None,
                json.dumps(error_details) if error_details else None
            )
        )

        conn.commit()
        conn.close()

    def list_pending_approval(self) -> list[dict]:
        """Get all pipelines pending approval."""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM processing_pipelines
            WHERE state = ?
            ORDER BY priority ASC, created_at ASC
            """,
            (PipelineState.PENDING_APPROVAL.value,)
        )
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def update_book_profile(self, pipeline_id: str, book_profile: dict) -> None:
        """Update the book profile from classifier."""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE processing_pipelines
            SET book_profile = ?, updated_at = ?
            WHERE id = ?
            """,
            (json.dumps(book_profile), datetime.utcnow().isoformat(), pipeline_id)
        )
        conn.commit()
        conn.close()

    def update_strategy_config(self, pipeline_id: str, strategy_config: dict) -> None:
        """Update the strategy config."""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE processing_pipelines
            SET strategy_config = ?, updated_at = ?
            WHERE id = ?
            """,
            (json.dumps(strategy_config), datetime.utcnow().isoformat(), pipeline_id)
        )
        conn.commit()
        conn.close()

    def update_validation_result(self, pipeline_id: str, validation_result: dict) -> None:
        """Update the validation result."""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE processing_pipelines
            SET validation_result = ?, updated_at = ?
            WHERE id = ?
            """,
            (json.dumps(validation_result), datetime.utcnow().isoformat(), pipeline_id)
        )
        conn.commit()
        conn.close()

    def mark_approved(self, pipeline_id: str, approved_by: str, confidence: float = None) -> None:
        """Mark a pipeline as approved."""
        conn = self._connect()
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()
        cursor.execute(
            """
            UPDATE processing_pipelines
            SET state = ?, approved_by = ?, approval_confidence = ?, updated_at = ?
            WHERE id = ?
            """,
            (PipelineState.APPROVED.value, approved_by, confidence, now, pipeline_id)
        )
        conn.commit()
        conn.close()

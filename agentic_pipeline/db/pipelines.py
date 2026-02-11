"""Pipeline repository for CRUD operations."""

import sqlite3
import uuid
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from agentic_pipeline.pipeline.states import PipelineState, can_transition


class PipelineRepository:
    """Repository for pipeline records."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
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
        now = datetime.now(timezone.utc).isoformat()

        conn = self._connect()
        try:
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
        finally:
            conn.close()

        return pipeline_id

    def get(self, pipeline_id: str) -> Optional[dict]:
        """Get a pipeline by ID."""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM processing_pipelines WHERE id = ?",
                (pipeline_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def find_by_hash(self, content_hash: str) -> Optional[dict]:
        """Find a pipeline by content hash."""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM processing_pipelines WHERE content_hash = ?",
                (content_hash,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def update_state(
        self,
        pipeline_id: str,
        new_state: PipelineState,
        agent_output: Optional[dict] = None,
        error_details: Optional[dict] = None
    ) -> None:
        """Update pipeline state and record history."""
        conn = self._connect()
        try:
            cursor = conn.cursor()

            # Get current state
            cursor.execute(
                "SELECT state, updated_at FROM processing_pipelines WHERE id = ?",
                (pipeline_id,)
            )
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Pipeline not found: {pipeline_id}")

            old_state = row["state"]
            old_updated = row["updated_at"]
            now = datetime.now(timezone.utc).isoformat()

            # Validate transition
            try:
                old_state_enum = PipelineState(old_state)
                if not can_transition(old_state_enum, new_state):
                    raise ValueError(
                        f"Invalid transition: {old_state} -> {new_state.value}"
                    )
            except ValueError as e:
                if "Invalid transition" in str(e):
                    raise
                # Unknown old state in DB — allow the transition

            # Calculate duration
            duration_ms = None
            if old_updated:
                try:
                    old_dt = datetime.fromisoformat(old_updated)
                    if old_dt.tzinfo is None:
                        old_dt = old_dt.replace(tzinfo=timezone.utc)
                    duration_ms = int((datetime.now(timezone.utc) - old_dt).total_seconds() * 1000)
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
        finally:
            conn.close()

    def list_pending_approval(self) -> list[dict]:
        """Get all pipelines pending approval."""
        conn = self._connect()
        try:
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
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def update_book_profile(self, pipeline_id: str, book_profile: dict) -> None:
        """Update the book profile from classifier."""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE processing_pipelines
                SET book_profile = ?, updated_at = ?
                WHERE id = ?
                """,
                (json.dumps(book_profile), datetime.now(timezone.utc).isoformat(), pipeline_id)
            )
            conn.commit()
        finally:
            conn.close()

    def update_strategy_config(self, pipeline_id: str, strategy_config: dict) -> None:
        """Update the strategy config."""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE processing_pipelines
                SET strategy_config = ?, updated_at = ?
                WHERE id = ?
                """,
                (json.dumps(strategy_config), datetime.now(timezone.utc).isoformat(), pipeline_id)
            )
            conn.commit()
        finally:
            conn.close()

    def update_validation_result(self, pipeline_id: str, validation_result: dict) -> None:
        """Update the validation result."""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE processing_pipelines
                SET validation_result = ?, updated_at = ?
                WHERE id = ?
                """,
                (json.dumps(validation_result), datetime.now(timezone.utc).isoformat(), pipeline_id)
            )
            conn.commit()
        finally:
            conn.close()

    def update_processing_result(self, pipeline_id: str, processing_result: dict) -> None:
        """Update the processing result from book-ingestion.

        Stores quality metrics, detection confidence, and warnings from
        the book-ingestion pipeline result.
        """
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE processing_pipelines
                SET processing_result = ?, updated_at = ?
                WHERE id = ?
                """,
                (json.dumps(processing_result), datetime.now(timezone.utc).isoformat(), pipeline_id)
            )
            conn.commit()
        finally:
            conn.close()

    def mark_approved(self, pipeline_id: str, approved_by: str, confidence: float = None) -> None:
        """Mark a pipeline as approved."""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            now = datetime.now(timezone.utc).isoformat()
            cursor.execute(
                """
                UPDATE processing_pipelines
                SET state = ?, approved_by = ?, approval_confidence = ?, updated_at = ?
                WHERE id = ?
                """,
                (PipelineState.APPROVED.value, approved_by, confidence, now, pipeline_id)
            )
            conn.commit()
        finally:
            conn.close()

    def find_by_state(
        self,
        state: PipelineState,
        limit: int = None
    ) -> list[dict]:
        """Find pipelines in a specific state."""
        conn = self._connect()
        try:
            cursor = conn.cursor()

            query = """
                SELECT * FROM processing_pipelines
                WHERE state = ?
                ORDER BY priority ASC, created_at ASC
            """
            params = [state.value]
            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def increment_retry_count(self, pipeline_id: str) -> int:
        """Increment retry count and return new value."""
        conn = self._connect()
        try:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE processing_pipelines
                SET retry_count = retry_count + 1, updated_at = ?
                WHERE id = ?
            """, (datetime.now(timezone.utc).isoformat(), pipeline_id))

            cursor.execute(
                "SELECT retry_count FROM processing_pipelines WHERE id = ?",
                (pipeline_id,)
            )
            new_count = cursor.fetchone()[0]

            conn.commit()
            return new_count
        finally:
            conn.close()

    def update_priority(self, pipeline_id: str, priority: int) -> None:
        """Update the priority of a pipeline."""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE processing_pipelines SET priority = ?, updated_at = ? WHERE id = ?",
                (priority, datetime.now(timezone.utc).isoformat(), pipeline_id)
            )
            conn.commit()
        finally:
            conn.close()

    def create_backfill(
        self,
        book_id: str,
        source_path: str,
        content_hash: str,
    ) -> bool:
        """Create a pipeline record for an existing library book.

        Inserts directly at COMPLETE state, bypassing normal transitions.
        Used to register legacy books that were ingested before the pipeline existed.

        Returns True if created, False if skipped (hash already exists).
        """
        now = datetime.now(timezone.utc).isoformat()
        conn = self._connect()
        try:
            cursor = conn.cursor()

            # Skip if content_hash already tracked
            cursor.execute(
                "SELECT id FROM processing_pipelines WHERE content_hash = ?",
                (content_hash,),
            )
            if cursor.fetchone():
                return False

            cursor.execute(
                """
                INSERT INTO processing_pipelines
                (id, source_path, content_hash, state, approved_by,
                 created_at, updated_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    book_id,
                    source_path,
                    content_hash,
                    PipelineState.COMPLETE.value,
                    "backfill:automated",
                    now,
                    now,
                    now,
                ),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def prepare_reingest(self, pipeline_id: str) -> str:
        """Archive an existing pipeline record and create a new one for reingestion.

        Returns the new pipeline_id.
        """
        conn = self._connect()
        try:
            cursor = conn.cursor()
            now = datetime.now(timezone.utc).isoformat()

            # Get existing record
            cursor.execute(
                "SELECT * FROM processing_pipelines WHERE id = ?",
                (pipeline_id,),
            )
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Pipeline not found: {pipeline_id}")

            old = dict(row)

            # Archive the old record
            cursor.execute(
                "UPDATE processing_pipelines SET state = ?, updated_at = ? WHERE id = ?",
                (PipelineState.ARCHIVED.value, now, pipeline_id),
            )

            # Record archive in state history
            cursor.execute(
                """
                INSERT INTO pipeline_state_history
                (pipeline_id, from_state, to_state, agent_output)
                VALUES (?, ?, ?, ?)
                """,
                (pipeline_id, old["state"], PipelineState.ARCHIVED.value,
                 json.dumps({"reason": "reingest_requested"})),
            )

            # Remove the UNIQUE constraint collision by clearing old hash
            old_hash = old["content_hash"]
            cursor.execute(
                "UPDATE processing_pipelines SET content_hash = ? WHERE id = ?",
                (f"archived:{old_hash}", pipeline_id),
            )

            # Create new pipeline record
            new_id = str(uuid.uuid4())
            cursor.execute(
                """
                INSERT INTO processing_pipelines
                (id, source_path, content_hash, state, priority, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (new_id, old["source_path"], old_hash,
                 PipelineState.DETECTED.value, 3, now, now),
            )

            conn.commit()
            return new_id
        finally:
            conn.close()

    def get_queue_by_priority(self) -> dict[int, int]:
        """Get count of queued items by priority."""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT priority, COUNT(*) as count
                FROM processing_pipelines
                WHERE state = ?
                GROUP BY priority
                ORDER BY priority
            """, (PipelineState.DETECTED.value,))
            rows = cursor.fetchall()
            return {row[0]: row[1] for row in rows}
        finally:
            conn.close()

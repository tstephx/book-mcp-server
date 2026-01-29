"""Health monitoring for the pipeline."""

import sqlite3
from pathlib import Path

from agentic_pipeline.db.pipelines import PipelineRepository
from agentic_pipeline.pipeline.states import PipelineState


# States that indicate active processing
ACTIVE_STATES = [
    PipelineState.HASHING,
    PipelineState.CLASSIFYING,
    PipelineState.SELECTING_STRATEGY,
    PipelineState.PROCESSING,
    PipelineState.VALIDATING,
    PipelineState.EMBEDDING,
]


class HealthMonitor:
    """Aggregates pipeline health metrics."""

    def __init__(
        self,
        db_path: Path,
        alert_queue_threshold: int = 100,
        alert_failure_rate: float = 0.20,
    ):
        self.db_path = db_path
        self.repo = PipelineRepository(db_path)
        self.alert_queue_threshold = alert_queue_threshold
        self.alert_failure_rate = alert_failure_rate

    def get_health(self) -> dict:
        """Get current system health."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Count active (processing)
        active_states = [s.value for s in ACTIVE_STATES]
        placeholders = ",".join("?" * len(active_states))
        cursor.execute(f"""
            SELECT COUNT(*) FROM processing_pipelines
            WHERE state IN ({placeholders})
        """, active_states)
        active = cursor.fetchone()[0]

        # Count queued (detected)
        cursor.execute("""
            SELECT COUNT(*) FROM processing_pipelines
            WHERE state = ?
        """, (PipelineState.DETECTED.value,))
        queued = cursor.fetchone()[0]

        # Count failed (needs_retry)
        cursor.execute("""
            SELECT COUNT(*) FROM processing_pipelines
            WHERE state = ?
        """, (PipelineState.NEEDS_RETRY.value,))
        failed = cursor.fetchone()[0]

        # Count completed in last 24h
        cursor.execute("""
            SELECT COUNT(*) FROM processing_pipelines
            WHERE state = ?
            AND updated_at > datetime('now', '-24 hours')
        """, (PipelineState.COMPLETE.value,))
        completed_24h = cursor.fetchone()[0]

        # Queue by priority
        queue_by_priority = self.repo.get_queue_by_priority()

        conn.close()

        # Generate alerts
        alerts = self._generate_alerts(queued, failed, completed_24h)

        # Determine status
        if active > 0:
            status = "processing"
        elif queued > 0:
            status = "queued"
        elif failed > 0:
            status = "has_failures"
        else:
            status = "idle"

        return {
            "active": active,
            "queued": queued,
            "failed": failed,
            "completed_24h": completed_24h,
            "stuck": [],  # Will be populated by stuck detector
            "queue_by_priority": queue_by_priority,
            "alerts": alerts,
            "status": status,
        }

    def _generate_alerts(self, queued: int, failed: int, completed_24h: int) -> list[dict]:
        """Generate alerts based on current state."""
        alerts = []

        if queued > self.alert_queue_threshold:
            alerts.append({
                "type": "queue_backup",
                "severity": "info",
                "message": f"Queue has {queued} books waiting (threshold: {self.alert_queue_threshold})"
            })

        # Check failure rate
        total_recent = completed_24h + failed
        if total_recent > 10:  # Only check if enough data
            failure_rate = failed / total_recent
            if failure_rate > self.alert_failure_rate:
                alerts.append({
                    "type": "high_failure_rate",
                    "severity": "warning",
                    "message": f"Failure rate is {failure_rate:.0%} (threshold: {self.alert_failure_rate:.0%})"
                })

        return alerts

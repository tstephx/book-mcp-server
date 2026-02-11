"""Stuck detection for pipelines."""

from datetime import datetime, timezone, timedelta
from pathlib import Path

from agentic_pipeline.db.connection import get_pipeline_db
from agentic_pipeline.pipeline.states import PipelineState


# Default timeout thresholds in seconds
DEFAULT_STATE_TIMEOUTS = {
    "DETECTED": 300,          # 5 minutes
    "HASHING": 60,            # 1 minute
    "CLASSIFYING": 120,       # 2 minutes
    "SELECTING_STRATEGY": 30, # 30 seconds
    "PROCESSING": 900,        # 15 minutes
    "VALIDATING": 60,         # 1 minute
    "PENDING_APPROVAL": None, # No timeout - waiting for human
    "APPROVED": 60,           # 1 minute
    "EMBEDDING": 600,         # 10 minutes
}

# States that should be checked for stuck
NON_TERMINAL_STATES = [
    PipelineState.DETECTED,
    PipelineState.HASHING,
    PipelineState.CLASSIFYING,
    PipelineState.SELECTING_STRATEGY,
    PipelineState.PROCESSING,
    PipelineState.VALIDATING,
    PipelineState.APPROVED,
    PipelineState.EMBEDDING,
]


class StuckDetector:
    """Detects pipelines that appear to be stuck."""

    def __init__(
        self,
        db_path: Path,
        stuck_multiplier: float = 2.0,
        custom_thresholds: dict = None,
    ):
        self.db_path = str(db_path)
        self.stuck_multiplier = stuck_multiplier
        self.thresholds = {**DEFAULT_STATE_TIMEOUTS, **(custom_thresholds or {})}

    def detect(self) -> list[dict]:
        """Find pipelines that appear to be stuck."""
        with get_pipeline_db(self.db_path) as conn:
            cursor = conn.cursor()

            stuck = []
            now = datetime.now(timezone.utc)

            for state in NON_TERMINAL_STATES:
                threshold_seconds = self.thresholds.get(state.value.upper())
                if threshold_seconds is None:
                    continue  # No timeout for this state (e.g., PENDING_APPROVAL)

                # Calculate stuck threshold
                stuck_threshold = threshold_seconds * self.stuck_multiplier
                cutoff = (now - timedelta(seconds=stuck_threshold)).isoformat()

                cursor.execute("""
                    SELECT * FROM processing_pipelines
                    WHERE state = ?
                    AND updated_at < ?
                """, (state.value, cutoff))

                for row in cursor.fetchall():
                    pipeline = dict(row)
                    updated_at_str = pipeline["updated_at"]

                    # Parse the datetime
                    try:
                        if updated_at_str.endswith("Z"):
                            updated_at = datetime.fromisoformat(updated_at_str.replace("Z", "+00:00"))
                        elif "+" in updated_at_str or updated_at_str.count("-") > 2:
                            updated_at = datetime.fromisoformat(updated_at_str)
                        else:
                            updated_at = datetime.fromisoformat(updated_at_str).replace(tzinfo=timezone.utc)
                    except (ValueError, AttributeError):
                        updated_at = now  # Fallback

                    if updated_at.tzinfo is None:
                        updated_at = updated_at.replace(tzinfo=timezone.utc)

                    stuck_duration = now - updated_at

                    stuck.append({
                        "id": pipeline["id"],
                        "state": pipeline["state"],
                        "source_path": pipeline["source_path"],
                        "stuck_since": pipeline["updated_at"],
                        "stuck_minutes": int(stuck_duration.total_seconds() / 60),
                        "expected_minutes": int(threshold_seconds / 60),
                    })

            return stuck

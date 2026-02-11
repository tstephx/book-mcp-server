"""Batch filter for selecting pipelines."""

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class BatchFilter:
    """Filter for batch operations."""

    book_type: Optional[str] = None
    min_confidence: Optional[float] = None
    max_confidence: Optional[float] = None
    state: Optional[str] = None
    created_before: Optional[datetime] = None
    created_after: Optional[datetime] = None
    source_path_pattern: Optional[str] = None
    max_count: int = 50

    def apply(self, db_path: Path) -> list[dict]:
        """Apply filter and return matching pipelines."""
        conn = sqlite3.connect(db_path, timeout=10)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            conditions = []
            params = []

            # Default to pending_approval if no state specified
            if self.state:
                conditions.append("state = ?")
                params.append(self.state)
            else:
                conditions.append("state = ?")
                params.append("pending_approval")

            if self.source_path_pattern:
                conditions.append("source_path GLOB ?")
                params.append(self.source_path_pattern)

            if self.created_before:
                conditions.append("created_at < ?")
                params.append(self.created_before.isoformat())

            if self.created_after:
                conditions.append("created_at > ?")
                params.append(self.created_after.isoformat())

            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

            cursor.execute(f"""
                SELECT * FROM processing_pipelines
                {where}
                ORDER BY priority ASC, created_at ASC
                LIMIT ?
            """, params + [self.max_count * 2])  # Fetch extra for post-filtering

            rows = cursor.fetchall()
        finally:
            conn.close()

        results = []
        for row in rows:
            pipeline = dict(row)

            # Filter by book profile fields (stored as JSON)
            if self.min_confidence is not None or self.max_confidence is not None or self.book_type:
                profile = json.loads(pipeline.get("book_profile") or "{}")

                if self.book_type and profile.get("book_type") != self.book_type:
                    continue

                confidence = profile.get("confidence", 0)
                if self.min_confidence is not None and confidence < self.min_confidence:
                    continue
                if self.max_confidence is not None and confidence > self.max_confidence:
                    continue

            results.append(pipeline)

            if len(results) >= self.max_count:
                break

        return results

    def to_dict(self) -> dict:
        """Convert filter to dictionary for audit logging."""
        d = asdict(self)
        # Convert datetime to string
        if d.get("created_before"):
            d["created_before"] = d["created_before"].isoformat()
        if d.get("created_after"):
            d["created_after"] = d["created_after"].isoformat()
        # Remove None values
        return {k: v for k, v in d.items() if v is not None}

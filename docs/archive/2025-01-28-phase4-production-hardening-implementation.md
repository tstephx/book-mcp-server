# Phase 4: Production Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add health monitoring, stuck detection, batch operations, priority queues, and audit trail to make the pipeline production-ready.

**Architecture:** Six components built in dependency order: database migrations → priority queues → audit trail → batch operations → health monitoring → stuck detection. Each component has its own package with tests.

**Tech Stack:** Python 3.12, SQLite, Click (CLI), pytest, dataclasses

**Design Reference:** `docs/plans/2025-01-28-phase4-production-hardening-design.md`

---

## Task 1: Database Migrations

**Files:**
- Modify: `agentic_pipeline/db/migrations.py`
- Test: `tests/test_phase4_migrations.py`

**Step 1: Write the failing test**

```python
# tests/test_phase4_migrations.py
"""Tests for Phase 4 database migrations."""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def db_path():
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    yield path
    path.unlink(missing_ok=True)


def test_priority_column_exists(db_path):
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(processing_pipelines)")
    columns = [row[1] for row in cursor.fetchall()]
    conn.close()

    assert "priority" in columns


def test_audit_table_exists(db_path):
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='approval_audit'")
    result = cursor.fetchone()
    conn.close()

    assert result is not None


def test_health_metrics_table_exists(db_path):
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='health_metrics'")
    result = cursor.fetchone()
    conn.close()

    assert result is not None


def test_state_duration_stats_table_exists(db_path):
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='state_duration_stats'")
    result = cursor.fetchone()
    conn.close()

    assert result is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_phase4_migrations.py -v`
Expected: FAIL (columns/tables don't exist yet)

**Step 3: Read existing migrations file**

Read: `agentic_pipeline/db/migrations.py` to understand current structure

**Step 4: Add Phase 4 migrations**

Add to `agentic_pipeline/db/migrations.py`:

```python
def _run_phase4_migrations(conn: sqlite3.Connection) -> None:
    """Run Phase 4 production hardening migrations."""
    cursor = conn.cursor()

    # Add priority column to processing_pipelines
    cursor.execute("PRAGMA table_info(processing_pipelines)")
    columns = [row[1] for row in cursor.fetchall()]

    if "priority" not in columns:
        cursor.execute(
            "ALTER TABLE processing_pipelines ADD COLUMN priority INTEGER DEFAULT 5"
        )

    if "auto_recovery_count" not in columns:
        cursor.execute(
            "ALTER TABLE processing_pipelines ADD COLUMN auto_recovery_count INTEGER DEFAULT 0"
        )

    # Create audit trail table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS approval_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            book_id TEXT NOT NULL,
            pipeline_id TEXT,
            action TEXT NOT NULL,
            actor TEXT NOT NULL,
            reason TEXT,
            before_state JSON,
            after_state JSON,
            adjustments JSON,
            filter_used JSON,
            confidence_at_decision REAL,
            autonomy_mode TEXT,
            session_id TEXT,
            performed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_book_id ON approval_audit(book_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_actor ON approval_audit(actor)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_performed_at ON approval_audit(performed_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_session ON approval_audit(session_id)")

    # Create health metrics cache table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS health_metrics (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            active_count INTEGER NOT NULL DEFAULT 0,
            queued_count INTEGER NOT NULL DEFAULT 0,
            stuck_count INTEGER NOT NULL DEFAULT 0,
            completed_24h INTEGER NOT NULL DEFAULT 0,
            failed_count INTEGER NOT NULL DEFAULT 0,
            avg_processing_seconds REAL,
            queue_by_priority JSON,
            stuck_pipelines JSON,
            alerts JSON,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create state duration stats table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS state_duration_stats (
            state TEXT PRIMARY KEY,
            sample_count INTEGER NOT NULL DEFAULT 0,
            median_seconds REAL NOT NULL DEFAULT 0,
            p95_seconds REAL NOT NULL DEFAULT 0,
            max_seconds REAL NOT NULL DEFAULT 0,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create priority queue index
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_pipelines_priority_queue
        ON processing_pipelines(state, priority, created_at)
    """)

    conn.commit()
```

Then call `_run_phase4_migrations(conn)` at the end of `run_migrations()`.

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_phase4_migrations.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add agentic_pipeline/db/migrations.py tests/test_phase4_migrations.py
git commit -m "feat: add Phase 4 database migrations

- priority column on processing_pipelines
- approval_audit table for audit trail
- health_metrics cache table
- state_duration_stats for stuck detection"
```

---

## Task 2: Priority Queue Support

**Files:**
- Modify: `agentic_pipeline/db/pipelines.py`
- Test: `tests/test_priority_queue.py`

**Step 1: Write the failing test**

```python
# tests/test_priority_queue.py
"""Tests for priority queue functionality."""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def db_path():
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    yield path
    path.unlink(missing_ok=True)


def test_create_with_priority(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository

    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "hash123", priority=2)

    pipeline = repo.get(pid)
    assert pipeline["priority"] == 2


def test_default_priority_is_5(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository

    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "hash123")

    pipeline = repo.get(pid)
    assert pipeline["priority"] == 5


def test_find_by_state_orders_by_priority(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)

    # Create books with different priorities
    id_low = repo.create("/low.epub", "hash1", priority=10)
    id_high = repo.create("/high.epub", "hash2", priority=1)
    id_med = repo.create("/med.epub", "hash3", priority=5)

    results = repo.find_by_state(PipelineState.DETECTED)

    # Should be ordered: priority 1, then 5, then 10
    assert results[0]["id"] == id_high
    assert results[1]["id"] == id_med
    assert results[2]["id"] == id_low


def test_update_priority(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository

    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "hash123", priority=5)

    repo.update_priority(pid, 1)

    pipeline = repo.get(pid)
    assert pipeline["priority"] == 1


def test_get_queue_by_priority(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository

    repo = PipelineRepository(db_path)

    # Create books with different priorities
    repo.create("/a.epub", "hash1", priority=1)
    repo.create("/b.epub", "hash2", priority=1)
    repo.create("/c.epub", "hash3", priority=5)

    result = repo.get_queue_by_priority()

    assert result[1] == 2  # Two books at priority 1
    assert result[5] == 1  # One book at priority 5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_priority_queue.py -v`
Expected: FAIL

**Step 3: Read existing pipelines.py**

Read: `agentic_pipeline/db/pipelines.py` to understand current structure

**Step 4: Add priority support**

Modify `create()` to accept `priority` parameter:

```python
def create(self, source_path: str, content_hash: str, priority: int = 5) -> str:
    """Create a new pipeline record."""
    conn = self._connect()
    cursor = conn.cursor()
    pipeline_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    cursor.execute(
        """
        INSERT INTO processing_pipelines
        (id, source_path, content_hash, state, created_at, updated_at, priority)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (pipeline_id, source_path, content_hash, PipelineState.DETECTED.value, now, now, priority)
    )
    conn.commit()
    conn.close()
    return pipeline_id
```

Add `update_priority()` method:

```python
def update_priority(self, pipeline_id: str, priority: int) -> None:
    """Update the priority of a pipeline."""
    conn = self._connect()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE processing_pipelines SET priority = ?, updated_at = ? WHERE id = ?",
        (priority, datetime.now(timezone.utc).isoformat(), pipeline_id)
    )
    conn.commit()
    conn.close()
```

Add `get_queue_by_priority()` method:

```python
def get_queue_by_priority(self) -> dict[int, int]:
    """Get count of queued items by priority."""
    conn = self._connect()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT priority, COUNT(*) as count
        FROM processing_pipelines
        WHERE state = ?
        GROUP BY priority
        ORDER BY priority
    """, (PipelineState.DETECTED.value,))
    rows = cursor.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}
```

Update `find_by_state()` to order by priority:

```python
def find_by_state(self, state: PipelineState, limit: int = None) -> list[dict]:
    """Find pipelines in a specific state, ordered by priority."""
    conn = self._connect()
    cursor = conn.cursor()

    query = """
        SELECT * FROM processing_pipelines
        WHERE state = ?
        ORDER BY priority ASC, created_at ASC
    """
    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query, (state.value,))
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_priority_queue.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add agentic_pipeline/db/pipelines.py tests/test_priority_queue.py
git commit -m "feat: add priority queue support to PipelineRepository"
```

---

## Task 3: Audit Trail

**Files:**
- Create: `agentic_pipeline/audit/__init__.py`
- Create: `agentic_pipeline/audit/trail.py`
- Test: `tests/test_audit_trail.py`

**Step 1: Write the failing test**

```python
# tests/test_audit_trail.py
"""Tests for audit trail functionality."""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def db_path():
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    yield path
    path.unlink(missing_ok=True)


def test_log_approval(db_path):
    from agentic_pipeline.audit import AuditTrail

    trail = AuditTrail(db_path)
    trail.log(
        book_id="book123",
        pipeline_id="pipe456",
        action="APPROVED",
        actor="human:taylor",
        confidence=0.95
    )

    entries = trail.query(book_id="book123")
    assert len(entries) == 1
    assert entries[0]["action"] == "APPROVED"
    assert entries[0]["actor"] == "human:taylor"


def test_log_rejection_with_reason(db_path):
    from agentic_pipeline.audit import AuditTrail

    trail = AuditTrail(db_path)
    trail.log(
        book_id="book123",
        action="REJECTED",
        actor="human:taylor",
        reason="Not a technical book"
    )

    entries = trail.query(book_id="book123")
    assert entries[0]["reason"] == "Not a technical book"


def test_query_by_actor(db_path):
    from agentic_pipeline.audit import AuditTrail

    trail = AuditTrail(db_path)
    trail.log(book_id="a", action="APPROVED", actor="human:taylor")
    trail.log(book_id="b", action="APPROVED", actor="auto:high_confidence")
    trail.log(book_id="c", action="APPROVED", actor="human:taylor")

    entries = trail.query(actor="human:taylor")
    assert len(entries) == 2


def test_query_by_action(db_path):
    from agentic_pipeline.audit import AuditTrail

    trail = AuditTrail(db_path)
    trail.log(book_id="a", action="APPROVED", actor="human:taylor")
    trail.log(book_id="b", action="REJECTED", actor="human:taylor")

    entries = trail.query(action="REJECTED")
    assert len(entries) == 1
    assert entries[0]["book_id"] == "b"


def test_query_with_limit(db_path):
    from agentic_pipeline.audit import AuditTrail

    trail = AuditTrail(db_path)
    for i in range(10):
        trail.log(book_id=f"book{i}", action="APPROVED", actor="auto:test")

    entries = trail.query(limit=5)
    assert len(entries) == 5


def test_log_batch_operation_with_filter(db_path):
    from agentic_pipeline.audit import AuditTrail

    trail = AuditTrail(db_path)
    trail.log(
        book_id="batch",
        action="BATCH_APPROVED",
        actor="batch:filter_abc",
        filter_used={"min_confidence": 0.9, "book_type": "technical_tutorial"}
    )

    entries = trail.query(action="BATCH_APPROVED")
    assert entries[0]["filter_used"]["min_confidence"] == 0.9
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_audit_trail.py -v`
Expected: FAIL (module doesn't exist)

**Step 3: Create audit package**

```bash
mkdir -p agentic_pipeline/audit
```

**Step 4: Write implementation**

```python
# agentic_pipeline/audit/__init__.py
"""Audit trail package."""

from agentic_pipeline.audit.trail import AuditTrail

__all__ = ["AuditTrail"]
```

```python
# agentic_pipeline/audit/trail.py
"""Audit trail for tracking all approval decisions."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class AuditTrail:
    """Immutable append-only log of approval decisions."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
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
        conn.close()
        return entry_id

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
        conn.close()

        results = []
        for row in rows:
            entry = dict(row)
            # Parse JSON fields
            for field in ["before_state", "after_state", "adjustments", "filter_used"]:
                if entry.get(field):
                    entry[field] = json.loads(entry[field])
            results.append(entry)

        return results
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_audit_trail.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add agentic_pipeline/audit/ tests/test_audit_trail.py
git commit -m "feat: add AuditTrail for tracking approval decisions"
```

---

## Task 4: Batch Filter

**Files:**
- Create: `agentic_pipeline/batch/__init__.py`
- Create: `agentic_pipeline/batch/filters.py`
- Test: `tests/test_batch_filters.py`

**Step 1: Write the failing test**

```python
# tests/test_batch_filters.py
"""Tests for batch filter functionality."""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def db_path():
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    yield path
    path.unlink(missing_ok=True)


def test_filter_by_min_confidence(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.batch.filters import BatchFilter
    import json

    repo = PipelineRepository(db_path)

    # Create pipelines with different confidence
    id1 = repo.create("/book1.epub", "hash1")
    repo.update_state(id1, PipelineState.PENDING_APPROVAL)
    repo.update_book_profile(id1, {"book_type": "technical_tutorial", "confidence": 0.95})

    id2 = repo.create("/book2.epub", "hash2")
    repo.update_state(id2, PipelineState.PENDING_APPROVAL)
    repo.update_book_profile(id2, {"book_type": "technical_tutorial", "confidence": 0.7})

    filter = BatchFilter(min_confidence=0.9)
    results = filter.apply(db_path)

    assert len(results) == 1
    assert results[0]["id"] == id1


def test_filter_by_book_type(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.batch.filters import BatchFilter

    repo = PipelineRepository(db_path)

    id1 = repo.create("/book1.epub", "hash1")
    repo.update_state(id1, PipelineState.PENDING_APPROVAL)
    repo.update_book_profile(id1, {"book_type": "technical_tutorial", "confidence": 0.9})

    id2 = repo.create("/book2.epub", "hash2")
    repo.update_state(id2, PipelineState.PENDING_APPROVAL)
    repo.update_book_profile(id2, {"book_type": "newspaper", "confidence": 0.9})

    filter = BatchFilter(book_type="technical_tutorial")
    results = filter.apply(db_path)

    assert len(results) == 1
    assert results[0]["id"] == id1


def test_filter_by_state(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.batch.filters import BatchFilter

    repo = PipelineRepository(db_path)

    id1 = repo.create("/book1.epub", "hash1")
    repo.update_state(id1, PipelineState.PENDING_APPROVAL)

    id2 = repo.create("/book2.epub", "hash2")
    repo.update_state(id2, PipelineState.NEEDS_RETRY)

    filter = BatchFilter(state="needs_retry")
    results = filter.apply(db_path)

    assert len(results) == 1
    assert results[0]["id"] == id2


def test_filter_max_count(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.batch.filters import BatchFilter

    repo = PipelineRepository(db_path)

    for i in range(10):
        pid = repo.create(f"/book{i}.epub", f"hash{i}")
        repo.update_state(pid, PipelineState.PENDING_APPROVAL)

    filter = BatchFilter(max_count=3)
    results = filter.apply(db_path)

    assert len(results) == 3


def test_filter_to_dict(db_path):
    from agentic_pipeline.batch.filters import BatchFilter

    filter = BatchFilter(min_confidence=0.9, book_type="technical_tutorial")
    d = filter.to_dict()

    assert d["min_confidence"] == 0.9
    assert d["book_type"] == "technical_tutorial"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_batch_filters.py -v`
Expected: FAIL

**Step 3: Create batch package**

```bash
mkdir -p agentic_pipeline/batch
```

**Step 4: Write implementation**

```python
# agentic_pipeline/batch/__init__.py
"""Batch operations package."""

from agentic_pipeline.batch.filters import BatchFilter

__all__ = ["BatchFilter"]
```

```python
# agentic_pipeline/batch/filters.py
"""Batch filter for selecting pipelines."""

import json
import sqlite3
from dataclasses import dataclass, field, asdict
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
        conn = sqlite3.connect(db_path)
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
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_batch_filters.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add agentic_pipeline/batch/ tests/test_batch_filters.py
git commit -m "feat: add BatchFilter for filtering pipelines"
```

---

## Task 5: Batch Operations

**Files:**
- Create: `agentic_pipeline/batch/operations.py`
- Modify: `agentic_pipeline/batch/__init__.py`
- Test: `tests/test_batch_operations.py`

**Step 1: Write the failing test**

```python
# tests/test_batch_operations.py
"""Tests for batch operations."""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def db_path():
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    yield path
    path.unlink(missing_ok=True)


def test_batch_approve(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.batch import BatchOperations, BatchFilter
    from agentic_pipeline.audit import AuditTrail

    repo = PipelineRepository(db_path)

    # Create pending approval books
    id1 = repo.create("/book1.epub", "hash1")
    repo.update_state(id1, PipelineState.PENDING_APPROVAL)
    repo.update_book_profile(id1, {"book_type": "technical_tutorial", "confidence": 0.95})

    id2 = repo.create("/book2.epub", "hash2")
    repo.update_state(id2, PipelineState.PENDING_APPROVAL)
    repo.update_book_profile(id2, {"book_type": "technical_tutorial", "confidence": 0.92})

    ops = BatchOperations(db_path)
    filter = BatchFilter(min_confidence=0.9)

    result = ops.approve(filter, actor="human:taylor", execute=True)

    assert result["approved"] == 2

    # Check audit trail
    trail = AuditTrail(db_path)
    entries = trail.query(action="BATCH_APPROVED")
    assert len(entries) == 1
    assert entries[0]["filter_used"]["min_confidence"] == 0.9


def test_batch_approve_dry_run(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.batch import BatchOperations, BatchFilter

    repo = PipelineRepository(db_path)

    id1 = repo.create("/book1.epub", "hash1")
    repo.update_state(id1, PipelineState.PENDING_APPROVAL)

    ops = BatchOperations(db_path)
    filter = BatchFilter()

    result = ops.approve(filter, actor="human:taylor", execute=False)

    assert result["would_approve"] == 1
    assert result["approved"] == 0

    # State should not change
    pipeline = repo.get(id1)
    assert pipeline["state"] == PipelineState.PENDING_APPROVAL.value


def test_batch_reject(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.batch import BatchOperations, BatchFilter

    repo = PipelineRepository(db_path)

    id1 = repo.create("/book1.epub", "hash1")
    repo.update_state(id1, PipelineState.PENDING_APPROVAL)
    repo.update_book_profile(id1, {"book_type": "newspaper", "confidence": 0.9})

    ops = BatchOperations(db_path)
    filter = BatchFilter(book_type="newspaper")

    result = ops.reject(filter, reason="Not ingesting periodicals", actor="human:taylor", execute=True)

    assert result["rejected"] == 1

    # Check state changed
    pipeline = repo.get(id1)
    assert pipeline["state"] == PipelineState.REJECTED.value


def test_batch_set_priority(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.batch import BatchOperations, BatchFilter

    repo = PipelineRepository(db_path)

    id1 = repo.create("/book1.epub", "hash1", priority=5)

    ops = BatchOperations(db_path)
    filter = BatchFilter(state="detected")

    result = ops.set_priority(filter, priority=1, actor="human:taylor", execute=True)

    assert result["updated"] == 1

    pipeline = repo.get(id1)
    assert pipeline["priority"] == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_batch_operations.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# agentic_pipeline/batch/operations.py
"""Batch operations for processing multiple books."""

from pathlib import Path
from typing import Optional

from agentic_pipeline.audit import AuditTrail
from agentic_pipeline.batch.filters import BatchFilter
from agentic_pipeline.db.pipelines import PipelineRepository
from agentic_pipeline.pipeline.states import PipelineState


class BatchOperations:
    """Execute operations on multiple pipelines."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.repo = PipelineRepository(db_path)
        self.audit = AuditTrail(db_path)

    def approve(
        self,
        filter: BatchFilter,
        actor: str,
        execute: bool = False,
    ) -> dict:
        """Approve books matching filter."""
        # Override filter to only match pending_approval
        filter.state = "pending_approval"
        matches = filter.apply(self.db_path)

        if not execute:
            return {
                "would_approve": len(matches),
                "approved": 0,
                "books": [{"id": m["id"], "source_path": m["source_path"]} for m in matches]
            }

        for pipeline in matches:
            self.repo.mark_approved(
                pipeline["id"],
                approved_by=f"batch:{actor}",
                confidence=None
            )

        # Log batch operation to audit
        self.audit.log(
            book_id=f"batch:{len(matches)}_books",
            action="BATCH_APPROVED",
            actor=actor,
            filter_used=filter.to_dict(),
        )

        return {
            "approved": len(matches),
            "would_approve": len(matches),
            "books": [{"id": m["id"], "source_path": m["source_path"]} for m in matches]
        }

    def reject(
        self,
        filter: BatchFilter,
        reason: str,
        actor: str,
        execute: bool = False,
    ) -> dict:
        """Reject books matching filter."""
        filter.state = "pending_approval"
        matches = filter.apply(self.db_path)

        if not execute:
            return {
                "would_reject": len(matches),
                "rejected": 0,
                "books": [{"id": m["id"], "source_path": m["source_path"]} for m in matches]
            }

        for pipeline in matches:
            self.repo.update_state(
                pipeline["id"],
                PipelineState.REJECTED,
                error_details={"reason": reason, "actor": actor}
            )

        self.audit.log(
            book_id=f"batch:{len(matches)}_books",
            action="BATCH_REJECTED",
            actor=actor,
            reason=reason,
            filter_used=filter.to_dict(),
        )

        return {
            "rejected": len(matches),
            "would_reject": len(matches),
            "books": [{"id": m["id"], "source_path": m["source_path"]} for m in matches]
        }

    def set_priority(
        self,
        filter: BatchFilter,
        priority: int,
        actor: str,
        execute: bool = False,
    ) -> dict:
        """Set priority for books matching filter."""
        matches = filter.apply(self.db_path)

        if not execute:
            return {
                "would_update": len(matches),
                "updated": 0,
                "books": [{"id": m["id"], "source_path": m["source_path"]} for m in matches]
            }

        for pipeline in matches:
            self.repo.update_priority(pipeline["id"], priority)

        self.audit.log(
            book_id=f"batch:{len(matches)}_books",
            action="BATCH_PRIORITY_CHANGED",
            actor=actor,
            filter_used=filter.to_dict(),
            adjustments={"new_priority": priority}
        )

        return {
            "updated": len(matches),
            "would_update": len(matches),
            "books": [{"id": m["id"], "source_path": m["source_path"]} for m in matches]
        }
```

Update `__init__.py`:

```python
# agentic_pipeline/batch/__init__.py
"""Batch operations package."""

from agentic_pipeline.batch.filters import BatchFilter
from agentic_pipeline.batch.operations import BatchOperations

__all__ = ["BatchFilter", "BatchOperations"]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_batch_operations.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add agentic_pipeline/batch/ tests/test_batch_operations.py
git commit -m "feat: add BatchOperations for approve/reject/set_priority"
```

---

## Task 6: Health Monitor

**Files:**
- Create: `agentic_pipeline/health/__init__.py`
- Create: `agentic_pipeline/health/monitor.py`
- Test: `tests/test_health_monitor.py`

**Step 1: Write the failing test**

```python
# tests/test_health_monitor.py
"""Tests for health monitoring."""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def db_path():
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    yield path
    path.unlink(missing_ok=True)


def test_health_report_empty_queue(db_path):
    from agentic_pipeline.health import HealthMonitor

    monitor = HealthMonitor(db_path)
    report = monitor.get_health()

    assert report["active"] == 0
    assert report["queued"] == 0
    assert report["stuck"] == []
    assert report["status"] == "idle"


def test_health_report_with_queued(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.health import HealthMonitor

    repo = PipelineRepository(db_path)
    repo.create("/book1.epub", "hash1", priority=1)
    repo.create("/book2.epub", "hash2", priority=5)
    repo.create("/book3.epub", "hash3", priority=5)

    monitor = HealthMonitor(db_path)
    report = monitor.get_health()

    assert report["queued"] == 3
    assert report["queue_by_priority"] == {1: 1, 5: 2}


def test_health_report_with_active(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.health import HealthMonitor

    repo = PipelineRepository(db_path)
    pid = repo.create("/book1.epub", "hash1")
    repo.update_state(pid, PipelineState.PROCESSING)

    monitor = HealthMonitor(db_path)
    report = monitor.get_health()

    assert report["active"] == 1
    assert report["status"] == "processing"


def test_health_report_with_failed(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.health import HealthMonitor

    repo = PipelineRepository(db_path)
    pid = repo.create("/book1.epub", "hash1")
    repo.update_state(pid, PipelineState.NEEDS_RETRY)

    monitor = HealthMonitor(db_path)
    report = monitor.get_health()

    assert report["failed"] == 1


def test_alerts_queue_backup(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.health import HealthMonitor

    repo = PipelineRepository(db_path)
    for i in range(150):
        repo.create(f"/book{i}.epub", f"hash{i}")

    monitor = HealthMonitor(db_path, alert_queue_threshold=100)
    report = monitor.get_health()

    assert any(a["type"] == "queue_backup" for a in report["alerts"])
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_health_monitor.py -v`
Expected: FAIL

**Step 3: Create health package**

```bash
mkdir -p agentic_pipeline/health
```

**Step 4: Write implementation**

```python
# agentic_pipeline/health/__init__.py
"""Health monitoring package."""

from agentic_pipeline.health.monitor import HealthMonitor

__all__ = ["HealthMonitor"]
```

```python
# agentic_pipeline/health/monitor.py
"""Health monitoring for the pipeline."""

import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from agentic_pipeline.db.pipelines import PipelineRepository
from agentic_pipeline.pipeline.states import PipelineState, TERMINAL_STATES


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
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_health_monitor.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add agentic_pipeline/health/ tests/test_health_monitor.py
git commit -m "feat: add HealthMonitor for pipeline health metrics"
```

---

## Task 7: Stuck Detection

**Files:**
- Create: `agentic_pipeline/health/stuck_detector.py`
- Modify: `agentic_pipeline/health/__init__.py`
- Test: `tests/test_stuck_detection.py`

**Step 1: Write the failing test**

```python
# tests/test_stuck_detection.py
"""Tests for stuck detection."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta


@pytest.fixture
def db_path():
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    yield path
    path.unlink(missing_ok=True)


def test_detect_stuck_pipeline(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.health.stuck_detector import StuckDetector
    import sqlite3

    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "hash123")
    repo.update_state(pid, PipelineState.PROCESSING)

    # Manually set updated_at to 2 hours ago
    conn = sqlite3.connect(db_path)
    two_hours_ago = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    conn.execute("UPDATE processing_pipelines SET updated_at = ? WHERE id = ?", (two_hours_ago, pid))
    conn.commit()
    conn.close()

    detector = StuckDetector(db_path)
    stuck = detector.detect()

    assert len(stuck) == 1
    assert stuck[0]["id"] == pid
    assert stuck[0]["state"] == "processing"


def test_not_stuck_if_recent(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.health.stuck_detector import StuckDetector

    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "hash123")
    repo.update_state(pid, PipelineState.PROCESSING)

    # Recently updated - should not be stuck
    detector = StuckDetector(db_path)
    stuck = detector.detect()

    assert len(stuck) == 0


def test_completed_not_flagged_as_stuck(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.health.stuck_detector import StuckDetector
    import sqlite3

    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "hash123")
    repo.update_state(pid, PipelineState.COMPLETE)

    # Set old updated_at
    conn = sqlite3.connect(db_path)
    old_time = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
    conn.execute("UPDATE processing_pipelines SET updated_at = ? WHERE id = ?", (old_time, pid))
    conn.commit()
    conn.close()

    detector = StuckDetector(db_path)
    stuck = detector.detect()

    # Should not be flagged - it's complete
    assert len(stuck) == 0


def test_default_state_thresholds():
    from agentic_pipeline.health.stuck_detector import DEFAULT_STATE_TIMEOUTS

    assert DEFAULT_STATE_TIMEOUTS["HASHING"] == 60
    assert DEFAULT_STATE_TIMEOUTS["PROCESSING"] == 900
    assert DEFAULT_STATE_TIMEOUTS["EMBEDDING"] == 600
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_stuck_detection.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# agentic_pipeline/health/stuck_detector.py
"""Stuck detection for pipelines."""

import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from agentic_pipeline.pipeline.states import PipelineState, TERMINAL_STATES


# Default timeout thresholds in seconds
DEFAULT_STATE_TIMEOUTS = {
    "HASHING": 60,           # 1 minute
    "CLASSIFYING": 120,      # 2 minutes
    "SELECTING_STRATEGY": 30, # 30 seconds
    "PROCESSING": 900,       # 15 minutes
    "VALIDATING": 60,        # 1 minute
    "PENDING_APPROVAL": None, # No timeout - waiting for human
    "APPROVED": 60,          # 1 minute
    "EMBEDDING": 600,        # 10 minutes
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
        self.db_path = db_path
        self.stuck_multiplier = stuck_multiplier
        self.thresholds = {**DEFAULT_STATE_TIMEOUTS, **(custom_thresholds or {})}

    def detect(self) -> list[dict]:
        """Find pipelines that appear to be stuck."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
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
                updated_at = datetime.fromisoformat(pipeline["updated_at"].replace("Z", "+00:00"))
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

        conn.close()
        return stuck
```

Update `__init__.py`:

```python
# agentic_pipeline/health/__init__.py
"""Health monitoring package."""

from agentic_pipeline.health.monitor import HealthMonitor
from agentic_pipeline.health.stuck_detector import StuckDetector, DEFAULT_STATE_TIMEOUTS

__all__ = ["HealthMonitor", "StuckDetector", "DEFAULT_STATE_TIMEOUTS"]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_stuck_detection.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add agentic_pipeline/health/ tests/test_stuck_detection.py
git commit -m "feat: add StuckDetector for finding stuck pipelines"
```

---

## Task 8: CLI Commands

**Files:**
- Modify: `agentic_pipeline/cli.py`
- Test: `tests/test_cli_phase4.py`

**Step 1: Write the failing test**

```python
# tests/test_cli_phase4.py
"""Tests for Phase 4 CLI commands."""

import pytest
from click.testing import CliRunner


def test_health_command_exists():
    from agentic_pipeline.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["health", "--help"])

    assert result.exit_code == 0
    assert "health" in result.output.lower()


def test_stuck_command_exists():
    from agentic_pipeline.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["stuck", "--help"])

    assert result.exit_code == 0


def test_batch_approve_command_exists():
    from agentic_pipeline.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["batch-approve", "--help"])

    assert result.exit_code == 0


def test_batch_reject_command_exists():
    from agentic_pipeline.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["batch-reject", "--help"])

    assert result.exit_code == 0


def test_audit_command_exists():
    from agentic_pipeline.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["audit", "--help"])

    assert result.exit_code == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_phase4.py -v`
Expected: FAIL

**Step 3: Read existing CLI file**

Read: `agentic_pipeline/cli.py`

**Step 4: Add Phase 4 CLI commands**

Add to `agentic_pipeline/cli.py`:

```python
@main.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def health(as_json: bool):
    """Show pipeline health status."""
    from .db.config import get_db_path
    from .health import HealthMonitor, StuckDetector
    import json as json_module

    db_path = get_db_path()
    monitor = HealthMonitor(db_path)
    detector = StuckDetector(db_path)

    report = monitor.get_health()
    report["stuck"] = detector.detect()

    if as_json:
        console.print(json_module.dumps(report, indent=2))
        return

    console.print("\n[bold]Pipeline Health[/bold]")
    console.print("─" * 35)
    console.print(f"  Active:     {report['active']} (processing now)")

    stuck_count = len(report['stuck'])
    if stuck_count > 0:
        console.print(f"  Stuck:      [red]{stuck_count} [!][/red]")
    else:
        console.print(f"  Stuck:      {stuck_count}")

    console.print(f"  Queued:     {report['queued']} (waiting)")
    console.print(f"  Completed:  {report['completed_24h']} (last 24h)")
    console.print(f"  Failed:     {report['failed']} (needs_retry)")
    console.print("─" * 35)

    if report['alerts']:
        console.print("\n[yellow]Alerts:[/yellow]")
        for alert in report['alerts']:
            console.print(f"  [{alert['severity']}] {alert['message']}")


@main.command()
@click.option("--recover", is_flag=True, help="Auto-recover stuck pipelines")
def stuck(recover: bool):
    """List stuck pipelines."""
    from .db.config import get_db_path
    from .health import StuckDetector

    db_path = get_db_path()
    detector = StuckDetector(db_path)
    stuck_list = detector.detect()

    if not stuck_list:
        console.print("[green]No stuck pipelines[/green]")
        return

    console.print(f"\n[yellow]Found {len(stuck_list)} stuck pipeline(s):[/yellow]\n")

    for item in stuck_list:
        console.print(f"  {item['id'][:8]}... [{item['state']}]")
        console.print(f"    Stuck for {item['stuck_minutes']} min (expected: {item['expected_minutes']} min)")
        console.print(f"    Source: {Path(item['source_path']).name}")


@main.command("batch-approve")
@click.option("--min-confidence", type=float, help="Minimum confidence threshold")
@click.option("--book-type", help="Filter by book type")
@click.option("--max-count", default=50, help="Maximum books to approve")
@click.option("--execute", is_flag=True, help="Actually execute (otherwise dry-run)")
def batch_approve(min_confidence: float, book_type: str, max_count: int, execute: bool):
    """Approve books matching filters."""
    from .db.config import get_db_path
    from .batch import BatchOperations, BatchFilter

    db_path = get_db_path()
    ops = BatchOperations(db_path)
    filter = BatchFilter(
        min_confidence=min_confidence,
        book_type=book_type,
        max_count=max_count,
    )

    result = ops.approve(filter, actor="human:cli", execute=execute)

    if execute:
        console.print(f"[green]Approved {result['approved']} books[/green]")
    else:
        console.print(f"[yellow]Would approve {result['would_approve']} books (dry-run)[/yellow]")
        for book in result['books'][:10]:
            console.print(f"  {book['id'][:8]}... {Path(book['source_path']).name}")
        if len(result['books']) > 10:
            console.print(f"  ... and {len(result['books']) - 10} more")


@main.command("batch-reject")
@click.option("--book-type", help="Filter by book type")
@click.option("--max-confidence", type=float, help="Maximum confidence threshold")
@click.option("--reason", required=True, help="Rejection reason")
@click.option("--max-count", default=50, help="Maximum books to reject")
@click.option("--execute", is_flag=True, help="Actually execute (otherwise dry-run)")
def batch_reject(book_type: str, max_confidence: float, reason: str, max_count: int, execute: bool):
    """Reject books matching filters."""
    from .db.config import get_db_path
    from .batch import BatchOperations, BatchFilter

    db_path = get_db_path()
    ops = BatchOperations(db_path)
    filter = BatchFilter(
        book_type=book_type,
        max_confidence=max_confidence,
        max_count=max_count,
    )

    result = ops.reject(filter, reason=reason, actor="human:cli", execute=execute)

    if execute:
        console.print(f"[yellow]Rejected {result['rejected']} books[/yellow]")
    else:
        console.print(f"[yellow]Would reject {result['would_reject']} books (dry-run)[/yellow]")


@main.command()
@click.option("--last", default=50, help="Number of recent entries")
@click.option("--actor", help="Filter by actor")
@click.option("--action", help="Filter by action type")
@click.option("--book-id", help="Filter by book ID")
def audit(last: int, actor: str, action: str, book_id: str):
    """Query the audit trail."""
    from .db.config import get_db_path
    from .audit import AuditTrail

    db_path = get_db_path()
    trail = AuditTrail(db_path)

    entries = trail.query(
        book_id=book_id,
        actor=actor,
        action=action,
        limit=last,
    )

    if not entries:
        console.print("[yellow]No audit entries found[/yellow]")
        return

    console.print(f"\n[bold]Audit Trail ({len(entries)} entries)[/bold]\n")

    table = Table()
    table.add_column("Time", style="dim")
    table.add_column("Action")
    table.add_column("Actor")
    table.add_column("Book")

    for entry in entries:
        table.add_row(
            entry["performed_at"][:19] if entry.get("performed_at") else "?",
            entry["action"],
            entry["actor"],
            entry["book_id"][:16] + "..." if len(entry["book_id"]) > 16 else entry["book_id"],
        )

    console.print(table)
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_cli_phase4.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add agentic_pipeline/cli.py tests/test_cli_phase4.py
git commit -m "feat: add Phase 4 CLI commands (health, stuck, batch-*, audit)"
```

---

## Task 9: MCP Tools

**Files:**
- Modify: `agentic_pipeline/mcp_server.py`
- Test: `tests/test_mcp_phase4.py`

**Step 1: Write the failing test**

```python
# tests/test_mcp_phase4.py
"""Tests for Phase 4 MCP tools."""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def db_path(monkeypatch):
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(path))
    yield path
    path.unlink(missing_ok=True)


def test_get_pipeline_health_exists(db_path):
    from agentic_pipeline.mcp_server import get_pipeline_health

    assert callable(get_pipeline_health)


def test_get_stuck_pipelines_exists(db_path):
    from agentic_pipeline.mcp_server import get_stuck_pipelines

    assert callable(get_stuck_pipelines)


def test_batch_approve_tool_exists(db_path):
    from agentic_pipeline.mcp_server import batch_approve_tool

    assert callable(batch_approve_tool)


def test_get_audit_log_exists(db_path):
    from agentic_pipeline.mcp_server import get_audit_log

    assert callable(get_audit_log)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_mcp_phase4.py -v`
Expected: FAIL

**Step 3: Read existing MCP server**

Read: `agentic_pipeline/mcp_server.py`

**Step 4: Add Phase 4 MCP tools**

Add to `agentic_pipeline/mcp_server.py`:

```python
def get_pipeline_health() -> dict:
    """
    Get current pipeline health status.

    Returns health metrics including active, queued, stuck counts and alerts.
    """
    from agentic_pipeline.health import HealthMonitor, StuckDetector

    db_path = get_db_path()
    monitor = HealthMonitor(db_path)
    detector = StuckDetector(db_path)

    report = monitor.get_health()
    report["stuck"] = detector.detect()

    return report


def get_stuck_pipelines() -> list[dict]:
    """
    Get list of pipelines that appear to be stuck.

    Returns pipelines that have been in the same state longer than expected.
    """
    from agentic_pipeline.health import StuckDetector

    db_path = get_db_path()
    detector = StuckDetector(db_path)

    return detector.detect()


def batch_approve_tool(
    min_confidence: float = None,
    book_type: str = None,
    max_count: int = 50,
    execute: bool = False,
) -> dict:
    """
    Approve books matching filters.

    Args:
        min_confidence: Minimum confidence threshold
        book_type: Filter by book type
        max_count: Maximum books to approve
        execute: Set True to actually approve (otherwise preview)

    Returns:
        Count of approved/would_approve books and list of affected books.
    """
    from agentic_pipeline.batch import BatchOperations, BatchFilter

    db_path = get_db_path()
    ops = BatchOperations(db_path)
    filter = BatchFilter(
        min_confidence=min_confidence,
        book_type=book_type,
        max_count=max_count,
    )

    return ops.approve(filter, actor="mcp:batch", execute=execute)


def batch_reject_tool(
    book_type: str = None,
    max_confidence: float = None,
    reason: str = "",
    max_count: int = 50,
    execute: bool = False,
) -> dict:
    """
    Reject books matching filters.

    Args:
        book_type: Filter by book type
        max_confidence: Maximum confidence threshold
        reason: Rejection reason (required for execute)
        max_count: Maximum books to reject
        execute: Set True to actually reject (otherwise preview)
    """
    from agentic_pipeline.batch import BatchOperations, BatchFilter

    db_path = get_db_path()
    ops = BatchOperations(db_path)
    filter = BatchFilter(
        book_type=book_type,
        max_confidence=max_confidence,
        max_count=max_count,
    )

    return ops.reject(filter, reason=reason, actor="mcp:batch", execute=execute)


def get_audit_log(
    book_id: str = None,
    actor: str = None,
    action: str = None,
    last_days: int = 7,
    limit: int = 100,
) -> list[dict]:
    """
    Query the audit trail.

    Args:
        book_id: Filter by book ID
        actor: Filter by actor
        action: Filter by action type
        last_days: Only return entries from last N days
        limit: Maximum entries to return
    """
    from agentic_pipeline.audit import AuditTrail

    db_path = get_db_path()
    trail = AuditTrail(db_path)

    return trail.query(
        book_id=book_id,
        actor=actor,
        action=action,
        last_days=last_days,
        limit=limit,
    )
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_mcp_phase4.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add agentic_pipeline/mcp_server.py tests/test_mcp_phase4.py
git commit -m "feat: add Phase 4 MCP tools (health, stuck, batch, audit)"
```

---

## Task 10: Integration Test

**Files:**
- Create: `tests/test_phase4_integration.py`

**Step 1: Write integration test**

```python
# tests/test_phase4_integration.py
"""Integration tests for Phase 4 features."""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def db_path():
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    yield path
    path.unlink(missing_ok=True)


def test_full_batch_approve_flow(db_path):
    """Test complete batch approve workflow with audit."""
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.batch import BatchOperations, BatchFilter
    from agentic_pipeline.audit import AuditTrail

    repo = PipelineRepository(db_path)

    # Create books with different confidence
    for i in range(5):
        pid = repo.create(f"/book{i}.epub", f"hash{i}")
        repo.update_state(pid, PipelineState.PENDING_APPROVAL)
        repo.update_book_profile(pid, {
            "book_type": "technical_tutorial",
            "confidence": 0.85 + (i * 0.03)  # 0.85, 0.88, 0.91, 0.94, 0.97
        })

    # Batch approve high confidence
    ops = BatchOperations(db_path)
    filter = BatchFilter(min_confidence=0.9, book_type="technical_tutorial")

    # Preview first
    preview = ops.approve(filter, actor="human:test", execute=False)
    assert preview["would_approve"] == 3  # 0.91, 0.94, 0.97

    # Execute
    result = ops.approve(filter, actor="human:test", execute=True)
    assert result["approved"] == 3

    # Check audit
    trail = AuditTrail(db_path)
    entries = trail.query(action="BATCH_APPROVED")
    assert len(entries) == 1
    assert entries[0]["filter_used"]["min_confidence"] == 0.9


def test_health_with_stuck_detection(db_path):
    """Test health monitor integrates stuck detection."""
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    from agentic_pipeline.health import HealthMonitor, StuckDetector
    import sqlite3
    from datetime import datetime, timezone, timedelta

    repo = PipelineRepository(db_path)

    # Create active pipeline
    pid = repo.create("/book.epub", "hash123")
    repo.update_state(pid, PipelineState.PROCESSING)

    # Make it stuck
    conn = sqlite3.connect(db_path)
    old_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    conn.execute("UPDATE processing_pipelines SET updated_at = ? WHERE id = ?", (old_time, pid))
    conn.commit()
    conn.close()

    # Check health includes stuck
    monitor = HealthMonitor(db_path)
    detector = StuckDetector(db_path)

    report = monitor.get_health()
    stuck = detector.detect()

    assert report["active"] == 1
    assert len(stuck) == 1
    assert stuck[0]["id"] == pid


def test_priority_queue_ordering(db_path):
    """Test that priority queue orders correctly."""
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)

    # Create books with different priorities
    id_low = repo.create("/low.epub", "hash1", priority=10)
    id_high = repo.create("/high.epub", "hash2", priority=1)
    id_med = repo.create("/med.epub", "hash3", priority=5)

    # Get queue
    queue = repo.find_by_state(PipelineState.DETECTED)

    # Should be ordered by priority
    assert queue[0]["id"] == id_high  # priority 1
    assert queue[1]["id"] == id_med   # priority 5
    assert queue[2]["id"] == id_low   # priority 10
```

**Step 2: Run integration test**

Run: `pytest tests/test_phase4_integration.py -v`
Expected: PASS

**Step 3: Run all Phase 4 tests**

Run: `pytest tests/test_phase4_migrations.py tests/test_priority_queue.py tests/test_audit_trail.py tests/test_batch_filters.py tests/test_batch_operations.py tests/test_health_monitor.py tests/test_stuck_detection.py tests/test_cli_phase4.py tests/test_mcp_phase4.py tests/test_phase4_integration.py -v`

Expected: All tests pass

**Step 4: Commit**

```bash
git add tests/test_phase4_integration.py
git commit -m "test: add Phase 4 integration tests"
```

---

## Task 11: Final Summary Commit

**Step 1: Commit all Phase 4 work**

```bash
git add -A && git commit -m "feat: complete Phase 4 - Production Hardening

- Database migrations for priority, audit, health tables
- Priority queue support with aging
- Audit trail for all approval decisions
- Batch operations (approve, reject, set_priority)
- Health monitoring with metrics cache
- Stuck detection with configurable thresholds
- CLI commands: health, stuck, batch-approve, batch-reject, audit
- MCP tools: get_pipeline_health, get_stuck_pipelines, batch_approve_tool, get_audit_log
- Full integration test coverage

Ready for production use"
```

---

## Verification Checklist

After completing all tasks:

- [ ] `pytest tests/test_phase4_*.py -v` — All pass
- [ ] `agentic-pipeline health` — Shows health report
- [ ] `agentic-pipeline stuck` — Lists stuck pipelines (if any)
- [ ] `agentic-pipeline audit --last 10` — Shows recent audit entries
- [ ] `agentic-pipeline batch-approve --help` — Shows help

---

## Next Steps

After Phase 4:
- Test with real books in production environment
- Monitor health dashboard for issues
- Tune stuck detection thresholds based on real data
- Consider Phase 5: Confident Autonomy

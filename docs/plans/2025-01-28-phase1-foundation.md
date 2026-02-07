# Phase 1: Foundation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Set up the agentic-pipeline package with database schema, strategy configs, basic state machine, and MCP approval tools.

**Architecture:** New Python package that wraps the existing book-ingestion-python CLI. State machine orchestrates flow, MCP tools enable Claude to approve/reject books. No LLM agents yet — manual classification for testing the pipeline.

**Tech Stack:** Python 3.12, SQLite, Click (CLI), FastMCP (MCP server)

---

## Prerequisites

- Working directory: `/path/to/book-mcp-server/.worktrees/agentic-pipeline`
- Existing ingestion pipeline: `/path/to/book-ingestion-python`
- Shared database: `/path/to/book-ingestion-python/data/library.db`

---

## Task 1: Package Scaffolding

**Files:**
- Create: `agentic_pipeline/__init__.py`
- Create: `agentic_pipeline/cli.py`
- Create: `pyproject.toml`
- Create: `requirements.txt`

**Step 1: Create package structure**

```bash
mkdir -p agentic_pipeline/{agents,pipeline,triggers,approval,autonomy,db,health}
mkdir -p config/{strategies,prompts}
mkdir -p tests/fixtures
```

**Step 2: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "agentic-pipeline"
version = "0.1.0"
description = "Agentic AI layer for book ingestion pipeline"
requires-python = ">=3.12"
dependencies = [
    "click>=8.0",
    "rich>=13.0",
    "watchdog>=3.0",
    "mcp>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
]

[project.scripts]
agentic-pipeline = "agentic_pipeline.cli:main"
```

**Step 3: Create requirements.txt**

```text
click>=8.0
rich>=13.0
watchdog>=3.0
mcp>=1.0
pytest>=7.0
pytest-asyncio>=0.21
```

**Step 4: Create agentic_pipeline/__init__.py**

```python
"""Agentic Pipeline - AI-powered book ingestion orchestration."""

__version__ = "0.1.0"
```

**Step 5: Create placeholder CLI**

```python
# agentic_pipeline/cli.py
"""Agentic Pipeline CLI."""

import click
from rich.console import Console

console = Console()


@click.group()
def main():
    """Agentic Pipeline - AI-powered book ingestion."""
    pass


@main.command()
def version():
    """Show version."""
    from . import __version__
    console.print(f"agentic-pipeline v{__version__}")


if __name__ == "__main__":
    main()
```

**Step 6: Create empty __init__.py files**

```bash
touch agentic_pipeline/agents/__init__.py
touch agentic_pipeline/pipeline/__init__.py
touch agentic_pipeline/triggers/__init__.py
touch agentic_pipeline/approval/__init__.py
touch agentic_pipeline/autonomy/__init__.py
touch agentic_pipeline/db/__init__.py
touch agentic_pipeline/health/__init__.py
```

**Step 7: Verify structure**

Run: `find agentic_pipeline -type f -name "*.py" | sort`

Expected:
```
agentic_pipeline/__init__.py
agentic_pipeline/agents/__init__.py
agentic_pipeline/approval/__init__.py
agentic_pipeline/autonomy/__init__.py
agentic_pipeline/cli.py
agentic_pipeline/db/__init__.py
agentic_pipeline/health/__init__.py
agentic_pipeline/pipeline/__init__.py
agentic_pipeline/triggers/__init__.py
```

**Step 8: Test CLI runs**

Run: `python -m agentic_pipeline.cli version`

Expected: `agentic-pipeline v0.1.0`

**Step 9: Commit**

```bash
git add -A
git commit -m "feat: scaffold agentic-pipeline package structure"
```

---

## Task 2: Database Configuration

**Files:**
- Create: `agentic_pipeline/db/config.py`
- Create: `tests/test_db_config.py`

**Step 1: Write the failing test**

```python
# tests/test_db_config.py
"""Tests for database configuration."""

import pytest
from pathlib import Path


def test_get_db_path_returns_path():
    from agentic_pipeline.db.config import get_db_path

    path = get_db_path()
    assert isinstance(path, Path)
    assert path.name == "library.db"


def test_get_db_path_uses_env_override(monkeypatch):
    from agentic_pipeline.db.config import get_db_path

    monkeypatch.setenv("AGENTIC_PIPELINE_DB", "/custom/path/test.db")
    path = get_db_path()
    assert path == Path("/custom/path/test.db")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_db_config.py -v`

Expected: FAIL with "No module named 'agentic_pipeline.db.config'"

**Step 3: Write minimal implementation**

```python
# agentic_pipeline/db/config.py
"""Database configuration."""

import os
from pathlib import Path

# Default path to shared library.db
DEFAULT_DB_PATH = Path("/path/to/book-ingestion-python/data/library.db")


def get_db_path() -> Path:
    """Get the database path, with environment override support."""
    env_path = os.environ.get("AGENTIC_PIPELINE_DB")
    if env_path:
        return Path(env_path)
    return DEFAULT_DB_PATH
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_db_config.py -v`

Expected: 2 passed

**Step 5: Commit**

```bash
git add agentic_pipeline/db/config.py tests/test_db_config.py
git commit -m "feat: add database path configuration with env override"
```

---

## Task 3: Database Migrations - Pipeline Tables

**Files:**
- Create: `agentic_pipeline/db/migrations.py`
- Create: `tests/test_migrations.py`

**Step 1: Write the failing test**

```python
# tests/test_migrations.py
"""Tests for database migrations."""

import pytest
import sqlite3
import tempfile
from pathlib import Path


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    db_path.unlink(missing_ok=True)


def test_run_migrations_creates_pipeline_tables(temp_db):
    from agentic_pipeline.db.migrations import run_migrations

    run_migrations(temp_db)

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    # Check processing_pipelines table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='processing_pipelines'"
    )
    assert cursor.fetchone() is not None

    # Check pipeline_state_history table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='pipeline_state_history'"
    )
    assert cursor.fetchone() is not None

    conn.close()


def test_run_migrations_creates_autonomy_tables(temp_db):
    from agentic_pipeline.db.migrations import run_migrations

    run_migrations(temp_db)

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    # Check autonomy_config table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='autonomy_config'"
    )
    assert cursor.fetchone() is not None

    # Check it has default row
    cursor.execute("SELECT current_mode FROM autonomy_config WHERE id = 1")
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == "supervised"

    conn.close()


def test_run_migrations_is_idempotent(temp_db):
    from agentic_pipeline.db.migrations import run_migrations

    # Run twice - should not raise
    run_migrations(temp_db)
    run_migrations(temp_db)

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM autonomy_config")
    assert cursor.fetchone()[0] == 1  # Still only one row
    conn.close()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_migrations.py -v`

Expected: FAIL with "No module named 'agentic_pipeline.db.migrations'"

**Step 3: Write minimal implementation**

```python
# agentic_pipeline/db/migrations.py
"""Database migrations for agentic pipeline tables."""

import sqlite3
from pathlib import Path


MIGRATIONS = [
    # Pipeline tracking
    """
    CREATE TABLE IF NOT EXISTS processing_pipelines (
        id TEXT PRIMARY KEY,
        source_path TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        state TEXT NOT NULL,
        book_profile JSON,
        strategy_config JSON,
        validation_result JSON,
        retry_count INTEGER DEFAULT 0,
        max_retries INTEGER DEFAULT 2,
        error_log JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP,
        completed_at TIMESTAMP,
        timeout_at TIMESTAMP,
        last_heartbeat TIMESTAMP,
        priority INTEGER DEFAULT 5,
        approved_by TEXT,
        approval_confidence REAL,
        UNIQUE(content_hash)
    )
    """,

    # State history
    """
    CREATE TABLE IF NOT EXISTS pipeline_state_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pipeline_id TEXT NOT NULL,
        from_state TEXT,
        to_state TEXT NOT NULL,
        duration_ms INTEGER,
        agent_output JSON,
        error_details JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (pipeline_id) REFERENCES processing_pipelines(id)
    )
    """,

    # Strategy configurations
    """
    CREATE TABLE IF NOT EXISTS processing_strategies (
        name TEXT PRIMARY KEY,
        book_type TEXT NOT NULL,
        config JSON NOT NULL,
        version INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_active BOOLEAN DEFAULT TRUE
    )
    """,

    # Pipeline config
    """
    CREATE TABLE IF NOT EXISTS pipeline_config (
        key TEXT PRIMARY KEY,
        value JSON NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,

    # Audit trail
    """
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
        confidence_at_decision REAL,
        autonomy_mode TEXT,
        session_id TEXT,
        performed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,

    # Retention policies
    """
    CREATE TABLE IF NOT EXISTS audit_retention (
        audit_type TEXT PRIMARY KEY,
        retain_days INTEGER NOT NULL,
        last_cleanup TIMESTAMP
    )
    """,

    # Autonomy metrics
    """
    CREATE TABLE IF NOT EXISTS autonomy_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        period_start DATE NOT NULL,
        period_end DATE NOT NULL,
        total_processed INTEGER DEFAULT 0,
        auto_approved INTEGER DEFAULT 0,
        human_approved INTEGER DEFAULT 0,
        human_rejected INTEGER DEFAULT 0,
        human_adjusted INTEGER DEFAULT 0,
        avg_confidence_auto_approved REAL,
        avg_confidence_human_approved REAL,
        avg_confidence_human_rejected REAL,
        auto_approved_later_rolled_back INTEGER DEFAULT 0,
        human_approved_later_rolled_back INTEGER DEFAULT 0,
        metrics_by_type JSON,
        confidence_buckets JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(period_start, period_end)
    )
    """,

    # Autonomy feedback
    """
    CREATE TABLE IF NOT EXISTS autonomy_feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        book_id TEXT NOT NULL,
        pipeline_id TEXT,
        original_decision TEXT NOT NULL,
        original_confidence REAL,
        original_book_type TEXT,
        human_decision TEXT NOT NULL,
        human_adjustments JSON,
        feedback_category TEXT,
        feedback_notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,

    # Autonomy config (singleton)
    """
    CREATE TABLE IF NOT EXISTS autonomy_config (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        current_mode TEXT DEFAULT 'supervised',
        auto_approve_threshold REAL DEFAULT 0.95,
        auto_retry_threshold REAL DEFAULT 0.70,
        require_known_book_type BOOLEAN DEFAULT TRUE,
        require_zero_issues BOOLEAN DEFAULT TRUE,
        max_auto_approvals_per_day INTEGER DEFAULT 50,
        spot_check_percentage REAL DEFAULT 0.10,
        escape_hatch_active BOOLEAN DEFAULT FALSE,
        escape_hatch_activated_at TIMESTAMP,
        escape_hatch_reason TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
]

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_pipelines_state ON processing_pipelines(state)",
    "CREATE INDEX IF NOT EXISTS idx_pipelines_hash ON processing_pipelines(content_hash)",
    "CREATE INDEX IF NOT EXISTS idx_pipelines_priority ON processing_pipelines(priority, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_audit_book ON approval_audit(book_id)",
    "CREATE INDEX IF NOT EXISTS idx_audit_action ON approval_audit(action, performed_at)",
    "CREATE INDEX IF NOT EXISTS idx_feedback_category ON autonomy_feedback(feedback_category)",
]

DEFAULT_RETENTION = [
    ("approved", 365),
    ("rejected", 90),
    ("rollback", -1),  # -1 = forever
    ("adjusted", 365),
]


def run_migrations(db_path: Path) -> None:
    """Run all migrations to set up agentic pipeline tables."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Run table creation
    for migration in MIGRATIONS:
        cursor.execute(migration)

    # Run index creation
    for index in INDEXES:
        cursor.execute(index)

    # Insert default autonomy config if not exists
    cursor.execute(
        "INSERT OR IGNORE INTO autonomy_config (id) VALUES (1)"
    )

    # Insert default retention policies
    for audit_type, retain_days in DEFAULT_RETENTION:
        cursor.execute(
            "INSERT OR IGNORE INTO audit_retention (audit_type, retain_days) VALUES (?, ?)",
            (audit_type, retain_days)
        )

    conn.commit()
    conn.close()
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_migrations.py -v`

Expected: 3 passed

**Step 5: Commit**

```bash
git add agentic_pipeline/db/migrations.py tests/test_migrations.py
git commit -m "feat: add database migrations for pipeline and autonomy tables"
```

---

## Task 4: Pipeline States

**Files:**
- Create: `agentic_pipeline/pipeline/states.py`
- Create: `tests/test_states.py`

**Step 1: Write the failing test**

```python
# tests/test_states.py
"""Tests for pipeline states."""

import pytest


def test_pipeline_state_enum_has_required_states():
    from agentic_pipeline.pipeline.states import PipelineState

    required = [
        "DETECTED", "HASHING", "DUPLICATE", "CLASSIFYING",
        "SELECTING_STRATEGY", "PROCESSING", "VALIDATING",
        "PENDING_APPROVAL", "NEEDS_RETRY", "APPROVED",
        "EMBEDDING", "COMPLETE", "REJECTED", "ARCHIVED"
    ]

    for state in required:
        assert hasattr(PipelineState, state), f"Missing state: {state}"


def test_can_transition_allows_valid_transitions():
    from agentic_pipeline.pipeline.states import PipelineState, can_transition

    assert can_transition(PipelineState.DETECTED, PipelineState.HASHING)
    assert can_transition(PipelineState.HASHING, PipelineState.CLASSIFYING)
    assert can_transition(PipelineState.PENDING_APPROVAL, PipelineState.APPROVED)


def test_can_transition_blocks_invalid_transitions():
    from agentic_pipeline.pipeline.states import PipelineState, can_transition

    # Can't go backwards
    assert not can_transition(PipelineState.COMPLETE, PipelineState.DETECTED)
    # Can't skip steps
    assert not can_transition(PipelineState.DETECTED, PipelineState.COMPLETE)


def test_is_terminal_state():
    from agentic_pipeline.pipeline.states import PipelineState, is_terminal_state

    assert is_terminal_state(PipelineState.COMPLETE)
    assert is_terminal_state(PipelineState.REJECTED)
    assert is_terminal_state(PipelineState.ARCHIVED)
    assert is_terminal_state(PipelineState.DUPLICATE)

    assert not is_terminal_state(PipelineState.PROCESSING)
    assert not is_terminal_state(PipelineState.PENDING_APPROVAL)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_states.py -v`

Expected: FAIL with "No module named 'agentic_pipeline.pipeline.states'"

**Step 3: Write minimal implementation**

```python
# agentic_pipeline/pipeline/states.py
"""Pipeline state definitions and transitions."""

from enum import Enum


class PipelineState(Enum):
    """States a book can be in during processing."""

    DETECTED = "detected"
    HASHING = "hashing"
    DUPLICATE = "duplicate"
    CLASSIFYING = "classifying"
    SELECTING_STRATEGY = "selecting_strategy"
    PROCESSING = "processing"
    VALIDATING = "validating"
    PENDING_APPROVAL = "pending_approval"
    NEEDS_RETRY = "needs_retry"
    APPROVED = "approved"
    EMBEDDING = "embedding"
    COMPLETE = "complete"
    REJECTED = "rejected"
    ARCHIVED = "archived"


# Valid state transitions
TRANSITIONS = {
    PipelineState.DETECTED: {PipelineState.HASHING},
    PipelineState.HASHING: {PipelineState.CLASSIFYING, PipelineState.DUPLICATE},
    PipelineState.DUPLICATE: set(),  # Terminal
    PipelineState.CLASSIFYING: {PipelineState.SELECTING_STRATEGY, PipelineState.REJECTED},
    PipelineState.SELECTING_STRATEGY: {PipelineState.PROCESSING},
    PipelineState.PROCESSING: {PipelineState.VALIDATING, PipelineState.NEEDS_RETRY, PipelineState.REJECTED},
    PipelineState.VALIDATING: {PipelineState.PENDING_APPROVAL, PipelineState.NEEDS_RETRY, PipelineState.REJECTED},
    PipelineState.PENDING_APPROVAL: {PipelineState.APPROVED, PipelineState.REJECTED, PipelineState.NEEDS_RETRY},
    PipelineState.NEEDS_RETRY: {PipelineState.PROCESSING, PipelineState.REJECTED},
    PipelineState.APPROVED: {PipelineState.EMBEDDING},
    PipelineState.EMBEDDING: {PipelineState.COMPLETE, PipelineState.REJECTED},
    PipelineState.COMPLETE: {PipelineState.ARCHIVED},  # Can archive completed books
    PipelineState.REJECTED: {PipelineState.ARCHIVED},
    PipelineState.ARCHIVED: set(),  # Terminal
}

TERMINAL_STATES = {
    PipelineState.COMPLETE,
    PipelineState.REJECTED,
    PipelineState.ARCHIVED,
    PipelineState.DUPLICATE,
}


def can_transition(from_state: PipelineState, to_state: PipelineState) -> bool:
    """Check if a state transition is valid."""
    return to_state in TRANSITIONS.get(from_state, set())


def is_terminal_state(state: PipelineState) -> bool:
    """Check if a state is terminal (no further transitions possible)."""
    return state in TERMINAL_STATES
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_states.py -v`

Expected: 4 passed

**Step 5: Commit**

```bash
git add agentic_pipeline/pipeline/states.py tests/test_states.py
git commit -m "feat: add pipeline state definitions and transition rules"
```

---

## Task 5: Pipeline Repository (CRUD)

**Files:**
- Create: `agentic_pipeline/db/pipelines.py`
- Create: `tests/test_pipelines.py`

**Step 1: Write the failing test**

```python
# tests/test_pipelines.py
"""Tests for pipeline repository."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime


@pytest.fixture
def db_path():
    """Create a temporary database."""
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    yield path
    path.unlink(missing_ok=True)


def test_create_pipeline(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)

    pipeline_id = repo.create(
        source_path="/path/to/book.epub",
        content_hash="abc123"
    )

    assert pipeline_id is not None

    pipeline = repo.get(pipeline_id)
    assert pipeline["source_path"] == "/path/to/book.epub"
    assert pipeline["content_hash"] == "abc123"
    assert pipeline["state"] == PipelineState.DETECTED.value


def test_update_state(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)
    pipeline_id = repo.create("/path/to/book.epub", "abc123")

    repo.update_state(pipeline_id, PipelineState.HASHING)

    pipeline = repo.get(pipeline_id)
    assert pipeline["state"] == PipelineState.HASHING.value


def test_find_by_hash(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository

    repo = PipelineRepository(db_path)
    pipeline_id = repo.create("/path/to/book.epub", "unique_hash_123")

    found = repo.find_by_hash("unique_hash_123")
    assert found is not None
    assert found["id"] == pipeline_id

    not_found = repo.find_by_hash("nonexistent")
    assert not_found is None


def test_list_pending_approval(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)

    # Create pipelines in various states
    id1 = repo.create("/book1.epub", "hash1")
    id2 = repo.create("/book2.epub", "hash2")
    id3 = repo.create("/book3.epub", "hash3")

    repo.update_state(id1, PipelineState.PENDING_APPROVAL)
    repo.update_state(id2, PipelineState.PENDING_APPROVAL)
    repo.update_state(id3, PipelineState.COMPLETE)

    pending = repo.list_pending_approval()
    assert len(pending) == 2
    assert all(p["state"] == PipelineState.PENDING_APPROVAL.value for p in pending)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipelines.py -v`

Expected: FAIL with "No module named 'agentic_pipeline.db.pipelines'"

**Step 3: Write minimal implementation**

```python
# agentic_pipeline/db/pipelines.py
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
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pipelines.py -v`

Expected: 4 passed

**Step 5: Commit**

```bash
git add agentic_pipeline/db/pipelines.py tests/test_pipelines.py
git commit -m "feat: add pipeline repository with CRUD operations"
```

---

## Task 6: Strategy Configurations

**Files:**
- Create: `config/strategies/technical_tutorial_v1.json`
- Create: `config/strategies/periodical_v1.json`
- Create: `config/strategies/narrative_v1.json`
- Create: `config/strategies/conservative_v1.json`
- Create: `agentic_pipeline/pipeline/strategy.py`
- Create: `tests/test_strategy.py`

**Step 1: Create strategy JSON files**

```json
// config/strategies/technical_tutorial_v1.json
{
  "name": "technical_tutorial_v1",
  "book_type": "technical_tutorial",
  "version": 1,
  "chapter_detection": {
    "method": "toc_with_explicit_fallback",
    "min_words_per_chapter": 1000,
    "max_words_per_chapter": 25000,
    "preserve_code_blocks": true
  },
  "text_cleaning": {
    "remove_headers_footers": true,
    "normalize_whitespace": true,
    "preserve_formatting": ["code", "lists", "tables"]
  },
  "section_splitting": {
    "enabled": true,
    "max_tokens_per_section": 15000
  },
  "quality_thresholds": {
    "min_chapters": 3,
    "max_chapters": 50,
    "min_avg_chapter_words": 500,
    "code_block_detection_required": true
  }
}
```

```json
// config/strategies/periodical_v1.json
{
  "name": "periodical_v1",
  "book_type": "periodical",
  "version": 1,
  "chapter_detection": {
    "method": "article_boundaries",
    "min_words_per_chapter": 200,
    "max_words_per_chapter": 10000,
    "detect_bylines": true,
    "detect_datelines": true
  },
  "text_cleaning": {
    "remove_headers_footers": true,
    "remove_advertisements": true,
    "normalize_whitespace": true
  },
  "section_splitting": {
    "enabled": false
  },
  "quality_thresholds": {
    "min_chapters": 1,
    "max_chapters": 100,
    "min_avg_chapter_words": 100
  }
}
```

```json
// config/strategies/narrative_v1.json
{
  "name": "narrative_v1",
  "book_type": "narrative_nonfiction",
  "version": 1,
  "chapter_detection": {
    "method": "explicit_with_semantic_fallback",
    "min_words_per_chapter": 2000,
    "max_words_per_chapter": 50000
  },
  "text_cleaning": {
    "remove_headers_footers": true,
    "normalize_whitespace": true,
    "preserve_formatting": ["quotes"]
  },
  "section_splitting": {
    "enabled": true,
    "max_tokens_per_section": 15000
  },
  "quality_thresholds": {
    "min_chapters": 3,
    "max_chapters": 40,
    "min_avg_chapter_words": 1000
  }
}
```

```json
// config/strategies/conservative_v1.json
{
  "name": "conservative_v1",
  "book_type": "unknown",
  "version": 1,
  "chapter_detection": {
    "method": "toc_only",
    "min_words_per_chapter": 500,
    "max_words_per_chapter": 30000
  },
  "text_cleaning": {
    "remove_headers_footers": true,
    "normalize_whitespace": true
  },
  "section_splitting": {
    "enabled": true,
    "max_tokens_per_section": 15000
  },
  "quality_thresholds": {
    "min_chapters": 1,
    "max_chapters": 100,
    "min_avg_chapter_words": 200
  },
  "flags_for_review": true
}
```

**Step 2: Write the failing test**

```python
# tests/test_strategy.py
"""Tests for strategy selection."""

import pytest
from pathlib import Path


def test_load_strategy():
    from agentic_pipeline.pipeline.strategy import StrategySelector

    selector = StrategySelector()
    strategy = selector.load_strategy("technical_tutorial_v1")

    assert strategy["name"] == "technical_tutorial_v1"
    assert strategy["book_type"] == "technical_tutorial"
    assert strategy["chapter_detection"]["preserve_code_blocks"] is True


def test_select_strategy_for_book_type():
    from agentic_pipeline.pipeline.strategy import StrategySelector

    selector = StrategySelector()

    # Technical tutorial with code
    profile = {"book_type": "technical_tutorial", "detected_features": {"has_code_blocks": True}}
    strategy = selector.select(profile)
    assert strategy["name"] == "technical_tutorial_v1"

    # Magazine/newspaper
    profile = {"book_type": "magazine"}
    strategy = selector.select(profile)
    assert strategy["name"] == "periodical_v1"

    # Unknown type
    profile = {"book_type": "unknown"}
    strategy = selector.select(profile)
    assert strategy["name"] == "conservative_v1"


def test_select_conservative_for_low_confidence():
    from agentic_pipeline.pipeline.strategy import StrategySelector

    selector = StrategySelector()

    # Low confidence should use conservative
    profile = {"book_type": "technical_tutorial", "confidence": 0.5}
    strategy = selector.select(profile)
    assert strategy["name"] == "conservative_v1"
```

**Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_strategy.py -v`

Expected: FAIL

**Step 4: Write minimal implementation**

```python
# agentic_pipeline/pipeline/strategy.py
"""Strategy selection for book processing."""

import json
from pathlib import Path
from typing import Optional

# Default strategies directory
STRATEGIES_DIR = Path(__file__).parent.parent.parent / "config" / "strategies"

# Book type to strategy mapping
STRATEGY_MAP = {
    "technical_tutorial": "technical_tutorial_v1",
    "technical_reference": "technical_tutorial_v1",
    "textbook": "technical_tutorial_v1",
    "narrative_nonfiction": "narrative_v1",
    "newspaper": "periodical_v1",
    "magazine": "periodical_v1",
    "research_collection": "technical_tutorial_v1",
    "unknown": "conservative_v1",
}

# Minimum confidence to use type-specific strategy
MIN_CONFIDENCE = 0.7


class StrategySelector:
    """Selects processing strategy based on book profile."""

    def __init__(self, strategies_dir: Optional[Path] = None):
        self.strategies_dir = strategies_dir or STRATEGIES_DIR
        self._cache = {}

    def load_strategy(self, name: str) -> dict:
        """Load a strategy configuration by name."""
        if name in self._cache:
            return self._cache[name]

        path = self.strategies_dir / f"{name}.json"
        if not path.exists():
            raise ValueError(f"Strategy not found: {name}")

        with open(path) as f:
            strategy = json.load(f)

        self._cache[name] = strategy
        return strategy

    def select(self, book_profile: dict) -> dict:
        """Select the best strategy for a book profile."""
        book_type = book_profile.get("book_type", "unknown")
        confidence = book_profile.get("confidence", 1.0)

        # Use conservative for low confidence
        if confidence < MIN_CONFIDENCE:
            return self.load_strategy("conservative_v1")

        # Map book type to strategy
        strategy_name = STRATEGY_MAP.get(book_type, "conservative_v1")
        return self.load_strategy(strategy_name)

    def list_strategies(self) -> list[str]:
        """List all available strategies."""
        return [p.stem for p in self.strategies_dir.glob("*.json")]
```

**Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_strategy.py -v`

Expected: 3 passed

**Step 6: Commit**

```bash
git add config/strategies/*.json agentic_pipeline/pipeline/strategy.py tests/test_strategy.py
git commit -m "feat: add strategy configurations and selector"
```

---

## Task 7: Approval Queue

**Files:**
- Create: `agentic_pipeline/approval/queue.py`
- Create: `tests/test_approval_queue.py`

**Step 1: Write the failing test**

```python
# tests/test_approval_queue.py
"""Tests for approval queue."""

import pytest
import tempfile
import json
from pathlib import Path


@pytest.fixture
def db_path():
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    yield path
    path.unlink(missing_ok=True)


def test_get_pending_returns_formatted_queue(db_path):
    from agentic_pipeline.approval.queue import ApprovalQueue
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    # Setup: create a pending pipeline
    repo = PipelineRepository(db_path)
    pid = repo.create("/path/to/book.epub", "hash123")
    repo.update_book_profile(pid, {
        "book_type": "technical_tutorial",
        "confidence": 0.92,
        "suggested_tags": ["ai", "python"]
    })
    repo.update_state(pid, PipelineState.PENDING_APPROVAL)

    # Test
    queue = ApprovalQueue(db_path)
    result = queue.get_pending()

    assert result["pending_count"] == 1
    assert len(result["books"]) == 1
    assert result["books"][0]["source_path"] == "/path/to/book.epub"
    assert result["books"][0]["book_type"] == "technical_tutorial"


def test_get_pending_calculates_stats(db_path):
    from agentic_pipeline.approval.queue import ApprovalQueue
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)

    # Create multiple pipelines
    for i, conf in enumerate([0.95, 0.85, 0.72]):
        pid = repo.create(f"/book{i}.epub", f"hash{i}")
        repo.update_book_profile(pid, {"confidence": conf, "book_type": "technical_tutorial"})
        repo.update_state(pid, PipelineState.PENDING_APPROVAL)

    queue = ApprovalQueue(db_path)
    result = queue.get_pending()

    assert result["pending_count"] == 3
    assert result["stats"]["high_confidence"] == 1  # >= 0.9
    assert result["stats"]["needs_attention"] == 1  # < 0.8
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_approval_queue.py -v`

Expected: FAIL

**Step 3: Write minimal implementation**

```python
# agentic_pipeline/approval/queue.py
"""Approval queue management."""

import json
from pathlib import Path
from typing import Optional

from agentic_pipeline.db.pipelines import PipelineRepository
from agentic_pipeline.pipeline.states import PipelineState


class ApprovalQueue:
    """Manages the queue of books pending approval."""

    def __init__(self, db_path: Path):
        self.repo = PipelineRepository(db_path)

    def get_pending(self, sort_by: str = "priority") -> dict:
        """Get all books pending approval with stats."""
        pipelines = self.repo.list_pending_approval()

        # Calculate stats
        high_confidence = 0
        needs_attention = 0
        total_confidence = 0

        books = []
        for p in pipelines:
            profile = json.loads(p.get("book_profile") or "{}")
            confidence = profile.get("confidence", 0)

            if confidence >= 0.9:
                high_confidence += 1
            elif confidence < 0.8:
                needs_attention += 1

            total_confidence += confidence

            books.append({
                "id": p["id"],
                "source_path": p["source_path"],
                "content_hash": p["content_hash"],
                "book_type": profile.get("book_type", "unknown"),
                "confidence": confidence,
                "suggested_tags": profile.get("suggested_tags", []),
                "created_at": p["created_at"],
                "priority": p["priority"],
            })

        avg_confidence = total_confidence / len(pipelines) if pipelines else 0

        return {
            "pending_count": len(pipelines),
            "stats": {
                "avg_confidence": round(avg_confidence, 2),
                "high_confidence": high_confidence,
                "needs_attention": needs_attention,
            },
            "books": books,
        }
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_approval_queue.py -v`

Expected: 2 passed

**Step 5: Commit**

```bash
git add agentic_pipeline/approval/queue.py tests/test_approval_queue.py
git commit -m "feat: add approval queue with stats calculation"
```

---

## Task 8: Approval Actions

**Files:**
- Create: `agentic_pipeline/approval/actions.py`
- Create: `tests/test_approval_actions.py`

**Step 1: Write the failing test**

```python
# tests/test_approval_actions.py
"""Tests for approval actions."""

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


def test_approve_book(db_path):
    from agentic_pipeline.approval.actions import approve_book
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "hash123")
    repo.update_state(pid, PipelineState.PENDING_APPROVAL)

    result = approve_book(db_path, pid, actor="human:taylor")

    assert result["success"] is True

    pipeline = repo.get(pid)
    assert pipeline["state"] == PipelineState.APPROVED.value
    assert pipeline["approved_by"] == "human:taylor"


def test_reject_book(db_path):
    from agentic_pipeline.approval.actions import reject_book
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "hash123")
    repo.update_state(pid, PipelineState.PENDING_APPROVAL)

    result = reject_book(db_path, pid, reason="Poor quality extraction", actor="human:taylor")

    assert result["success"] is True

    pipeline = repo.get(pid)
    assert pipeline["state"] == PipelineState.REJECTED.value


def test_approve_creates_audit_record(db_path):
    from agentic_pipeline.approval.actions import approve_book
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState
    import sqlite3

    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "hash123")
    repo.update_state(pid, PipelineState.PENDING_APPROVAL)

    approve_book(db_path, pid, actor="human:taylor")

    # Check audit record
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM approval_audit WHERE pipeline_id = ?", (pid,))
    audit = cursor.fetchone()
    conn.close()

    assert audit is not None
    assert audit["action"] == "approved"
    assert audit["actor"] == "human:taylor"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_approval_actions.py -v`

Expected: FAIL

**Step 3: Write minimal implementation**

```python
# agentic_pipeline/approval/actions.py
"""Approval actions - approve, reject, rollback."""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from agentic_pipeline.db.pipelines import PipelineRepository
from agentic_pipeline.pipeline.states import PipelineState


def _record_audit(
    db_path: Path,
    pipeline_id: str,
    book_id: Optional[str],
    action: str,
    actor: str,
    reason: Optional[str] = None,
    before_state: Optional[dict] = None,
    after_state: Optional[dict] = None,
    adjustments: Optional[dict] = None,
    confidence: Optional[float] = None,
) -> None:
    """Record an action in the audit trail."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO approval_audit
        (book_id, pipeline_id, action, actor, reason, before_state, after_state,
         adjustments, confidence_at_decision, autonomy_mode)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            book_id or "",
            pipeline_id,
            action,
            actor,
            reason,
            json.dumps(before_state) if before_state else None,
            json.dumps(after_state) if after_state else None,
            json.dumps(adjustments) if adjustments else None,
            confidence,
            "supervised",  # TODO: get from autonomy_config
        )
    )
    conn.commit()
    conn.close()


def approve_book(
    db_path: Path,
    pipeline_id: str,
    actor: str,
    adjustments: Optional[dict] = None,
) -> dict:
    """Approve a book for ingestion."""
    repo = PipelineRepository(db_path)
    pipeline = repo.get(pipeline_id)

    if not pipeline:
        return {"success": False, "error": f"Pipeline not found: {pipeline_id}"}

    if pipeline["state"] != PipelineState.PENDING_APPROVAL.value:
        return {"success": False, "error": f"Pipeline not in pending state: {pipeline['state']}"}

    # Get confidence from profile
    profile = json.loads(pipeline.get("book_profile") or "{}")
    confidence = profile.get("confidence")

    before_state = {"state": pipeline["state"]}

    # Mark as approved
    repo.mark_approved(pipeline_id, approved_by=actor, confidence=confidence)

    after_state = {"state": PipelineState.APPROVED.value}

    # Record audit
    _record_audit(
        db_path,
        pipeline_id,
        pipeline.get("book_id"),
        action="approved",
        actor=actor,
        before_state=before_state,
        after_state=after_state,
        adjustments=adjustments,
        confidence=confidence,
    )

    return {
        "success": True,
        "pipeline_id": pipeline_id,
        "state": PipelineState.APPROVED.value,
    }


def reject_book(
    db_path: Path,
    pipeline_id: str,
    reason: str,
    actor: str,
    retry: bool = False,
) -> dict:
    """Reject a book."""
    repo = PipelineRepository(db_path)
    pipeline = repo.get(pipeline_id)

    if not pipeline:
        return {"success": False, "error": f"Pipeline not found: {pipeline_id}"}

    before_state = {"state": pipeline["state"]}

    if retry:
        new_state = PipelineState.NEEDS_RETRY
    else:
        new_state = PipelineState.REJECTED

    repo.update_state(pipeline_id, new_state)

    after_state = {"state": new_state.value}

    # Record audit
    _record_audit(
        db_path,
        pipeline_id,
        pipeline.get("book_id"),
        action="rejected",
        actor=actor,
        reason=reason,
        before_state=before_state,
        after_state=after_state,
    )

    return {
        "success": True,
        "pipeline_id": pipeline_id,
        "state": new_state.value,
        "retry_queued": retry,
    }


def rollback_book(
    db_path: Path,
    pipeline_id: str,
    reason: str,
    actor: str,
) -> dict:
    """Rollback an approved/completed book."""
    repo = PipelineRepository(db_path)
    pipeline = repo.get(pipeline_id)

    if not pipeline:
        return {"success": False, "error": f"Pipeline not found: {pipeline_id}"}

    before_state = {"state": pipeline["state"]}

    repo.update_state(pipeline_id, PipelineState.ARCHIVED)

    after_state = {"state": PipelineState.ARCHIVED.value}

    # Record audit
    _record_audit(
        db_path,
        pipeline_id,
        pipeline.get("book_id"),
        action="rollback",
        actor=actor,
        reason=reason,
        before_state=before_state,
        after_state=after_state,
    )

    return {
        "success": True,
        "pipeline_id": pipeline_id,
        "state": PipelineState.ARCHIVED.value,
        "reason": reason,
    }
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_approval_actions.py -v`

Expected: 3 passed

**Step 5: Commit**

```bash
git add agentic_pipeline/approval/actions.py tests/test_approval_actions.py
git commit -m "feat: add approval actions (approve, reject, rollback)"
```

---

## Task 9: MCP Tools - Review Pending Books

**Files:**
- Create: `agentic_pipeline/mcp_server.py`
- Create: `tests/test_mcp_tools.py`

**Step 1: Write the failing test**

```python
# tests/test_mcp_tools.py
"""Tests for MCP tools."""

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


@pytest.fixture
def setup_pending_books(db_path):
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)
    pid = repo.create("/test/book.epub", "testhash")
    repo.update_book_profile(pid, {
        "book_type": "technical_tutorial",
        "confidence": 0.92
    })
    repo.update_state(pid, PipelineState.PENDING_APPROVAL)
    return pid


def test_review_pending_books_tool(db_path, setup_pending_books):
    from agentic_pipeline.mcp_server import review_pending_books

    result = review_pending_books(str(db_path))

    assert result["pending_count"] == 1
    assert len(result["books"]) == 1
    assert result["books"][0]["book_type"] == "technical_tutorial"


def test_approve_book_tool(db_path, setup_pending_books):
    from agentic_pipeline.mcp_server import approve_book_tool

    pid = setup_pending_books
    result = approve_book_tool(str(db_path), pid)

    assert result["success"] is True


def test_reject_book_tool(db_path, setup_pending_books):
    from agentic_pipeline.mcp_server import reject_book_tool

    pid = setup_pending_books
    result = reject_book_tool(str(db_path), pid, "Test rejection")

    assert result["success"] is True
    assert result["state"] == "rejected"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mcp_tools.py -v`

Expected: FAIL

**Step 3: Write minimal implementation**

```python
# agentic_pipeline/mcp_server.py
"""MCP server with approval tools."""

from pathlib import Path
from typing import Optional

from agentic_pipeline.approval.queue import ApprovalQueue
from agentic_pipeline.approval.actions import approve_book, reject_book, rollback_book
from agentic_pipeline.db.config import get_db_path


def review_pending_books(db_path: Optional[str] = None, sort_by: str = "priority") -> dict:
    """
    Get all books pending approval.

    Returns queue of books awaiting review with stats.
    """
    path = Path(db_path) if db_path else get_db_path()
    queue = ApprovalQueue(path)
    return queue.get_pending(sort_by=sort_by)


def approve_book_tool(
    db_path: Optional[str] = None,
    pipeline_id: str = "",
    actor: str = "human:unknown",
    adjustments: Optional[dict] = None,
) -> dict:
    """
    Approve a book for ingestion into the library.

    Args:
        pipeline_id: The pipeline ID to approve
        actor: Who is approving (e.g., "human:taylor", "auto:confident")
        adjustments: Optional adjustments to apply before ingestion
    """
    path = Path(db_path) if db_path else get_db_path()
    return approve_book(path, pipeline_id, actor, adjustments)


def reject_book_tool(
    db_path: Optional[str] = None,
    pipeline_id: str = "",
    reason: str = "",
    actor: str = "human:unknown",
    retry: bool = False,
) -> dict:
    """
    Reject a book.

    Args:
        pipeline_id: The pipeline ID to reject
        reason: Why the book is being rejected
        actor: Who is rejecting
        retry: If True, queue for retry with adjustments instead of permanent rejection
    """
    path = Path(db_path) if db_path else get_db_path()
    return reject_book(path, pipeline_id, reason, actor, retry)


def rollback_book_tool(
    db_path: Optional[str] = None,
    pipeline_id: str = "",
    reason: str = "",
    actor: str = "human:unknown",
) -> dict:
    """
    Rollback an approved/completed book from the library.

    Args:
        pipeline_id: The pipeline ID to rollback
        reason: Why the book is being rolled back
        actor: Who is performing the rollback
    """
    path = Path(db_path) if db_path else get_db_path()
    return rollback_book(path, pipeline_id, reason, actor)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_mcp_tools.py -v`

Expected: 3 passed

**Step 5: Commit**

```bash
git add agentic_pipeline/mcp_server.py tests/test_mcp_tools.py
git commit -m "feat: add MCP tools for review, approve, reject"
```

---

## Task 10: CLI Commands

**Files:**
- Modify: `agentic_pipeline/cli.py`
- Create: `tests/test_cli.py`

**Step 1: Update CLI with commands**

```python
# agentic_pipeline/cli.py
"""Agentic Pipeline CLI."""

import click
from rich.console import Console
from rich.table import Table
from pathlib import Path

console = Console()


@click.group()
def main():
    """Agentic Pipeline - AI-powered book ingestion."""
    pass


@main.command()
def version():
    """Show version."""
    from . import __version__
    console.print(f"agentic-pipeline v{__version__}")


@main.command()
def init():
    """Initialize the database with agentic pipeline tables."""
    from .db.config import get_db_path
    from .db.migrations import run_migrations

    db_path = get_db_path()
    console.print(f"[blue]Initializing database at {db_path}[/blue]")

    run_migrations(db_path)

    console.print("[green]Database initialized successfully![/green]")


@main.command()
def pending():
    """List books pending approval."""
    from .db.config import get_db_path
    from .approval.queue import ApprovalQueue

    db_path = get_db_path()
    queue = ApprovalQueue(db_path)
    result = queue.get_pending()

    if result["pending_count"] == 0:
        console.print("[yellow]No books pending approval[/yellow]")
        return

    console.print(f"\n[bold]Pending Approval: {result['pending_count']} books[/bold]")
    console.print(f"  High confidence (≥90%): {result['stats']['high_confidence']}")
    console.print(f"  Needs attention (<80%): {result['stats']['needs_attention']}")
    console.print()

    table = Table()
    table.add_column("ID", style="dim")
    table.add_column("Type")
    table.add_column("Confidence")
    table.add_column("Source")

    for book in result["books"]:
        conf = book["confidence"]
        conf_style = "green" if conf >= 0.9 else "yellow" if conf >= 0.8 else "red"
        table.add_row(
            book["id"][:8] + "...",
            book["book_type"],
            f"[{conf_style}]{conf:.0%}[/{conf_style}]",
            Path(book["source_path"]).name[:40],
        )

    console.print(table)


@main.command()
@click.argument("pipeline_id")
def approve(pipeline_id: str):
    """Approve a pending book."""
    from .db.config import get_db_path
    from .approval.actions import approve_book

    db_path = get_db_path()
    result = approve_book(db_path, pipeline_id, actor="human:cli")

    if result["success"]:
        console.print(f"[green]Approved: {pipeline_id}[/green]")
    else:
        console.print(f"[red]Failed: {result.get('error')}[/red]")


@main.command()
@click.argument("pipeline_id")
@click.option("--reason", "-r", required=True, help="Reason for rejection")
@click.option("--retry", is_flag=True, help="Queue for retry instead of permanent rejection")
def reject(pipeline_id: str, reason: str, retry: bool):
    """Reject a pending book."""
    from .db.config import get_db_path
    from .approval.actions import reject_book

    db_path = get_db_path()
    result = reject_book(db_path, pipeline_id, reason, actor="human:cli", retry=retry)

    if result["success"]:
        action = "Queued for retry" if retry else "Rejected"
        console.print(f"[yellow]{action}: {pipeline_id}[/yellow]")
    else:
        console.print(f"[red]Failed: {result.get('error')}[/red]")


@main.command()
def strategies():
    """List available processing strategies."""
    from .pipeline.strategy import StrategySelector

    selector = StrategySelector()
    names = selector.list_strategies()

    console.print("\n[bold]Available Strategies:[/bold]\n")
    for name in sorted(names):
        strategy = selector.load_strategy(name)
        console.print(f"  [cyan]{name}[/cyan]")
        console.print(f"    Book type: {strategy['book_type']}")
        console.print(f"    Version: {strategy.get('version', 1)}")
        console.print()


if __name__ == "__main__":
    main()
```

**Step 2: Write CLI test**

```python
# tests/test_cli.py
"""Tests for CLI commands."""

import pytest
from click.testing import CliRunner


def test_version_command():
    from agentic_pipeline.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["version"])

    assert result.exit_code == 0
    assert "agentic-pipeline v" in result.output


def test_strategies_command():
    from agentic_pipeline.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["strategies"])

    assert result.exit_code == 0
    assert "technical_tutorial_v1" in result.output
```

**Step 3: Run tests**

Run: `python -m pytest tests/test_cli.py -v`

Expected: 2 passed

**Step 4: Commit**

```bash
git add agentic_pipeline/cli.py tests/test_cli.py
git commit -m "feat: add CLI commands for pending, approve, reject, strategies"
```

---

## Task 11: Run All Tests & Final Commit

**Step 1: Run full test suite**

Run: `python -m pytest tests/ -v`

Expected: All tests pass (15+ tests)

**Step 2: Verify CLI works**

Run: `python -m agentic_pipeline.cli version`
Run: `python -m agentic_pipeline.cli strategies`

**Step 3: Final commit with feature summary**

```bash
git add -A
git commit -m "feat: complete Phase 1 foundation

- Package structure with modular design
- Database migrations for pipeline, audit, and autonomy tables
- Pipeline state machine with valid transitions
- Strategy configurations for book types
- Approval queue and actions
- MCP tools for review/approve/reject
- CLI commands for management

Ready for Phase 2: Classifier Agent"
```

---

## Summary

Phase 1 delivers:

| Component | Status |
|-----------|--------|
| Package scaffolding | ✅ |
| Database migrations | ✅ |
| Pipeline states | ✅ |
| Pipeline repository | ✅ |
| Strategy configs (4 types) | ✅ |
| Approval queue | ✅ |
| Approval actions | ✅ |
| MCP tools | ✅ |
| CLI commands | ✅ |

**Total: 11 tasks, ~50 steps, ~15 commits**

Next: Phase 2 - Classifier Agent (LLM-powered book type detection)

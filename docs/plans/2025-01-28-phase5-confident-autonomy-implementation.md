# Phase 5: Confident Autonomy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement graduated autonomy system that auto-approves high-confidence books while maintaining human oversight for edge cases.

**Architecture:** Autonomy package with config management, metrics collection, calibration engine, threshold calculator, and escape hatch. Integrates with existing approval flow via the orchestrator.

**Tech Stack:** Python 3.12+, SQLite, Click CLI, existing agentic_pipeline infrastructure

---

## Task 1: Database Migrations for Autonomy Tables

**Files:**
- Modify: `agentic_pipeline/db/migrations.py`
- Test: `tests/test_phase5_migrations.py`

**Step 1: Write the failing test**

```python
# tests/test_phase5_migrations.py
"""Tests for Phase 5 database migrations."""

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


def test_autonomy_config_table_exists(db_path):
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='autonomy_config'")
    result = cursor.fetchone()
    conn.close()

    assert result is not None


def test_autonomy_thresholds_table_exists(db_path):
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='autonomy_thresholds'")
    result = cursor.fetchone()
    conn.close()

    assert result is not None


def test_autonomy_feedback_table_exists(db_path):
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='autonomy_feedback'")
    result = cursor.fetchone()
    conn.close()

    assert result is not None


def test_spot_checks_table_exists(db_path):
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='spot_checks'")
    result = cursor.fetchone()
    conn.close()

    assert result is not None


def test_autonomy_config_has_default_row(db_path):
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT current_mode FROM autonomy_config WHERE id = 1")
    result = cursor.fetchone()
    conn.close()

    assert result is not None
    assert result[0] == "supervised"
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_phase5_migrations.py -v`
Expected: FAIL (tables don't exist)

**Step 3: Write minimal implementation**

Add to `agentic_pipeline/db/migrations.py` MIGRATIONS list:

```python
    # Autonomy config (singleton) - Phase 5
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

    # Per-type thresholds - Phase 5
    """
    CREATE TABLE IF NOT EXISTS autonomy_thresholds (
        book_type TEXT PRIMARY KEY,
        auto_approve_threshold REAL,
        sample_count INTEGER NOT NULL DEFAULT 0,
        measured_accuracy REAL,
        last_calculated TIMESTAMP,
        calibration_data JSON,
        manual_override REAL,
        override_reason TEXT
    )
    """,

    # Feedback log - Phase 5
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

    # Spot-check tracking - Phase 5
    """
    CREATE TABLE IF NOT EXISTS spot_checks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        book_id TEXT NOT NULL,
        pipeline_id TEXT,
        original_classification TEXT,
        original_confidence REAL,
        auto_approved_at TIMESTAMP,
        classification_correct BOOLEAN,
        quality_acceptable BOOLEAN,
        reviewer TEXT,
        notes TEXT,
        checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
```

Add to `run_migrations()` after autonomy_config insert:

```python
    # Insert default autonomy config if not exists
    cursor.execute(
        "INSERT OR IGNORE INTO autonomy_config (id, current_mode) VALUES (1, 'supervised')"
    )
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/test_phase5_migrations.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add agentic_pipeline/db/migrations.py tests/test_phase5_migrations.py
git commit -m "feat: add Phase 5 autonomy database tables"
```

---

## Task 2: Autonomy Config Manager

**Files:**
- Create: `agentic_pipeline/autonomy/config.py`
- Modify: `agentic_pipeline/autonomy/__init__.py`
- Test: `tests/test_autonomy_config.py`

**Step 1: Write the failing test**

```python
# tests/test_autonomy_config.py
"""Tests for autonomy config manager."""

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


def test_get_current_mode(db_path):
    from agentic_pipeline.autonomy import AutonomyConfig

    config = AutonomyConfig(db_path)
    assert config.get_mode() == "supervised"


def test_set_mode(db_path):
    from agentic_pipeline.autonomy import AutonomyConfig

    config = AutonomyConfig(db_path)
    config.set_mode("partial")
    assert config.get_mode() == "partial"


def test_escape_hatch_activate(db_path):
    from agentic_pipeline.autonomy import AutonomyConfig

    config = AutonomyConfig(db_path)
    config.set_mode("partial")
    config.activate_escape_hatch("Testing")

    assert config.is_escape_hatch_active()
    assert config.get_mode() == "supervised"  # Reverts to supervised


def test_escape_hatch_deactivate(db_path):
    from agentic_pipeline.autonomy import AutonomyConfig

    config = AutonomyConfig(db_path)
    config.activate_escape_hatch("Testing")
    config.deactivate_escape_hatch()

    assert not config.is_escape_hatch_active()


def test_should_auto_approve_when_supervised(db_path):
    from agentic_pipeline.autonomy import AutonomyConfig

    config = AutonomyConfig(db_path)
    assert config.should_auto_approve("technical_tutorial", 0.99) is False


def test_should_auto_approve_when_partial(db_path):
    from agentic_pipeline.autonomy import AutonomyConfig

    config = AutonomyConfig(db_path)
    config.set_mode("partial")

    # No threshold set yet, should not auto-approve
    assert config.should_auto_approve("technical_tutorial", 0.99) is False
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_autonomy_config.py -v`
Expected: FAIL (AutonomyConfig not found)

**Step 3: Write minimal implementation**

```python
# agentic_pipeline/autonomy/config.py
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
```

Update `agentic_pipeline/autonomy/__init__.py`:

```python
"""Autonomy package for graduated trust management."""

from agentic_pipeline.autonomy.config import AutonomyConfig

__all__ = ["AutonomyConfig"]
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/test_autonomy_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add agentic_pipeline/autonomy/ tests/test_autonomy_config.py
git commit -m "feat: add AutonomyConfig for mode and escape hatch management"
```

---

## Task 3: Metrics Collector

**Files:**
- Create: `agentic_pipeline/autonomy/metrics.py`
- Modify: `agentic_pipeline/autonomy/__init__.py`
- Test: `tests/test_autonomy_metrics.py`

**Step 1: Write the failing test**

```python
# tests/test_autonomy_metrics.py
"""Tests for autonomy metrics collection."""

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


def test_record_approval_decision(db_path):
    from agentic_pipeline.autonomy import MetricsCollector

    collector = MetricsCollector(db_path)
    collector.record_decision(
        book_id="book123",
        book_type="technical_tutorial",
        confidence=0.92,
        decision="approved",
        actor="human:taylor"
    )

    metrics = collector.get_metrics(days=1)
    assert metrics["total_processed"] == 1
    assert metrics["human_approved"] == 1


def test_record_auto_approval(db_path):
    from agentic_pipeline.autonomy import MetricsCollector

    collector = MetricsCollector(db_path)
    collector.record_decision(
        book_id="book123",
        book_type="technical_tutorial",
        confidence=0.95,
        decision="approved",
        actor="auto:high_confidence"
    )

    metrics = collector.get_metrics(days=1)
    assert metrics["auto_approved"] == 1


def test_get_accuracy_by_type(db_path):
    from agentic_pipeline.autonomy import MetricsCollector

    collector = MetricsCollector(db_path)

    # Record some decisions
    collector.record_decision("b1", "technical_tutorial", 0.92, "approved", "auto:test")
    collector.record_decision("b2", "technical_tutorial", 0.88, "approved", "auto:test")
    collector.record_decision("b3", "technical_tutorial", 0.85, "rejected", "human:taylor")  # Override

    accuracy = collector.get_accuracy_by_type("technical_tutorial")
    # 2 approved, 1 rejected (override) = 2/3 = 66.7%
    assert accuracy["sample_count"] == 3
    assert 0.65 < accuracy["accuracy"] < 0.68
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_autonomy_metrics.py -v`
Expected: FAIL (MetricsCollector not found)

**Step 3: Write minimal implementation**

```python
# agentic_pipeline/autonomy/metrics.py
"""Autonomy metrics collection."""

import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional


class MetricsCollector:
    """Collects and aggregates autonomy metrics."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def record_decision(
        self,
        book_id: str,
        book_type: str,
        confidence: float,
        decision: str,
        actor: str,
        pipeline_id: str = None,
        adjustments: dict = None,
    ) -> None:
        """Record a decision for metrics tracking."""
        conn = self._connect()
        cursor = conn.cursor()

        # Determine original decision type
        original_decision = "auto_approved" if actor.startswith("auto:") else "human_review"

        cursor.execute("""
            INSERT INTO autonomy_feedback
            (book_id, pipeline_id, original_decision, original_confidence, original_book_type,
             human_decision, human_adjustments, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            book_id,
            pipeline_id,
            original_decision,
            confidence,
            book_type,
            decision,
            json.dumps(adjustments) if adjustments else None,
            datetime.now(timezone.utc).isoformat()
        ))

        conn.commit()
        conn.close()

    def get_metrics(self, days: int = 30) -> dict:
        """Get aggregated metrics for the specified period."""
        conn = self._connect()
        cursor = conn.cursor()

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN original_decision = 'auto_approved' AND human_decision = 'approved' THEN 1 ELSE 0 END) as auto_approved,
                SUM(CASE WHEN original_decision = 'human_review' AND human_decision = 'approved' THEN 1 ELSE 0 END) as human_approved,
                SUM(CASE WHEN human_decision = 'rejected' THEN 1 ELSE 0 END) as human_rejected,
                SUM(CASE WHEN human_adjustments IS NOT NULL THEN 1 ELSE 0 END) as human_adjusted,
                AVG(CASE WHEN original_decision = 'auto_approved' THEN original_confidence END) as avg_conf_auto,
                AVG(CASE WHEN original_decision = 'human_review' THEN original_confidence END) as avg_conf_human
            FROM autonomy_feedback
            WHERE created_at > ?
        """, (cutoff,))

        row = cursor.fetchone()
        conn.close()

        return {
            "total_processed": row["total"] or 0,
            "auto_approved": row["auto_approved"] or 0,
            "human_approved": row["human_approved"] or 0,
            "human_rejected": row["human_rejected"] or 0,
            "human_adjusted": row["human_adjusted"] or 0,
            "avg_confidence_auto": row["avg_conf_auto"],
            "avg_confidence_human": row["avg_conf_human"],
        }

    def get_accuracy_by_type(self, book_type: str, days: int = 90) -> dict:
        """Get accuracy metrics for a specific book type."""
        conn = self._connect()
        cursor = conn.cursor()

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN human_decision = 'approved' THEN 1 ELSE 0 END) as correct
            FROM autonomy_feedback
            WHERE original_book_type = ?
            AND created_at > ?
        """, (book_type, cutoff))

        row = cursor.fetchone()
        conn.close()

        total = row["total"] or 0
        correct = row["correct"] or 0

        return {
            "book_type": book_type,
            "sample_count": total,
            "accuracy": correct / total if total > 0 else None,
        }
```

Update `agentic_pipeline/autonomy/__init__.py`:

```python
"""Autonomy package for graduated trust management."""

from agentic_pipeline.autonomy.config import AutonomyConfig
from agentic_pipeline.autonomy.metrics import MetricsCollector

__all__ = ["AutonomyConfig", "MetricsCollector"]
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/test_autonomy_metrics.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add agentic_pipeline/autonomy/ tests/test_autonomy_metrics.py
git commit -m "feat: add MetricsCollector for tracking autonomy decisions"
```

---

## Task 4: Calibration Engine

**Files:**
- Create: `agentic_pipeline/autonomy/calibration.py`
- Modify: `agentic_pipeline/autonomy/__init__.py`
- Test: `tests/test_calibration.py`

**Step 1: Write the failing test**

```python
# tests/test_calibration.py
"""Tests for calibration engine."""

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


def test_calculate_calibration_insufficient_data(db_path):
    from agentic_pipeline.autonomy import CalibrationEngine

    engine = CalibrationEngine(db_path)
    result = engine.calculate_calibration("technical_tutorial")

    assert result is None  # Not enough data


def test_calculate_calibration_with_data(db_path):
    from agentic_pipeline.autonomy import MetricsCollector, CalibrationEngine

    collector = MetricsCollector(db_path)

    # Record 60 decisions for technical_tutorial
    for i in range(60):
        # 55 correct (approved), 5 wrong (rejected)
        decision = "rejected" if i < 5 else "approved"
        confidence = 0.90 + (i % 10) * 0.01  # 0.90-0.99

        collector.record_decision(
            book_id=f"book{i}",
            book_type="technical_tutorial",
            confidence=confidence,
            decision=decision,
            actor="auto:test"
        )

    engine = CalibrationEngine(db_path, min_samples=50)
    result = engine.calculate_calibration("technical_tutorial")

    assert result is not None
    assert "accuracy" in result
    assert 0.90 < result["accuracy"] < 0.95  # ~55/60 = 91.7%


def test_calculate_threshold(db_path):
    from agentic_pipeline.autonomy import MetricsCollector, CalibrationEngine

    collector = MetricsCollector(db_path)

    # Record 100 high-confidence correct decisions
    for i in range(100):
        collector.record_decision(
            book_id=f"book{i}",
            book_type="technical_tutorial",
            confidence=0.92,
            decision="approved",
            actor="auto:test"
        )

    engine = CalibrationEngine(db_path, min_samples=50, target_accuracy=0.95)
    threshold = engine.calculate_threshold("technical_tutorial")

    assert threshold is not None
    assert threshold <= 0.92  # Can auto-approve at 92% since accuracy is 100%
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_calibration.py -v`
Expected: FAIL (CalibrationEngine not found)

**Step 3: Write minimal implementation**

```python
# agentic_pipeline/autonomy/calibration.py
"""Calibration engine for measuring accuracy vs confidence."""

import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional


class CalibrationEngine:
    """Calculates calibration metrics and thresholds."""

    def __init__(
        self,
        db_path: Path,
        min_samples: int = 50,
        target_accuracy: float = 0.95,
    ):
        self.db_path = db_path
        self.min_samples = min_samples
        self.target_accuracy = target_accuracy

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def calculate_calibration(self, book_type: str, days: int = 90) -> Optional[dict]:
        """Calculate calibration metrics for a book type."""
        conn = self._connect()
        cursor = conn.cursor()

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN human_decision = 'approved' THEN 1 ELSE 0 END) as correct,
                AVG(original_confidence) as avg_confidence
            FROM autonomy_feedback
            WHERE original_book_type = ?
            AND created_at > ?
        """, (book_type, cutoff))

        row = cursor.fetchone()
        conn.close()

        total = row["total"] or 0

        if total < self.min_samples:
            return None

        correct = row["correct"] or 0
        accuracy = correct / total

        return {
            "book_type": book_type,
            "sample_count": total,
            "accuracy": accuracy,
            "avg_confidence": row["avg_confidence"],
        }

    def calculate_threshold(self, book_type: str) -> Optional[float]:
        """Calculate the safe auto-approve threshold for a book type."""
        conn = self._connect()
        cursor = conn.cursor()

        cutoff = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()

        # Try progressively lower thresholds
        for threshold in [0.95, 0.92, 0.90, 0.88, 0.85, 0.82, 0.80]:
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN human_decision = 'approved' THEN 1 ELSE 0 END) as correct
                FROM autonomy_feedback
                WHERE original_book_type = ?
                AND original_confidence >= ?
                AND created_at > ?
            """, (book_type, threshold, cutoff))

            row = cursor.fetchone()
            total = row["total"] or 0

            if total < 10:  # Need at least 10 samples at this threshold
                continue

            correct = row["correct"] or 0
            accuracy = correct / total

            if accuracy >= self.target_accuracy:
                conn.close()
                return threshold

        conn.close()
        return None

    def update_thresholds(self) -> dict:
        """Recalculate and update all thresholds."""
        conn = self._connect()
        cursor = conn.cursor()

        # Get all book types with data
        cursor.execute("""
            SELECT DISTINCT original_book_type FROM autonomy_feedback
            WHERE original_book_type IS NOT NULL
        """)
        book_types = [row[0] for row in cursor.fetchall()]

        results = {}
        for book_type in book_types:
            calibration = self.calculate_calibration(book_type)
            threshold = self.calculate_threshold(book_type)

            if calibration:
                cursor.execute("""
                    INSERT OR REPLACE INTO autonomy_thresholds
                    (book_type, auto_approve_threshold, sample_count, measured_accuracy, last_calculated)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    book_type,
                    threshold,
                    calibration["sample_count"],
                    calibration["accuracy"],
                    datetime.now(timezone.utc).isoformat()
                ))

                results[book_type] = {
                    "threshold": threshold,
                    "sample_count": calibration["sample_count"],
                    "accuracy": calibration["accuracy"],
                }

        conn.commit()
        conn.close()
        return results
```

Update `agentic_pipeline/autonomy/__init__.py`:

```python
"""Autonomy package for graduated trust management."""

from agentic_pipeline.autonomy.config import AutonomyConfig
from agentic_pipeline.autonomy.metrics import MetricsCollector
from agentic_pipeline.autonomy.calibration import CalibrationEngine

__all__ = ["AutonomyConfig", "MetricsCollector", "CalibrationEngine"]
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/test_calibration.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add agentic_pipeline/autonomy/ tests/test_calibration.py
git commit -m "feat: add CalibrationEngine for threshold calculation"
```

---

## Task 5: Spot-Check System

**Files:**
- Create: `agentic_pipeline/autonomy/spot_check.py`
- Modify: `agentic_pipeline/autonomy/__init__.py`
- Test: `tests/test_spot_check.py`

**Step 1: Write the failing test**

```python
# tests/test_spot_check.py
"""Tests for spot-check system."""

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


def test_select_spot_check_empty(db_path):
    from agentic_pipeline.autonomy import SpotCheckManager

    manager = SpotCheckManager(db_path)
    selected = manager.select_for_review()

    assert selected == []


def test_select_spot_check_with_candidates(db_path):
    from agentic_pipeline.autonomy import MetricsCollector, SpotCheckManager

    collector = MetricsCollector(db_path)

    # Record 20 auto-approved books
    for i in range(20):
        collector.record_decision(
            book_id=f"book{i}",
            book_type="technical_tutorial",
            confidence=0.92,
            decision="approved",
            actor="auto:high_confidence"
        )

    manager = SpotCheckManager(db_path, sample_rate=0.10)
    selected = manager.select_for_review()

    assert len(selected) >= 2  # At least 10% of 20


def test_submit_spot_check_result(db_path):
    from agentic_pipeline.autonomy import SpotCheckManager

    manager = SpotCheckManager(db_path)
    manager.submit_result(
        book_id="book123",
        classification_correct=True,
        quality_acceptable=True,
        reviewer="human:taylor"
    )

    results = manager.get_results(days=1)
    assert len(results) == 1
    assert results[0]["book_id"] == "book123"
    assert results[0]["classification_correct"] is True
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_spot_check.py -v`
Expected: FAIL (SpotCheckManager not found)

**Step 3: Write minimal implementation**

```python
# agentic_pipeline/autonomy/spot_check.py
"""Spot-check system for ongoing verification."""

import random
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional


class SpotCheckManager:
    """Manages spot-check selection and results."""

    def __init__(self, db_path: Path, sample_rate: float = 0.10):
        self.db_path = db_path
        self.sample_rate = sample_rate

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def select_for_review(self, days: int = 7) -> list[dict]:
        """Select auto-approved books for spot-check review."""
        conn = self._connect()
        cursor = conn.cursor()

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        # Get auto-approved books not yet spot-checked
        cursor.execute("""
            SELECT f.book_id, f.original_book_type, f.original_confidence, f.created_at
            FROM autonomy_feedback f
            LEFT JOIN spot_checks s ON f.book_id = s.book_id
            WHERE f.original_decision = 'auto_approved'
            AND f.human_decision = 'approved'
            AND f.created_at > ?
            AND s.id IS NULL
        """, (cutoff,))

        candidates = [dict(row) for row in cursor.fetchall()]
        conn.close()

        if not candidates:
            return []

        # Calculate sample size
        sample_size = max(1, int(len(candidates) * self.sample_rate))

        # Random sample
        selected = random.sample(candidates, min(sample_size, len(candidates)))

        return selected

    def submit_result(
        self,
        book_id: str,
        classification_correct: bool,
        quality_acceptable: bool,
        reviewer: str,
        notes: str = None,
        pipeline_id: str = None,
    ) -> None:
        """Submit a spot-check review result."""
        conn = self._connect()
        cursor = conn.cursor()

        # Get original info
        cursor.execute("""
            SELECT original_book_type, original_confidence, created_at
            FROM autonomy_feedback
            WHERE book_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (book_id,))
        original = cursor.fetchone()

        cursor.execute("""
            INSERT INTO spot_checks
            (book_id, pipeline_id, original_classification, original_confidence,
             auto_approved_at, classification_correct, quality_acceptable, reviewer, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            book_id,
            pipeline_id,
            original["original_book_type"] if original else None,
            original["original_confidence"] if original else None,
            original["created_at"] if original else None,
            classification_correct,
            quality_acceptable,
            reviewer,
            notes,
        ))

        conn.commit()
        conn.close()

    def get_results(self, days: int = 30) -> list[dict]:
        """Get spot-check results for the specified period."""
        conn = self._connect()
        cursor = conn.cursor()

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        cursor.execute("""
            SELECT * FROM spot_checks
            WHERE checked_at > ?
            ORDER BY checked_at DESC
        """, (cutoff,))

        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    def get_accuracy_rate(self, days: int = 30) -> Optional[float]:
        """Get the accuracy rate from spot-checks."""
        conn = self._connect()
        cursor = conn.cursor()

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN classification_correct = 1 THEN 1 ELSE 0 END) as correct
            FROM spot_checks
            WHERE checked_at > ?
        """, (cutoff,))

        row = cursor.fetchone()
        conn.close()

        total = row["total"] or 0
        if total == 0:
            return None

        return row["correct"] / total
```

Update `agentic_pipeline/autonomy/__init__.py`:

```python
"""Autonomy package for graduated trust management."""

from agentic_pipeline.autonomy.config import AutonomyConfig
from agentic_pipeline.autonomy.metrics import MetricsCollector
from agentic_pipeline.autonomy.calibration import CalibrationEngine
from agentic_pipeline.autonomy.spot_check import SpotCheckManager

__all__ = ["AutonomyConfig", "MetricsCollector", "CalibrationEngine", "SpotCheckManager"]
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/test_spot_check.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add agentic_pipeline/autonomy/ tests/test_spot_check.py
git commit -m "feat: add SpotCheckManager for ongoing verification"
```

---

## Task 6: Phase 5 CLI Commands

**Files:**
- Modify: `agentic_pipeline/cli.py`
- Test: `tests/test_cli_phase5.py`

**Step 1: Write the failing test**

```python
# tests/test_cli_phase5.py
"""Tests for Phase 5 CLI commands."""

import pytest
from click.testing import CliRunner


def test_autonomy_status_command_exists():
    from agentic_pipeline.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["autonomy", "status", "--help"])

    assert result.exit_code == 0


def test_autonomy_enable_command_exists():
    from agentic_pipeline.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["autonomy", "enable", "--help"])

    assert result.exit_code == 0


def test_escape_hatch_command_exists():
    from agentic_pipeline.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["escape-hatch", "--help"])

    assert result.exit_code == 0


def test_spot_check_command_exists():
    from agentic_pipeline.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["spot-check", "--help"])

    assert result.exit_code == 0
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_cli_phase5.py -v`
Expected: FAIL (commands don't exist)

**Step 3: Write minimal implementation**

Add to `agentic_pipeline/cli.py` before `if __name__ == "__main__":`:

```python
# Phase 5: Autonomy Commands

@main.group()
def autonomy():
    """Manage autonomy settings."""
    pass


@autonomy.command("status")
def autonomy_status():
    """Show current autonomy mode and thresholds."""
    from .db.config import get_db_path
    from .autonomy import AutonomyConfig, MetricsCollector

    db_path = get_db_path()
    config = AutonomyConfig(db_path)
    collector = MetricsCollector(db_path)

    mode = config.get_mode()
    escape_active = config.is_escape_hatch_active()
    metrics = collector.get_metrics(days=30)

    console.print(f"\n[bold]Autonomy Status[/bold]")
    console.print("-" * 35)

    if escape_active:
        console.print(f"  Mode: [red]ESCAPE HATCH ACTIVE[/red]")
    else:
        console.print(f"  Mode: [cyan]{mode}[/cyan]")

    console.print(f"\n[bold]Last 30 Days:[/bold]")
    console.print(f"  Total processed: {metrics['total_processed']}")
    console.print(f"  Auto-approved:   {metrics['auto_approved']}")
    console.print(f"  Human approved:  {metrics['human_approved']}")
    console.print(f"  Human rejected:  {metrics['human_rejected']}")


@autonomy.command("enable")
@click.argument("mode", type=click.Choice(["partial", "confident"]))
def autonomy_enable(mode: str):
    """Enable an autonomy mode."""
    from .db.config import get_db_path
    from .autonomy import AutonomyConfig

    db_path = get_db_path()
    config = AutonomyConfig(db_path)

    if config.is_escape_hatch_active():
        console.print("[red]Cannot enable autonomy while escape hatch is active.[/red]")
        console.print("Run: agentic-pipeline autonomy resume")
        return

    config.set_mode(mode)
    console.print(f"[green]Autonomy mode set to: {mode}[/green]")


@autonomy.command("disable")
def autonomy_disable():
    """Disable autonomy (revert to supervised)."""
    from .db.config import get_db_path
    from .autonomy import AutonomyConfig

    db_path = get_db_path()
    config = AutonomyConfig(db_path)
    config.set_mode("supervised")
    console.print("[yellow]Autonomy disabled. All books require human review.[/yellow]")


@autonomy.command("resume")
def autonomy_resume():
    """Resume autonomy after escape hatch."""
    from .db.config import get_db_path
    from .autonomy import AutonomyConfig

    db_path = get_db_path()
    config = AutonomyConfig(db_path)

    if not config.is_escape_hatch_active():
        console.print("[yellow]Escape hatch is not active.[/yellow]")
        return

    config.deactivate_escape_hatch()
    console.print("[green]Escape hatch deactivated. Autonomy resumed.[/green]")


@main.command("escape-hatch")
@click.argument("reason")
def escape_hatch(reason: str):
    """Activate escape hatch - immediately revert to supervised mode."""
    from .db.config import get_db_path
    from .autonomy import AutonomyConfig

    db_path = get_db_path()
    config = AutonomyConfig(db_path)
    config.activate_escape_hatch(reason)

    console.print("\n[red bold]⚠️  ESCAPE HATCH ACTIVATED[/red bold]")
    console.print("\nAll autonomy disabled. Reverting to supervised mode.")
    console.print(f"Reason: {reason}")
    console.print("\nTo resume: agentic-pipeline autonomy resume")


@main.command("spot-check")
@click.option("--list", "list_pending", is_flag=True, help="List pending spot-checks")
def spot_check(list_pending: bool):
    """Start or manage spot-check reviews."""
    from .db.config import get_db_path
    from .autonomy import SpotCheckManager

    db_path = get_db_path()
    manager = SpotCheckManager(db_path)

    if list_pending:
        pending = manager.select_for_review()
        if not pending:
            console.print("[green]No spot-checks pending.[/green]")
            return

        console.print(f"\n[bold]Pending Spot-Checks ({len(pending)} books)[/bold]\n")
        for book in pending[:10]:
            console.print(f"  {book['book_id'][:8]}... [{book['original_book_type']}] {book['original_confidence']:.0%}")
    else:
        console.print("Use --list to see pending spot-checks")
        console.print("Interactive spot-check not yet implemented")
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/test_cli_phase5.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add agentic_pipeline/cli.py tests/test_cli_phase5.py
git commit -m "feat: add Phase 5 CLI commands (autonomy, escape-hatch, spot-check)"
```

---

## Task 7: Phase 5 MCP Tools

**Files:**
- Modify: `agentic_pipeline/mcp_server.py`
- Test: `tests/test_mcp_phase5.py`

**Step 1: Write the failing test**

```python
# tests/test_mcp_phase5.py
"""Tests for Phase 5 MCP tools."""

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


def test_get_autonomy_status_exists(db_path):
    from agentic_pipeline.mcp_server import get_autonomy_status

    assert callable(get_autonomy_status)


def test_set_autonomy_mode_exists(db_path):
    from agentic_pipeline.mcp_server import set_autonomy_mode

    assert callable(set_autonomy_mode)


def test_activate_escape_hatch_exists(db_path):
    from agentic_pipeline.mcp_server import activate_escape_hatch_tool

    assert callable(activate_escape_hatch_tool)


def test_get_autonomy_readiness_exists(db_path):
    from agentic_pipeline.mcp_server import get_autonomy_readiness

    assert callable(get_autonomy_readiness)
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_mcp_phase5.py -v`
Expected: FAIL (functions not found)

**Step 3: Write minimal implementation**

Add to `agentic_pipeline/mcp_server.py`:

```python
# Phase 5: Autonomy Tools

def get_autonomy_status() -> dict:
    """
    Get current autonomy mode, thresholds, and metrics summary.
    """
    from agentic_pipeline.autonomy import AutonomyConfig, MetricsCollector

    db_path = get_db_path()
    config = AutonomyConfig(db_path)
    collector = MetricsCollector(db_path)

    mode = config.get_mode()
    metrics = collector.get_metrics(days=30)

    return {
        "mode": mode,
        "escape_hatch_active": config.is_escape_hatch_active(),
        "metrics_30d": metrics,
    }


def set_autonomy_mode(mode: str) -> dict:
    """
    Change autonomy mode.

    Args:
        mode: One of "supervised", "partial", "confident"
    """
    from agentic_pipeline.autonomy import AutonomyConfig

    db_path = get_db_path()
    config = AutonomyConfig(db_path)

    if config.is_escape_hatch_active():
        return {"error": "Cannot change mode while escape hatch is active"}

    config.set_mode(mode)
    return {"mode": mode, "success": True}


def activate_escape_hatch_tool(reason: str) -> dict:
    """
    Immediately revert to supervised mode.

    Args:
        reason: Why the escape hatch is being activated
    """
    from agentic_pipeline.autonomy import AutonomyConfig

    db_path = get_db_path()
    config = AutonomyConfig(db_path)
    config.activate_escape_hatch(reason)

    return {
        "success": True,
        "message": "Escape hatch activated. All books now require human review.",
        "reason": reason,
    }


def get_autonomy_readiness() -> dict:
    """
    Check if the system is ready to advance to the next autonomy mode.
    """
    from agentic_pipeline.autonomy import AutonomyConfig, MetricsCollector, CalibrationEngine

    db_path = get_db_path()
    config = AutonomyConfig(db_path)
    collector = MetricsCollector(db_path)
    engine = CalibrationEngine(db_path)

    mode = config.get_mode()
    metrics = collector.get_metrics(days=90)

    # Calculate override rate
    total = metrics["total_processed"]
    overrides = metrics["human_rejected"] + metrics["human_adjusted"]
    override_rate = overrides / total if total > 0 else None

    # Get thresholds
    thresholds = engine.update_thresholds()

    ready_for_partial = total >= 100 and (override_rate or 1) < 0.15
    ready_for_confident = total >= 500 and (override_rate or 1) < 0.05

    return {
        "current_mode": mode,
        "total_processed": total,
        "override_rate": override_rate,
        "thresholds": thresholds,
        "ready_for_partial": ready_for_partial,
        "ready_for_confident": ready_for_confident,
        "recommendation": "confident" if ready_for_confident else ("partial" if ready_for_partial else "supervised"),
    }
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/test_mcp_phase5.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add agentic_pipeline/mcp_server.py tests/test_mcp_phase5.py
git commit -m "feat: add Phase 5 MCP tools (autonomy status, mode, escape hatch, readiness)"
```

---

## Task 8: Phase 5 Integration Tests

**Files:**
- Create: `tests/test_phase5_integration.py`

**Step 1: Write the integration test**

```python
# tests/test_phase5_integration.py
"""Integration tests for Phase 5 features."""

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


def test_full_autonomy_flow(db_path):
    """Test complete autonomy workflow."""
    from agentic_pipeline.autonomy import (
        AutonomyConfig, MetricsCollector, CalibrationEngine, SpotCheckManager
    )

    config = AutonomyConfig(db_path)
    collector = MetricsCollector(db_path)
    engine = CalibrationEngine(db_path, min_samples=10)
    spot_check = SpotCheckManager(db_path)

    # Start in supervised mode
    assert config.get_mode() == "supervised"

    # Record 50 decisions
    for i in range(50):
        collector.record_decision(
            book_id=f"book{i}",
            book_type="technical_tutorial",
            confidence=0.92,
            decision="approved",
            actor="human:taylor"
        )

    # Calculate calibration
    calibration = engine.calculate_calibration("technical_tutorial")
    assert calibration is not None
    assert calibration["accuracy"] == 1.0  # All approved

    # Calculate threshold
    threshold = engine.calculate_threshold("technical_tutorial")
    assert threshold is not None
    assert threshold <= 0.92

    # Enable partial autonomy
    config.set_mode("partial")
    assert config.get_mode() == "partial"

    # Test escape hatch
    config.activate_escape_hatch("Testing")
    assert config.is_escape_hatch_active()
    assert config.get_mode() == "supervised"  # Reverts to supervised

    # Resume
    config.deactivate_escape_hatch()
    assert not config.is_escape_hatch_active()


def test_spot_check_integration(db_path):
    """Test spot-check workflow."""
    from agentic_pipeline.autonomy import MetricsCollector, SpotCheckManager

    collector = MetricsCollector(db_path)
    spot_check = SpotCheckManager(db_path, sample_rate=0.20)

    # Record 10 auto-approved books
    for i in range(10):
        collector.record_decision(
            book_id=f"book{i}",
            book_type="technical_tutorial",
            confidence=0.95,
            decision="approved",
            actor="auto:high_confidence"
        )

    # Select for review
    pending = spot_check.select_for_review()
    assert len(pending) >= 2  # At least 20% of 10

    # Submit results
    spot_check.submit_result(
        book_id="book0",
        classification_correct=True,
        quality_acceptable=True,
        reviewer="human:taylor"
    )

    # Check accuracy
    accuracy = spot_check.get_accuracy_rate()
    assert accuracy == 1.0  # 1/1 correct
```

**Step 2: Run test**

Run: `source .venv/bin/activate && python -m pytest tests/test_phase5_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_phase5_integration.py
git commit -m "test: add Phase 5 integration tests"
```

---

## Task 9: Final Summary Commit

**Step 1: Run all Phase 5 tests**

```bash
source .venv/bin/activate && python -m pytest tests/test_phase5*.py tests/test_autonomy*.py tests/test_calibration.py tests/test_spot_check.py tests/test_cli_phase5.py tests/test_mcp_phase5.py -v
```

Expected: All tests pass

**Step 2: Commit summary**

```bash
git add -A
git commit -m "feat: complete Phase 5 Confident Autonomy implementation

Features:
- AutonomyConfig: Mode management and escape hatch
- MetricsCollector: Track decisions and accuracy
- CalibrationEngine: Calculate safe thresholds per book type
- SpotCheckManager: Ongoing verification sampling
- CLI commands: autonomy status/enable/disable, escape-hatch, spot-check
- MCP tools: get_autonomy_status, set_autonomy_mode, activate_escape_hatch

All tests passing."
```

**Step 3: Push**

```bash
git push origin main
```

---

## Summary

| Task | Component | Tests |
|------|-----------|-------|
| 1 | Database migrations | 5 tests |
| 2 | AutonomyConfig | 6 tests |
| 3 | MetricsCollector | 3 tests |
| 4 | CalibrationEngine | 3 tests |
| 5 | SpotCheckManager | 3 tests |
| 6 | CLI commands | 4 tests |
| 7 | MCP tools | 4 tests |
| 8 | Integration tests | 2 tests |
| 9 | Final verification | - |

**Total: ~30 tests**

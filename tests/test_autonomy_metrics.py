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
        book_id="book123", book_type="technical_tutorial", confidence=0.92, decision="approved", actor="human:taylor"
    )

    metrics = collector.get_metrics(days=1)
    assert metrics["total_processed"] == 1
    assert metrics["human_approved"] == 1


def test_record_auto_approval(db_path):
    from agentic_pipeline.autonomy import MetricsCollector
    from agentic_pipeline.db.connection import get_pipeline_db

    collector = MetricsCollector(db_path)
    collector.record_decision(
        book_id="book123",
        book_type="technical_tutorial",
        confidence=0.95,
        decision="approved",
        actor="auto:high_confidence",
    )

    metrics = collector.get_metrics(days=1)
    assert metrics["auto_approved"] == 1
    with get_pipeline_db(str(db_path)) as conn:
        row = conn.execute("SELECT human_decision FROM autonomy_feedback WHERE book_id = 'book123'").fetchone()
    assert row["human_decision"] == "pending_review"


def test_get_accuracy_by_type(db_path):
    from agentic_pipeline.autonomy import MetricsCollector, SpotCheckManager

    collector = MetricsCollector(db_path)
    spot_checks = SpotCheckManager(db_path)

    # Record some decisions
    collector.record_decision("b1", "technical_tutorial", 0.92, "approved", "auto:test")
    collector.record_decision("b2", "technical_tutorial", 0.88, "approved", "auto:test")
    collector.record_decision("b3", "technical_tutorial", 0.85, "rejected", "human:taylor")  # Override
    spot_checks.submit_result("b1", True, True, "human:taylor")
    spot_checks.submit_result("b2", True, True, "human:taylor")

    accuracy = collector.get_accuracy_by_type("technical_tutorial")
    # 2 approved, 1 rejected (override) = 2/3 = 66.7%
    assert accuracy["sample_count"] == 3
    assert 0.65 < accuracy["accuracy"] < 0.68


def test_unreviewed_auto_approval_is_excluded_from_accuracy(db_path):
    from agentic_pipeline.autonomy import MetricsCollector

    collector = MetricsCollector(db_path)
    collector.record_decision(
        "auto-pending",
        "technical_tutorial",
        0.99,
        "approved",
        "auto:partial",
    )

    accuracy = collector.get_accuracy_by_type("technical_tutorial")

    assert accuracy["sample_count"] == 0
    assert accuracy["accuracy"] is None

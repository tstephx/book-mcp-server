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

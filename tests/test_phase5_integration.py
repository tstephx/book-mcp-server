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

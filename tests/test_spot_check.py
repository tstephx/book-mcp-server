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
    assert results[0]["classification_correct"] == 1  # SQLite stores booleans as integers

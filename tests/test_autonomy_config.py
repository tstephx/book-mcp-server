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

    assert config.should_auto_approve("technical_tutorial", 0.96) is True


def _update_autonomy_config(db_path, **values):
    from agentic_pipeline.db.connection import get_pipeline_db

    assignments = ", ".join(f"{key} = ?" for key in values)
    with get_pipeline_db(str(db_path)) as conn:
        conn.execute(
            f"UPDATE autonomy_config SET {assignments} WHERE id = 1",
            tuple(values.values()),
        )
        conn.commit()


def _set_type_threshold(db_path, book_type, threshold):
    from agentic_pipeline.db.connection import get_pipeline_db

    with get_pipeline_db(str(db_path)) as conn:
        conn.execute(
            """INSERT INTO autonomy_thresholds
               (book_type, auto_approve_threshold, sample_count, measured_accuracy)
               VALUES (?, ?, 100, 0.99)""",
            (book_type, threshold),
        )
        conn.commit()


def test_evaluate_auto_approval_explains_supervised_denial(db_path):
    from agentic_pipeline.autonomy import AutonomyConfig

    decision = AutonomyConfig(db_path).evaluate_auto_approval(
        "technical_tutorial",
        0.99,
    )

    assert decision.should_auto_approve is False
    assert decision.mode == "supervised"
    assert decision.reason == "supervised_mode"
    assert decision.threshold is None


def test_escape_hatch_blocks_partial_auto_approval(db_path):
    from agentic_pipeline.autonomy import AutonomyConfig

    config = AutonomyConfig(db_path)
    config.set_mode("partial")
    config.activate_escape_hatch("operator requested stop")

    decision = config.evaluate_auto_approval("technical_tutorial", 0.99)

    assert decision.should_auto_approve is False
    assert decision.reason == "escape_hatch_active"


def test_partial_mode_uses_global_database_threshold(db_path):
    from agentic_pipeline.autonomy import AutonomyConfig

    config = AutonomyConfig(db_path)
    config.set_mode("partial")
    _update_autonomy_config(db_path, auto_approve_threshold=0.93)

    below = config.evaluate_auto_approval("technical_tutorial", 0.92)
    at_threshold = config.evaluate_auto_approval("technical_tutorial", 0.93)

    assert below.should_auto_approve is False
    assert below.reason == "below_threshold"
    assert below.threshold == pytest.approx(0.93)
    assert at_threshold.should_auto_approve is True
    assert at_threshold.reason == "approved"


def test_confident_mode_requires_calibrated_type_threshold(db_path):
    from agentic_pipeline.autonomy import AutonomyConfig

    config = AutonomyConfig(db_path)
    config.set_mode("confident")

    missing = config.evaluate_auto_approval("technical_tutorial", 0.99)
    _set_type_threshold(db_path, "technical_tutorial", 0.91)
    calibrated = config.evaluate_auto_approval("technical_tutorial", 0.92)

    assert missing.should_auto_approve is False
    assert missing.reason == "threshold_unavailable"
    assert calibrated.should_auto_approve is True
    assert calibrated.threshold == pytest.approx(0.91)


def test_confident_mode_honors_zero_manual_override(db_path):
    from agentic_pipeline.autonomy import AutonomyConfig
    from agentic_pipeline.db.connection import get_pipeline_db

    config = AutonomyConfig(db_path)
    config.set_mode("confident")
    _set_type_threshold(db_path, "technical_tutorial", 0.91)
    with get_pipeline_db(str(db_path)) as conn:
        conn.execute(
            """UPDATE autonomy_thresholds
               SET manual_override = 0.0
               WHERE book_type = 'technical_tutorial'"""
        )
        conn.commit()

    decision = config.evaluate_auto_approval("technical_tutorial", 0.10)

    assert decision.should_auto_approve is True
    assert decision.threshold == pytest.approx(0.0)


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_confidence_fails_closed(db_path, confidence):
    from agentic_pipeline.autonomy import AutonomyConfig

    config = AutonomyConfig(db_path)
    config.set_mode("partial")

    decision = config.evaluate_auto_approval("technical_tutorial", confidence)

    assert decision.should_auto_approve is False
    assert decision.reason == "invalid_confidence"


def test_unknown_book_type_fails_closed(db_path):
    from agentic_pipeline.autonomy import AutonomyConfig

    config = AutonomyConfig(db_path)
    config.set_mode("partial")

    decision = config.evaluate_auto_approval("unknown", 0.99)

    assert decision.should_auto_approve is False
    assert decision.reason == "unknown_book_type"


def test_validation_failure_fails_closed(db_path):
    from agentic_pipeline.autonomy import AutonomyConfig

    config = AutonomyConfig(db_path)
    config.set_mode("partial")

    decision = config.evaluate_auto_approval(
        "technical_tutorial",
        0.99,
        validation_passed=False,
    )

    assert decision.should_auto_approve is False
    assert decision.reason == "validation_failed"


def test_needs_review_fails_closed(db_path):
    from agentic_pipeline.autonomy import AutonomyConfig

    config = AutonomyConfig(db_path)
    config.set_mode("partial")

    decision = config.evaluate_auto_approval(
        "technical_tutorial",
        0.99,
        needs_review=True,
    )

    assert decision.should_auto_approve is False
    assert decision.reason == "needs_review"


def test_daily_auto_approval_cap_fails_closed(db_path):
    from agentic_pipeline.autonomy import AutonomyConfig
    from agentic_pipeline.audit import AuditTrail

    config = AutonomyConfig(db_path)
    config.set_mode("partial")
    _update_autonomy_config(db_path, max_auto_approvals_per_day=1)
    AuditTrail(db_path).log(
        book_id="already-approved",
        pipeline_id="pipeline-1",
        action="approved",
        actor="auto:partial",
    )

    decision = config.evaluate_auto_approval("technical_tutorial", 0.99)

    assert decision.should_auto_approve is False
    assert decision.reason == "daily_limit_reached"

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

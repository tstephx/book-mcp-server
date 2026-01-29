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

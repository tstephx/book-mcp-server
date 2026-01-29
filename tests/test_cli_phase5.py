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

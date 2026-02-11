# tests/test_cli_orchestrator.py
"""Tests for orchestrator CLI commands."""

import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock


def test_process_command_exists():
    from agentic_pipeline.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["process", "--help"])

    assert result.exit_code == 0
    assert "process" in result.output.lower()


def test_worker_command_exists():
    from agentic_pipeline.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["worker", "--help"])

    assert result.exit_code == 0


def test_retry_command_exists():
    from agentic_pipeline.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["retry", "--help"])

    assert result.exit_code == 0


def test_status_command_exists():
    from agentic_pipeline.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["status", "--help"])

    assert result.exit_code == 0


def test_worker_command_accepts_watch_dir():
    from agentic_pipeline.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["worker", "--help"])

    assert result.exit_code == 0
    assert "--watch-dir" in result.output

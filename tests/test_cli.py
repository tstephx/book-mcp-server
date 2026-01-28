"""Tests for CLI commands."""

import pytest
from click.testing import CliRunner


def test_version_command():
    from agentic_pipeline.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["version"])

    assert result.exit_code == 0
    assert "agentic-pipeline v" in result.output


def test_strategies_command():
    from agentic_pipeline.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["strategies"])

    assert result.exit_code == 0
    assert "technical_tutorial_v1" in result.output

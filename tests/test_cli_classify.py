# tests/test_cli_classify.py
"""Tests for classify CLI command."""

import pytest
import tempfile
from pathlib import Path
from click.testing import CliRunner
from unittest.mock import patch, Mock


@pytest.fixture
def temp_db(monkeypatch):
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(path))
    yield path
    path.unlink(missing_ok=True)


def test_classify_command_exists():
    from agentic_pipeline.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["classify", "--help"])

    assert result.exit_code == 0
    assert "Classify" in result.output or "classify" in result.output


def test_classify_command_with_mock_provider(temp_db):
    from agentic_pipeline.cli import main
    from agentic_pipeline.agents.classifier_types import BookProfile, BookType

    mock_profile = BookProfile(
        book_type=BookType.TECHNICAL_TUTORIAL,
        confidence=0.9,
        suggested_tags=["python"],
        reasoning="Test"
    )

    mock_agent = Mock()
    mock_agent.classify.return_value = mock_profile

    with patch('agentic_pipeline.agents.providers.openai_provider.OpenAI'):
        with patch('agentic_pipeline.agents.classifier.ClassifierAgent', return_value=mock_agent):
            runner = CliRunner()
            result = runner.invoke(main, ["classify", "--text", "Sample book content"])

    assert result.exit_code == 0, f"Failed with: {result.output}"
    assert "technical_tutorial" in result.output.lower()

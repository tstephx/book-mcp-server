# tests/test_cli_phase4.py
"""Tests for Phase 4 CLI commands."""

import pytest
from unittest.mock import patch
from click.testing import CliRunner


class TestApproveCommandEmbeds:
    """`agentic-pipeline approve` must not exit before embedding finishes.

    Regression: the CLI took approve_book()'s default background=True, which
    embeds on a daemon thread. The CLI process exits as soon as it prints, so
    the thread was killed and the book sat in APPROVED with no chunks and no
    error. It only ever completed because a background worker swept up APPROVED
    books; with no worker running, the content was silently unsearchable.
    """

    def test_approve_requests_foreground_embedding(self):
        from agentic_pipeline.cli import main

        with patch("agentic_pipeline.approval.actions.approve_book") as mock:
            mock.return_value = {"success": True, "state": "complete", "chapters_embedded": 3}
            result = CliRunner().invoke(main, ["approve", "some-pipeline-id"])

        assert result.exit_code == 0
        assert mock.call_args.kwargs.get("background") is False, (
            "CLI must pass background=False; it exits on return and would kill the embed thread"
        )

    def test_approve_reports_the_embedding_outcome(self):
        from agentic_pipeline.cli import main

        with patch("agentic_pipeline.approval.actions.approve_book") as mock:
            mock.return_value = {"success": True, "state": "complete", "chapters_embedded": 42}
            result = CliRunner().invoke(main, ["approve", "some-pipeline-id"])

        assert "42" in result.output

    def test_approve_surfaces_embedding_failure(self):
        from agentic_pipeline.cli import main

        with patch("agentic_pipeline.approval.actions.approve_book") as mock:
            mock.return_value = {
                "success": True,
                "state": "needs_retry",
                "embedding_error": "insufficient_quota",
            }
            result = CliRunner().invoke(main, ["approve", "some-pipeline-id"])

        assert "insufficient_quota" in result.output


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

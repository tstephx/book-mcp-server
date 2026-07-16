"""ClaudeCodeProvider — classification via `claude -p` (all subprocess mocked)."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from agentic_pipeline.agents.providers.claude_code_provider import ClaudeCodeProvider

VALID_JSON = '{"book_type": "technical_tutorial", "confidence": 0.85, "suggested_tags": ["git"], "reasoning": "vcs"}'


def _proc(returncode=0, stdout=VALID_JSON, stderr=""):
    m = MagicMock()
    m.returncode = returncode
    m.stdout = stdout
    m.stderr = stderr
    return m


class TestClaudeCodeProvider:
    def test_name(self):
        assert ClaudeCodeProvider().name == "claude-code"

    @patch("agentic_pipeline.agents.providers.claude_code_provider.subprocess.run")
    def test_classify_success_bare_json(self, mock_run):
        mock_run.return_value = _proc()
        profile = ClaudeCodeProvider().classify("some book text")
        assert profile.book_type.value == "technical_tutorial"
        assert isinstance(profile.confidence, float)
        # list args, no shell, prompt contains the text
        args, kwargs = mock_run.call_args
        cmd = args[0]
        assert cmd[0] == "claude" and cmd[1] == "-p"
        assert "some book text" in cmd[2]
        assert kwargs.get("shell") is not True
        assert kwargs["timeout"] == 120

    @patch("agentic_pipeline.agents.providers.claude_code_provider.subprocess.run")
    def test_classify_success_fenced_json(self, mock_run):
        mock_run.return_value = _proc(stdout=f"```json\n{VALID_JSON}\n```")
        assert ClaudeCodeProvider().classify("t").book_type.value == "technical_tutorial"

    @patch("agentic_pipeline.agents.providers.claude_code_provider.subprocess.run")
    def test_nonzero_exit_raises_runtime_error(self, mock_run):
        mock_run.return_value = _proc(returncode=1, stdout="", stderr="not logged in")
        with pytest.raises(RuntimeError, match="exit 1"):
            ClaudeCodeProvider().classify("t")

    @patch("agentic_pipeline.agents.providers.claude_code_provider.subprocess.run")
    def test_timeout_raises_runtime_error(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=120)
        with pytest.raises(RuntimeError, match="timed out"):
            ClaudeCodeProvider().classify("t")

    @patch("agentic_pipeline.agents.providers.claude_code_provider.subprocess.run")
    def test_missing_binary_raises_runtime_error(self, mock_run):
        mock_run.side_effect = FileNotFoundError("claude")
        with pytest.raises(RuntimeError, match="unavailable"):
            ClaudeCodeProvider().classify("t")

    @patch("agentic_pipeline.agents.providers.claude_code_provider.subprocess.run")
    def test_garbage_output_raises_value_error_family(self, mock_run):
        mock_run.return_value = _proc(stdout="no json here")
        with pytest.raises((RuntimeError, ValueError)):
            ClaudeCodeProvider().classify("t")

    @patch("agentic_pipeline.agents.providers.claude_code_provider.subprocess.run")
    def test_text_truncated_to_40k(self, mock_run):
        mock_run.return_value = _proc()
        ClaudeCodeProvider().classify("x" * 50000)
        prompt = mock_run.call_args[0][0][2]
        # the prompt embeds at most 40000 chars of book text
        assert "x" * 40001 not in prompt
        assert "x" * 1000 in prompt

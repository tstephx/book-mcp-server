# agentic_pipeline/agents/providers/claude_code_provider.py
"""Claude Code CLI provider — classification via `claude -p`.

Uses the local claude CLI (subscription billing, no API key). Any
failure raises RuntimeError so ClassifierAgent falls back to the next
provider.
"""

import subprocess
from typing import Optional

from agentic_pipeline.agents.classifier_types import BookProfile
from agentic_pipeline.agents.providers.base import LLMProvider, parse_profile_json
from agentic_pipeline.agents.prompts import load_prompt


class ClaudeCodeProvider(LLMProvider):
    """Classification through the `claude` CLI (measured ~14s/book)."""

    def __init__(self, timeout: int = 120):
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "claude-code"

    def classify(self, text: str, metadata: Optional[dict] = None) -> BookProfile:
        prompt_template = load_prompt("classify")
        prompt = prompt_template.format(text=text[:40000])

        try:
            proc = subprocess.run(
                ["claude", "-p", prompt],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"claude -p timed out after {self.timeout}s") from e
        except OSError as e:
            raise RuntimeError(f"claude CLI unavailable: {e}") from e

        if proc.returncode != 0:
            raise RuntimeError(f"claude -p exit {proc.returncode}: {proc.stderr.strip()[:200]}")

        return parse_profile_json(proc.stdout.strip())

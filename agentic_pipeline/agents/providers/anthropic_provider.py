# agentic_pipeline/agents/providers/anthropic_provider.py
"""Anthropic Claude LLM provider."""

import json
import os
from typing import Optional

from anthropic import Anthropic

from agentic_pipeline.agents.classifier_types import BookProfile
from agentic_pipeline.agents.providers.base import LLMProvider
from agentic_pipeline.agents.prompts import load_prompt


class AnthropicProvider(LLMProvider):
    """Anthropic Claude-based classification provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-haiku-4-5-20251001",
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self._client = Anthropic(api_key=self.api_key)

    @property
    def name(self) -> str:
        return "anthropic"

    def classify(self, text: str, metadata: Optional[dict] = None) -> BookProfile:
        """Classify text using Anthropic Claude."""
        prompt_template = load_prompt("classify")
        prompt = prompt_template.format(text=text[:40000])  # ~10K tokens

        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt},
            ],
            system="You are a book classifier. Return only valid JSON.",
            temperature=0.1,
        )

        content = response.content[0].text.strip()
        return self._parse_response(content)

    def _parse_response(self, content: str) -> BookProfile:
        """Parse JSON response into BookProfile."""
        # Try to extract JSON if wrapped in markdown code blocks
        if "```" in content:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                content = content[start:end]

        try:
            data = json.loads(content)
            return BookProfile.from_dict(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")

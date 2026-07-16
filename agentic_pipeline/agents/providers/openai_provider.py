# agentic_pipeline/agents/providers/openai_provider.py
"""OpenAI LLM provider."""

import os
from typing import Optional

from openai import OpenAI

from agentic_pipeline.agents.classifier_types import BookProfile
from agentic_pipeline.agents.providers.base import LLMProvider, parse_profile_json
from agentic_pipeline.agents.prompts import load_prompt


class OpenAIProvider(LLMProvider):
    """OpenAI-based classification provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4.1-mini",
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self._client = OpenAI(api_key=self.api_key)

    @property
    def name(self) -> str:
        return "openai"

    def _normalize_text(self, text: str) -> str:
        """Replace smart quotes and other problematic unicode with ASCII equivalents."""
        replacements = {
            "\u201c": '"',  # Left double quote
            "\u201d": '"',  # Right double quote
            "\u2018": "'",  # Left single quote
            "\u2019": "'",  # Right single quote
            "\u2013": "-",  # En dash
            "\u2014": "-",  # Em dash
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def classify(self, text: str, metadata: Optional[dict] = None) -> BookProfile:
        """Classify text using OpenAI."""
        # Normalize input text first (user might have curly quotes from terminal)
        text = self._normalize_text(text)
        prompt_template = load_prompt("classify")
        prompt = self._normalize_text(prompt_template.format(text=text[:40000]))

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a book classifier. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,  # Low temperature for consistent classification
        )

        content = response.choices[0].message.content.strip()
        return self._parse_response(content)

    def _parse_response(self, content: str) -> BookProfile:
        """Parse JSON response into BookProfile (shared implementation)."""
        return parse_profile_json(content)

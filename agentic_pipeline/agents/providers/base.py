# agentic_pipeline/agents/providers/base.py
"""Abstract base class for LLM providers."""

import json
from abc import ABC, abstractmethod
from typing import Optional

from agentic_pipeline.agents.classifier_types import BookProfile


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def classify(self, text: str, metadata: Optional[dict] = None) -> BookProfile:
        """
        Classify book text and return a BookProfile.

        Args:
            text: Pre-extracted book content (truncated to ~10K tokens)
            metadata: Optional hints (filename, source folder, etc.)

        Returns:
            BookProfile with classification results
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging (e.g., 'openai', 'anthropic')."""
        pass


def parse_profile_json(content: str) -> BookProfile:
    """Parse an LLM classification response into a BookProfile.

    Tolerates markdown code fences by extracting the outermost {...}
    span. Raises ValueError on unparseable or non-dict JSON so provider
    callers surface a clean failure the agent can fall back on.
    """
    if "```" in content:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            content = content[start:end]

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}")
    if not isinstance(data, dict):
        raise ValueError(f"LLM response is not a JSON object: {type(data).__name__}")
    return BookProfile.from_dict(data)

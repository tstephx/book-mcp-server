# agentic_pipeline/agents/providers/base.py
"""Abstract base class for LLM providers."""

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

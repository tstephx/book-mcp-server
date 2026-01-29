# agentic_pipeline/agents/classifier.py
"""Classifier Agent - orchestrates book classification."""

import logging
from pathlib import Path
from typing import Optional

from agentic_pipeline.agents.classifier_types import BookProfile
from agentic_pipeline.agents.providers.base import LLMProvider
from agentic_pipeline.agents.providers.openai_provider import OpenAIProvider
from agentic_pipeline.agents.providers.anthropic_provider import AnthropicProvider
from agentic_pipeline.db.pipelines import PipelineRepository

logger = logging.getLogger(__name__)


class ClassifierAgent:
    """
    Orchestrates book classification using LLM providers.

    Flow:
    1. Check if we've seen this content hash before (cache)
    2. Try primary provider (OpenAI by default)
    3. Try fallback provider (Anthropic by default)
    4. Return unknown profile if all fail
    """

    def __init__(
        self,
        db_path: Path,
        primary: Optional[LLMProvider] = None,
        fallback: Optional[LLMProvider] = None,
    ):
        self.db_path = db_path
        self.repo = PipelineRepository(db_path)
        self.primary = primary
        self.fallback = fallback

        # Lazy initialization of default providers
        self._primary_initialized = primary is not None
        self._fallback_initialized = fallback is not None

    def _get_primary(self) -> LLMProvider:
        """Get or initialize primary provider."""
        if not self._primary_initialized:
            self.primary = OpenAIProvider()
            self._primary_initialized = True
        return self.primary

    def _get_fallback(self) -> LLMProvider:
        """Get or initialize fallback provider."""
        if not self._fallback_initialized:
            self.fallback = AnthropicProvider()
            self._fallback_initialized = True
        return self.fallback

    def classify(
        self,
        text: str,
        content_hash: str,
        metadata: Optional[dict] = None,
    ) -> BookProfile:
        """
        Classify book text and return a BookProfile.

        Args:
            text: Pre-extracted book content
            content_hash: Hash of book content (for caching)
            metadata: Optional hints (filename, etc.)

        Returns:
            BookProfile with classification results
        """
        # 1. Check cache (existing pipeline with this hash)
        cached = self._check_cache(content_hash)
        if cached:
            logger.info(f"Cache hit for {content_hash[:8]}")
            return cached

        # 2. Try primary provider
        try:
            primary = self._get_primary()
            logger.info(f"Calling primary provider: {primary.name}")
            return primary.classify(text, metadata)
        except Exception as e:
            logger.warning(f"Primary provider ({self._get_primary().name}) failed: {e}")

        # 3. Try fallback provider
        try:
            fallback = self._get_fallback()
            logger.info(f"Calling fallback provider: {fallback.name}")
            return fallback.classify(text, metadata)
        except Exception as e:
            logger.warning(f"Fallback provider ({self._get_fallback().name}) failed: {e}")

        # 4. Return safe default
        logger.error("All providers failed, returning unknown")
        return BookProfile.unknown("Classification failed - all providers unavailable")

    def _check_cache(self, content_hash: str) -> Optional[BookProfile]:
        """Check if we have a cached classification for this hash."""
        existing = self.repo.find_by_hash(content_hash)
        if existing and existing.get("book_profile"):
            try:
                import json
                profile_data = existing["book_profile"]
                if isinstance(profile_data, str):
                    profile_data = json.loads(profile_data)
                return BookProfile.from_dict(profile_data)
            except Exception as e:
                logger.warning(f"Failed to parse cached profile: {e}")
        return None

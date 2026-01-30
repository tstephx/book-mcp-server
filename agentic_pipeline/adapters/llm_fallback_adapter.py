"""
LLM Fallback Adapter - implements the LLMFallbackPort protocol.

This adapter wraps the agentic pipeline's LLM capabilities to provide
chapter detection fallback when the book-ingestion library has low confidence.
"""

import logging
from typing import Optional

# Import the protocol from book-ingestion
from book_ingestion.ports.llm_fallback import (
    LLMFallbackPort,
    LLMFallbackRequest,
    LLMFallbackResponse,
)

logger = logging.getLogger(__name__)


class LLMFallbackAdapter:
    """
    Adapter that implements LLMFallbackPort using agentic pipeline's LLM.

    This enables the book-ingestion library to request LLM assistance when
    chapter detection confidence is low.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        max_text_sample: int = 10000,
    ):
        """
        Initialize the LLM fallback adapter.

        Args:
            confidence_threshold: Trigger LLM if detection confidence below this
            max_text_sample: Maximum characters to send to LLM
        """
        self.confidence_threshold = confidence_threshold
        self.max_text_sample = max_text_sample
        self._llm_client = None  # Lazy-loaded

    def _get_llm_client(self):
        """Lazy-load the LLM client."""
        if self._llm_client is None:
            try:
                # Try to import anthropic client
                import anthropic
                self._llm_client = anthropic.Anthropic()
            except ImportError:
                logger.warning("anthropic package not available for LLM fallback")
                return None
        return self._llm_client

    def should_trigger(self, confidence: float, method: str) -> bool:
        """
        Determine if LLM fallback should be triggered.

        Args:
            confidence: Current detection confidence (0-1)
            method: Detection method used

        Returns:
            True if LLM fallback should be attempted
        """
        # Always skip for high confidence
        if confidence >= 0.7:
            return False

        # Always trigger for low confidence
        if confidence < self.confidence_threshold:
            return True

        # For medium confidence, only trigger for certain methods
        weak_methods = {"fallback", "heuristic", "pattern"}
        if method in weak_methods and confidence < 0.6:
            return True

        return False

    def improve_detection(
        self, request: LLMFallbackRequest
    ) -> Optional[LLMFallbackResponse]:
        """
        Attempt to improve chapter detection using LLM analysis.

        Args:
            request: Context about the book and current detection results

        Returns:
            Improved detection results, or None if LLM cannot improve
        """
        client = self._get_llm_client()
        if client is None:
            logger.warning("LLM client not available, skipping fallback")
            return None

        try:
            prompt = self._build_prompt(request)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )

            return self._parse_response(response, request)

        except Exception as e:
            logger.error(f"LLM fallback failed: {e}")
            return None

    def _build_prompt(self, request: LLMFallbackRequest) -> str:
        """Build the prompt for LLM analysis."""
        chapters_summary = "\n".join(
            f"  {i+1}. {ch.get('title', 'Untitled')} ({ch.get('word_count', 0)} words)"
            for i, ch in enumerate(request.detected_chapters[:20])
        )

        return f"""Analyze this book's chapter structure and suggest improvements.

Book: {request.book_metadata.get('title', 'Unknown')}
Author: {request.book_metadata.get('author', 'Unknown')}
Detection method: {request.detection_method}
Current confidence: {request.detection_confidence:.1%}

Current chapters detected:
{chapters_summary}

Text sample (first {len(request.text_sample)} chars):
---
{request.text_sample[:self.max_text_sample]}
---

Based on this information:
1. Are the detected chapters correct?
2. Should any chapters be merged (e.g., fragments that are part of a larger chapter)?
3. Should any chapters be split (e.g., multiple topics in one chapter)?
4. What corrections would you suggest?

Respond in this JSON format:
{{
    "chapters_look_correct": true/false,
    "confidence_improvement": 0.0-0.3,
    "merge_pairs": [[0, 1], [3, 4]],  // Indices of chapters to merge
    "split_chapters": [2],  // Indices of chapters to split
    "corrections": ["Description of each correction made"],
    "improved_chapters": [  // Optional: only if corrections needed
        {{"chapter_number": 1, "title": "...", "word_count": 1234}}
    ]
}}"""

    def _parse_response(
        self, response, request: LLMFallbackRequest
    ) -> Optional[LLMFallbackResponse]:
        """Parse LLM response into LLMFallbackResponse."""
        import json

        try:
            # Extract text content
            content = response.content[0].text

            # Find JSON in response
            start = content.find("{")
            end = content.rfind("}") + 1
            if start == -1 or end == 0:
                return None

            data = json.loads(content[start:end])

            # If chapters look correct, return minimal improvement
            if data.get("chapters_look_correct", True):
                return LLMFallbackResponse(
                    improved_chapters=[],
                    confidence_delta=0.1,  # Slight boost for LLM confirmation
                    corrections_made=["LLM confirmed chapter detection is reasonable"],
                    should_merge=[],
                    should_split=[],
                )

            # Extract improvements
            return LLMFallbackResponse(
                improved_chapters=data.get("improved_chapters", []),
                confidence_delta=data.get("confidence_improvement", 0.0),
                corrections_made=data.get("corrections", []),
                should_merge=[tuple(pair) for pair in data.get("merge_pairs", [])],
                should_split=data.get("split_chapters", []),
            )

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return None

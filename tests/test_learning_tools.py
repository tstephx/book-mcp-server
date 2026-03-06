"""Tests for learning_tools edge cases."""

import pytest
from unittest.mock import patch, MagicMock


def test_generate_business_impact_empty_sources():
    """_generate_business_impact should return fallback when sources is empty."""
    from src.tools.learning_tools import _generate_business_impact

    result = _generate_business_impact("kubernetes", [])
    assert "kubernetes" in result.lower() or "informed decisions" in result.lower()
    # Must NOT raise IndexError


def test_generate_tradeoffs_empty_sources():
    """_generate_tradeoffs should return fallback when sources is empty."""
    from src.tools.learning_tools import _generate_tradeoffs

    result = _generate_tradeoffs("kubernetes", [])
    assert "kubernetes" in result.lower() or "tradeoffs" in result.lower()
    # Must NOT raise IndexError


def test_split_sentences():
    """_split_sentences should split on sentence boundaries."""
    from src.tools.learning_tools import _split_sentences

    result = _split_sentences("Hello world. This is a test. Another one!")
    assert len(result) >= 2
    assert "Hello world" in result[0]


def test_extract_vocabulary_with_sources():
    """_extract_vocabulary should extract terms from source excerpts."""
    from src.tools.learning_tools import _extract_vocabulary

    sources = [
        {
            "book_title": "Test Book",
            "excerpt": "**Kubernetes** is a container orchestration platform. **Pod** refers to the smallest deployable unit.",
        }
    ]
    result = _extract_vocabulary("kubernetes", sources)
    assert isinstance(result, dict)


def test_extract_vocabulary_empty_sources():
    """_extract_vocabulary should return empty dict for no sources."""
    from src.tools.learning_tools import _extract_vocabulary

    result = _extract_vocabulary("kubernetes", [])
    assert result == {}


def test_extract_related_concepts_empty_sources():
    """_extract_related_concepts should return empty list for no sources."""
    from src.tools.learning_tools import _extract_related_concepts

    result = _extract_related_concepts("kubernetes", [])
    assert result == []


def test_generate_decisions_empty_sources():
    """_generate_decisions should return fallback for no sources."""
    from src.tools.learning_tools import _generate_decisions

    result = _generate_decisions("kubernetes", [])
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_quick_summary_empty_sources():
    """_generate_quick_summary should return fallback when sources is empty."""
    from src.tools.learning_tools import _generate_quick_summary

    result = _generate_quick_summary("kubernetes", [])
    assert "kubernetes" in result.lower()

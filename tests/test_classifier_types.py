# tests/test_classifier_types.py
"""Tests for classifier data types."""

import pytest


def test_book_type_enum_has_required_types():
    from agentic_pipeline.agents.classifier_types import BookType

    required = [
        "TECHNICAL_TUTORIAL", "TECHNICAL_REFERENCE", "TEXTBOOK",
        "NARRATIVE_NONFICTION", "PERIODICAL", "RESEARCH_COLLECTION", "UNKNOWN"
    ]

    for book_type in required:
        assert hasattr(BookType, book_type), f"Missing type: {book_type}"


def test_book_profile_creation():
    from agentic_pipeline.agents.classifier_types import BookProfile, BookType

    profile = BookProfile(
        book_type=BookType.TECHNICAL_TUTORIAL,
        confidence=0.85,
        suggested_tags=["python", "web"],
        reasoning="Contains code examples"
    )

    assert profile.book_type == BookType.TECHNICAL_TUTORIAL
    assert profile.confidence == 0.85
    assert profile.suggested_tags == ["python", "web"]
    assert profile.reasoning == "Contains code examples"


def test_book_profile_to_dict():
    from agentic_pipeline.agents.classifier_types import BookProfile, BookType

    profile = BookProfile(
        book_type=BookType.TEXTBOOK,
        confidence=0.72,
        suggested_tags=["economics"],
        reasoning="Academic structure"
    )

    d = profile.to_dict()

    assert d["book_type"] == "textbook"
    assert d["confidence"] == 0.72
    assert d["suggested_tags"] == ["economics"]
    assert d["reasoning"] == "Academic structure"


def test_book_profile_from_dict():
    from agentic_pipeline.agents.classifier_types import BookProfile, BookType

    d = {
        "book_type": "periodical",
        "confidence": 0.90,
        "suggested_tags": ["news", "politics"],
        "reasoning": "Article format with bylines"
    }

    profile = BookProfile.from_dict(d)

    assert profile.book_type == BookType.PERIODICAL
    assert profile.confidence == 0.90
    assert profile.suggested_tags == ["news", "politics"]

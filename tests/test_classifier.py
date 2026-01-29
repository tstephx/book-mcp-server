# tests/test_classifier.py
"""Tests for the ClassifierAgent."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock


@pytest.fixture
def db_path():
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    yield path
    path.unlink(missing_ok=True)


class MockProvider:
    """Mock LLM provider for testing."""

    def __init__(self, response=None, should_fail=False):
        from agentic_pipeline.agents.classifier_types import BookProfile, BookType
        self.response = response or BookProfile(
            book_type=BookType.TECHNICAL_TUTORIAL,
            confidence=0.9,
            suggested_tags=["test"],
            reasoning="Test response"
        )
        self.should_fail = should_fail
        self.call_count = 0

    @property
    def name(self):
        return "mock"

    def classify(self, text, metadata=None):
        self.call_count += 1
        if self.should_fail:
            raise Exception("Mock failure")
        return self.response


def test_classifier_returns_cached_result(db_path):
    from agentic_pipeline.agents.classifier import ClassifierAgent
    from agentic_pipeline.agents.classifier_types import BookType
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    # Setup: create pipeline with existing profile
    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "hash123")
    repo.update_book_profile(pid, {
        "book_type": "textbook",
        "confidence": 0.85,
        "suggested_tags": ["economics"],
        "reasoning": "Cached result"
    })

    mock_provider = MockProvider()
    agent = ClassifierAgent(db_path, primary=mock_provider)

    # Should return cached, not call provider
    result = agent.classify("any text", content_hash="hash123")

    assert result.book_type == BookType.TEXTBOOK
    assert result.confidence == 0.85
    assert mock_provider.call_count == 0  # Did not call LLM


def test_classifier_calls_primary_on_cache_miss(db_path):
    from agentic_pipeline.agents.classifier import ClassifierAgent
    from agentic_pipeline.agents.classifier_types import BookType

    mock_provider = MockProvider()
    agent = ClassifierAgent(db_path, primary=mock_provider)

    result = agent.classify("book text", content_hash="new-hash")

    assert result.book_type == BookType.TECHNICAL_TUTORIAL
    assert mock_provider.call_count == 1


def test_classifier_falls_back_on_primary_failure(db_path):
    from agentic_pipeline.agents.classifier import ClassifierAgent
    from agentic_pipeline.agents.classifier_types import BookProfile, BookType

    primary = MockProvider(should_fail=True)
    fallback = MockProvider(response=BookProfile(
        book_type=BookType.PERIODICAL,
        confidence=0.8,
        suggested_tags=["news"],
        reasoning="Fallback response"
    ))

    agent = ClassifierAgent(db_path, primary=primary, fallback=fallback)

    result = agent.classify("text", content_hash="hash456")

    assert result.book_type == BookType.PERIODICAL
    assert primary.call_count == 1
    assert fallback.call_count == 1


def test_classifier_returns_unknown_when_both_fail(db_path):
    from agentic_pipeline.agents.classifier import ClassifierAgent
    from agentic_pipeline.agents.classifier_types import BookType

    primary = MockProvider(should_fail=True)
    fallback = MockProvider(should_fail=True)

    agent = ClassifierAgent(db_path, primary=primary, fallback=fallback)

    result = agent.classify("text", content_hash="hash789")

    assert result.book_type == BookType.UNKNOWN
    assert result.confidence == 0.0
    assert "failed" in result.reasoning.lower()

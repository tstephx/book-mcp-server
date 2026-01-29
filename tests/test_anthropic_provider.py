# tests/test_anthropic_provider.py
"""Tests for Anthropic provider."""

import pytest
import json
from unittest.mock import Mock, patch


def test_anthropic_provider_has_correct_name():
    from agentic_pipeline.agents.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(api_key="test-key")
    assert provider.name == "anthropic"


def test_anthropic_provider_default_model():
    from agentic_pipeline.agents.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(api_key="test-key")
    assert "claude" in provider.model.lower()


def test_anthropic_provider_parses_valid_response():
    from agentic_pipeline.agents.providers.anthropic_provider import AnthropicProvider
    from agentic_pipeline.agents.classifier_types import BookType

    provider = AnthropicProvider(api_key="test-key")

    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = json.dumps({
        "book_type": "narrative_nonfiction",
        "confidence": 0.88,
        "suggested_tags": ["biography", "technology"],
        "reasoning": "Narrative structure following a person's life"
    })

    with patch.object(provider, '_client') as mock_client:
        mock_client.messages.create.return_value = mock_response
        result = provider.classify("sample text")

    assert result.book_type == BookType.NARRATIVE_NONFICTION
    assert result.confidence == 0.88


def test_anthropic_provider_handles_malformed_json():
    from agentic_pipeline.agents.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(api_key="test-key")

    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = "Not valid JSON at all"

    with patch.object(provider, '_client') as mock_client:
        mock_client.messages.create.return_value = mock_response

        with pytest.raises(ValueError, match="Failed to parse"):
            provider.classify("sample text")

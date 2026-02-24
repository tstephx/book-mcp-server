# tests/test_openai_provider.py
"""Tests for OpenAI provider."""

import pytest
import json
from unittest.mock import Mock, patch


def test_openai_provider_has_correct_name():
    from agentic_pipeline.agents.providers.openai_provider import OpenAIProvider

    with patch("agentic_pipeline.agents.providers.openai_provider.OpenAI"):
        provider = OpenAIProvider(api_key="test-key")
    assert provider.name == "openai"


def test_openai_provider_default_model():
    from agentic_pipeline.agents.providers.openai_provider import OpenAIProvider

    with patch("agentic_pipeline.agents.providers.openai_provider.OpenAI"):
        provider = OpenAIProvider(api_key="test-key")
    assert provider.model == "gpt-4.1-mini"


def test_openai_provider_custom_model():
    from agentic_pipeline.agents.providers.openai_provider import OpenAIProvider

    with patch("agentic_pipeline.agents.providers.openai_provider.OpenAI"):
        provider = OpenAIProvider(api_key="test-key", model="gpt-4o")
    assert provider.model == "gpt-4o"


def test_openai_provider_parses_valid_response():
    from agentic_pipeline.agents.providers.openai_provider import OpenAIProvider
    from agentic_pipeline.agents.classifier_types import BookType

    with patch("agentic_pipeline.agents.providers.openai_provider.OpenAI"):
        provider = OpenAIProvider(api_key="test-key")

    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = json.dumps({
        "book_type": "technical_tutorial",
        "confidence": 0.92,
        "suggested_tags": ["python", "programming"],
        "reasoning": "Contains code examples and exercises"
    })

    with patch.object(provider, '_client') as mock_client:
        mock_client.chat.completions.create.return_value = mock_response
        result = provider.classify("sample text")

    assert result.book_type == BookType.TECHNICAL_TUTORIAL
    assert result.confidence == 0.92
    assert "python" in result.suggested_tags


def test_openai_provider_handles_malformed_json():
    from agentic_pipeline.agents.providers.openai_provider import OpenAIProvider

    with patch("agentic_pipeline.agents.providers.openai_provider.OpenAI"):
        provider = OpenAIProvider(api_key="test-key")

    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "This is not JSON"

    with patch.object(provider, '_client') as mock_client:
        mock_client.chat.completions.create.return_value = mock_response

        with pytest.raises(ValueError, match="Failed to parse"):
            provider.classify("sample text")

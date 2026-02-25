"""Tests for LLMFallbackAdapter — verifies it uses OpenAI, not Anthropic."""

import json
from unittest.mock import MagicMock, patch

import pytest

from agentic_pipeline.adapters.llm_fallback_adapter import LLMFallbackAdapter
from book_ingestion.ports.llm_fallback import LLMFallbackRequest


def _make_request():
    return LLMFallbackRequest(
        text_sample="Chapter 1: Introduction\nThis is the intro...",
        detected_chapters=[{"title": "Section 1", "word_count": 2000}],
        detection_confidence=0.3,
        detection_method="fallback",
        book_metadata={"title": "Test Book", "author": "Author"},
    )


def _openai_response(content: str):
    """Build a minimal mock that looks like an OpenAI chat completion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def test_fallback_adapter_uses_openai_client():
    """_get_llm_client must instantiate openai.OpenAI, not anthropic.Anthropic."""
    adapter = LLMFallbackAdapter()

    with patch("agentic_pipeline.adapters.llm_fallback_adapter.openai") as mock_openai:
        mock_openai.OpenAI.return_value = MagicMock()
        client = adapter._get_llm_client()

    mock_openai.OpenAI.assert_called_once()


def test_improve_detection_calls_openai_chat_completions():
    """improve_detection must call client.chat.completions.create, not client.messages.create."""
    adapter = LLMFallbackAdapter()

    payload = json.dumps({
        "chapters_look_correct": True,
        "confidence_improvement": 0.1,
        "merge_pairs": [],
        "split_chapters": [],
        "corrections": [],
        "improved_chapters": [],
    })

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _openai_response(payload)
    adapter._llm_client = mock_client

    result = adapter.improve_detection(_make_request())

    mock_client.chat.completions.create.assert_called_once()
    assert result is not None
    # Anthropic path would call client.messages.create — verify it was NOT called
    mock_client.messages.create.assert_not_called()


def test_parse_response_reads_openai_format():
    """_parse_response must read response.choices[0].message.content."""
    adapter = LLMFallbackAdapter()

    payload = json.dumps({
        "chapters_look_correct": False,
        "confidence_improvement": 0.2,
        "merge_pairs": [[0, 1]],
        "split_chapters": [],
        "corrections": ["Merged intro fragments"],
        "improved_chapters": [{"chapter_number": 1, "title": "Introduction", "word_count": 4000}],
    })

    response = _openai_response(payload)
    result = adapter._parse_response(response, _make_request())

    assert result is not None
    assert result.confidence_delta == pytest.approx(0.2)
    assert result.corrections_made == ["Merged intro fragments"]
    assert result.should_merge == [(0, 1)]

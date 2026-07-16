# tests/test_provider_base.py
"""Tests for LLM provider base class."""

import pytest

from agentic_pipeline.agents.providers.base import parse_profile_json


def test_llm_provider_is_abstract():
    from agentic_pipeline.agents.providers.base import LLMProvider

    with pytest.raises(TypeError):
        LLMProvider()  # Can't instantiate abstract class


def test_llm_provider_defines_required_methods():
    from agentic_pipeline.agents.providers.base import LLMProvider
    import inspect

    # Check abstract methods exist
    assert hasattr(LLMProvider, "classify")
    assert hasattr(LLMProvider, "name")

    # Check they are abstract
    assert getattr(LLMProvider.classify, "__isabstractmethod__", False)
    assert isinstance(inspect.getattr_static(LLMProvider, "name"), property)


class TestParseProfileJson:
    VALID = '{"book_type": "technical_tutorial", "confidence": 0.9, "suggested_tags": ["python"], "reasoning": "code-heavy"}'

    def test_bare_json(self):
        profile = parse_profile_json(self.VALID)
        assert profile.book_type.value == "technical_tutorial"
        assert isinstance(profile.confidence, float)
        assert profile.confidence == 0.9

    def test_fenced_json(self):
        fenced = f"Here you go:\n```json\n{self.VALID}\n```\nDone."
        profile = parse_profile_json(fenced)
        assert profile.book_type.value == "technical_tutorial"

    def test_garbage_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_profile_json("I could not classify this book.")

    def test_non_dict_json_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_profile_json('["a", "list"]')

    def test_unknown_book_type_maps_to_unknown(self):
        profile = parse_profile_json('{"book_type": "not-a-real-type", "confidence": 0.5}')
        assert profile.book_type.value == "unknown"

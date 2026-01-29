# tests/test_provider_base.py
"""Tests for LLM provider base class."""

import pytest


def test_llm_provider_is_abstract():
    from agentic_pipeline.agents.providers.base import LLMProvider

    with pytest.raises(TypeError):
        LLMProvider()  # Can't instantiate abstract class


def test_llm_provider_defines_required_methods():
    from agentic_pipeline.agents.providers.base import LLMProvider
    import inspect

    # Check abstract methods exist
    assert hasattr(LLMProvider, 'classify')
    assert hasattr(LLMProvider, 'name')

    # Check they are abstract
    assert getattr(LLMProvider.classify, '__isabstractmethod__', False)
    assert isinstance(inspect.getattr_static(LLMProvider, 'name'), property)

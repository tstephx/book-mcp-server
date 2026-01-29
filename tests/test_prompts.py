# tests/test_prompts.py
"""Tests for prompt loading."""

import pytest
from pathlib import Path


def test_load_classify_prompt():
    from agentic_pipeline.agents.prompts import load_prompt

    prompt = load_prompt("classify")

    assert "book classifier" in prompt.lower()
    assert "{text}" in prompt
    assert "technical_tutorial" in prompt


def test_format_classify_prompt():
    from agentic_pipeline.agents.prompts import load_prompt

    prompt = load_prompt("classify")
    formatted = prompt.format(text="Sample book content here")

    assert "Sample book content here" in formatted
    assert "{text}" not in formatted

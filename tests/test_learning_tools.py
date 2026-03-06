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


def test_generate_quick_summary_empty_sources():
    """_generate_quick_summary should return fallback when sources is empty."""
    from src.tools.learning_tools import _generate_quick_summary

    result = _generate_quick_summary("kubernetes", [])
    assert "kubernetes" in result.lower()

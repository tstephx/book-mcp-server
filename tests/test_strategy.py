"""Tests for strategy selection."""

import pytest
from pathlib import Path


def test_load_strategy():
    from agentic_pipeline.pipeline.strategy import StrategySelector

    selector = StrategySelector()
    strategy = selector.load_strategy("technical_tutorial_v1")

    assert strategy["name"] == "technical_tutorial_v1"
    assert strategy["book_type"] == "technical_tutorial"
    assert strategy["chapter_detection"]["preserve_code_blocks"] is True


def test_select_strategy_for_book_type():
    from agentic_pipeline.pipeline.strategy import StrategySelector

    selector = StrategySelector()

    # Technical tutorial with code
    profile = {"book_type": "technical_tutorial", "detected_features": {"has_code_blocks": True}}
    strategy = selector.select(profile)
    assert strategy["name"] == "technical_tutorial_v1"

    # Magazine/newspaper
    profile = {"book_type": "magazine"}
    strategy = selector.select(profile)
    assert strategy["name"] == "periodical_v1"

    # Unknown type
    profile = {"book_type": "unknown"}
    strategy = selector.select(profile)
    assert strategy["name"] == "conservative_v1"


def test_select_conservative_for_low_confidence():
    from agentic_pipeline.pipeline.strategy import StrategySelector

    selector = StrategySelector()

    # Low confidence should use conservative
    profile = {"book_type": "technical_tutorial", "confidence": 0.5}
    strategy = selector.select(profile)
    assert strategy["name"] == "conservative_v1"

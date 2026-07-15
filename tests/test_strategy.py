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

    # Periodical — the value the classifier actually emits (BookType.PERIODICAL)
    profile = {"book_type": "periodical"}
    strategy = selector.select(profile)
    assert strategy["name"] == "periodical_v1"

    # Unknown type
    profile = {"book_type": "unknown"}
    strategy = selector.select(profile)
    assert strategy["name"] == "conservative_v1"


def test_select_strategy_for_travel_guide():
    from agentic_pipeline.pipeline.strategy import StrategySelector

    selector = StrategySelector()

    strategy = selector.select({"book_type": "travel_guide", "confidence": 0.9})

    assert strategy["name"] == "conservative_v1"
    # Guidebooks are prose, not tutorials: code-block gating must not apply.
    assert "code_block_detection_required" not in strategy["quality_thresholds"]


def test_every_book_type_maps_to_a_loadable_strategy():
    """Every BookType the classifier can emit must resolve to a real strategy.

    Guards the enum/STRATEGY_MAP pair: a type present in the enum but absent
    from the map silently degrades to conservative_v1 instead of failing.
    """
    from agentic_pipeline.agents.classifier_types import BookType
    from agentic_pipeline.pipeline.strategy import STRATEGY_MAP, StrategySelector

    selector = StrategySelector()

    for book_type in BookType:
        assert book_type.value in STRATEGY_MAP, f"BookType.{book_type.name} missing from STRATEGY_MAP"
        strategy = selector.select({"book_type": book_type.value, "confidence": 1.0})
        assert strategy["name"] == STRATEGY_MAP[book_type.value]


def test_strategy_map_has_no_keys_outside_book_type():
    """STRATEGY_MAP keys the classifier can never emit are dead config."""
    from agentic_pipeline.agents.classifier_types import BookType
    from agentic_pipeline.pipeline.strategy import STRATEGY_MAP

    valid = {b.value for b in BookType}

    assert set(STRATEGY_MAP) - valid == set()


def test_select_conservative_for_low_confidence():
    from agentic_pipeline.pipeline.strategy import StrategySelector

    selector = StrategySelector()

    # Low confidence should use conservative
    profile = {"book_type": "technical_tutorial", "confidence": 0.5}
    strategy = selector.select(profile)
    assert strategy["name"] == "conservative_v1"

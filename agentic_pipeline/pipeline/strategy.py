"""Strategy selection for book processing."""

import json
from pathlib import Path
from typing import Optional

# Default strategies directory
STRATEGIES_DIR = Path(__file__).parent.parent.parent / "config" / "strategies"

# Book type to strategy mapping
STRATEGY_MAP = {
    "technical_tutorial": "technical_tutorial_v1",
    "technical_reference": "technical_tutorial_v1",
    "textbook": "technical_tutorial_v1",
    "narrative_nonfiction": "narrative_v1",
    "newspaper": "periodical_v1",
    "magazine": "periodical_v1",
    "research_collection": "technical_tutorial_v1",
    "unknown": "conservative_v1",
}

# Minimum confidence to use type-specific strategy
MIN_CONFIDENCE = 0.7


class StrategySelector:
    """Selects processing strategy based on book profile."""

    def __init__(self, strategies_dir: Optional[Path] = None):
        self.strategies_dir = strategies_dir or STRATEGIES_DIR
        self._cache = {}

    def load_strategy(self, name: str) -> dict:
        """Load a strategy configuration by name."""
        if name in self._cache:
            return self._cache[name]

        path = self.strategies_dir / f"{name}.json"
        if not path.exists():
            raise ValueError(f"Strategy not found: {name}")

        with open(path) as f:
            strategy = json.load(f)

        self._cache[name] = strategy
        return strategy

    def select(self, book_profile: dict) -> dict:
        """Select the best strategy for a book profile."""
        book_type = book_profile.get("book_type", "unknown")
        confidence = book_profile.get("confidence", 1.0)

        # Use conservative for low confidence
        if confidence < MIN_CONFIDENCE:
            return self.load_strategy("conservative_v1")

        # Map book type to strategy
        strategy_name = STRATEGY_MAP.get(book_type, "conservative_v1")
        return self.load_strategy(strategy_name)

    def list_strategies(self) -> list[str]:
        """List all available strategies."""
        return [p.stem for p in self.strategies_dir.glob("*.json")]

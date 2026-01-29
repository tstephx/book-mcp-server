# agentic_pipeline/agents/prompts/__init__.py
"""Prompt template loading."""

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """Load a prompt template by name."""
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise ValueError(f"Prompt not found: {name}")
    return path.read_text(encoding="utf-8")

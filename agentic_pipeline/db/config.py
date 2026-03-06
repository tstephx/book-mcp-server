"""Database configuration."""

import os
import warnings
from pathlib import Path

# Default path to shared library.db (override with AGENTIC_PIPELINE_DB env var)
DEFAULT_DB_PATH = Path.home() / "Library" / "Application Support" / "book-library" / "library.db"


def get_db_path() -> Path:
    """Get the database path, with environment override support."""
    env_path = os.environ.get("AGENTIC_PIPELINE_DB")
    if env_path:
        return Path(env_path)

    if not DEFAULT_DB_PATH.exists():
        warnings.warn(
            f"Default DB path not found: {DEFAULT_DB_PATH}. Set AGENTIC_PIPELINE_DB environment variable.",
            stacklevel=2,
        )
    return DEFAULT_DB_PATH

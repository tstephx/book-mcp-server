"""Tests for database configuration."""

import pytest
from pathlib import Path


def test_get_db_path_returns_path():
    from agentic_pipeline.db.config import get_db_path

    path = get_db_path()
    assert isinstance(path, Path)
    assert path.name == "library.db"


def test_get_db_path_uses_env_override(monkeypatch):
    from agentic_pipeline.db.config import get_db_path

    monkeypatch.setenv("AGENTIC_PIPELINE_DB", "/custom/path/test.db")
    path = get_db_path()
    assert path == Path("/custom/path/test.db")

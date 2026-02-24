# tests/test_config.py
"""Tests for orchestrator configuration."""

import pytest
import os


def test_config_loads_defaults():
    from agentic_pipeline.config import OrchestratorConfig

    config = OrchestratorConfig()

    assert config.processing_timeout == 600
    assert config.embedding_timeout == 300
    assert config.confidence_threshold == 0.7
    assert config.worker_poll_interval == 5
    assert config.max_retry_attempts == 3


def test_config_from_env(monkeypatch):
    from agentic_pipeline.config import OrchestratorConfig

    monkeypatch.setenv("PROCESSING_TIMEOUT_SECONDS", "900")
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.8")

    config = OrchestratorConfig.from_env()

    assert config.processing_timeout == 900
    assert config.confidence_threshold == 0.8


def test_config_has_no_book_ingestion_path():
    from agentic_pipeline.config import OrchestratorConfig

    config = OrchestratorConfig()

    assert not hasattr(config, "book_ingestion_path")


def test_config_watch_dir_default_none():
    from agentic_pipeline.config import OrchestratorConfig

    config = OrchestratorConfig()
    assert config.watch_dir is None


def test_config_watch_dir_from_env(monkeypatch):
    from pathlib import Path
    from agentic_pipeline.config import OrchestratorConfig

    monkeypatch.setenv("WATCH_DIR", "/tmp/books")
    config = OrchestratorConfig.from_env()
    assert config.watch_dir == Path("/tmp/books")


def test_config_watch_dir_not_set_in_env(monkeypatch):
    from agentic_pipeline.config import OrchestratorConfig

    monkeypatch.delenv("WATCH_DIR", raising=False)
    config = OrchestratorConfig.from_env()
    assert config.watch_dir is None


def test_default_chapter_tokens_exists():
    from src.config import Config
    assert hasattr(Config, 'DEFAULT_CHAPTER_TOKENS')
    assert Config.DEFAULT_CHAPTER_TOKENS == 8000

def test_default_chapter_tokens_env_override():
    import os
    from importlib import reload
    import src.config as config_module
    os.environ['DEFAULT_CHAPTER_TOKENS'] = '4000'
    reload(config_module)
    assert config_module.Config.DEFAULT_CHAPTER_TOKENS == 4000
    del os.environ['DEFAULT_CHAPTER_TOKENS']
    reload(config_module)

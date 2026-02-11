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

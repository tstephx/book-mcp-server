# tests/test_orchestrator_logging.py
"""Tests for pipeline logging."""

import pytest
import json
import logging


def test_pipeline_logger_state_transition(caplog):
    from agentic_pipeline.orchestrator.logging import PipelineLogger

    logger = PipelineLogger()

    with caplog.at_level(logging.INFO):
        logger.state_transition("abc123", "CLASSIFYING", "SELECTING_STRATEGY")

    assert "state_transition" in caplog.text
    assert "abc123" in caplog.text
    assert "CLASSIFYING" in caplog.text


def test_pipeline_logger_error(caplog):
    from agentic_pipeline.orchestrator.logging import PipelineLogger

    logger = PipelineLogger()

    with caplog.at_level(logging.ERROR):
        logger.error("abc123", "ProcessingError", "Failed to extract text")

    assert "error" in caplog.text
    assert "ProcessingError" in caplog.text


def test_pipeline_logger_json_format(caplog):
    from agentic_pipeline.orchestrator.logging import PipelineLogger

    logger = PipelineLogger()

    with caplog.at_level(logging.INFO):
        logger.processing_started("abc123", "/path/to/book.epub")

    # Should be valid JSON
    log_line = caplog.records[0].message
    data = json.loads(log_line)
    assert data["event"] == "processing_started"
    assert data["pipeline_id"] == "abc123"

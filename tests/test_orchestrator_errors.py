# tests/test_orchestrator_errors.py
"""Tests for orchestrator error types."""

import pytest


def test_processing_error():
    from agentic_pipeline.orchestrator.errors import ProcessingError

    error = ProcessingError("Failed to extract text", exit_code=1)

    assert str(error) == "Failed to extract text"
    assert error.exit_code == 1


def test_embedding_error():
    from agentic_pipeline.orchestrator.errors import EmbeddingError

    error = EmbeddingError("Model not found")

    assert str(error) == "Model not found"


def test_timeout_error():
    from agentic_pipeline.orchestrator.errors import PipelineTimeoutError

    error = PipelineTimeoutError("Processing exceeded 600s", timeout=600)

    assert error.timeout == 600

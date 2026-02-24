"""Tests for pipeline states."""

import pytest


def test_pipeline_state_enum_has_required_states():
    from agentic_pipeline.pipeline.states import PipelineState

    required = [
        "DETECTED", "HASHING", "DUPLICATE", "CLASSIFYING",
        "SELECTING_STRATEGY", "PROCESSING", "VALIDATING",
        "PENDING_APPROVAL", "NEEDS_RETRY", "APPROVED",
        "EMBEDDING", "COMPLETE", "REJECTED", "ARCHIVED", "FAILED"
    ]

    for state in required:
        assert hasattr(PipelineState, state), f"Missing state: {state}"


def test_can_transition_allows_valid_transitions():
    from agentic_pipeline.pipeline.states import PipelineState, can_transition

    assert can_transition(PipelineState.DETECTED, PipelineState.HASHING)
    assert can_transition(PipelineState.HASHING, PipelineState.CLASSIFYING)
    assert can_transition(PipelineState.PENDING_APPROVAL, PipelineState.APPROVED)


def test_can_transition_blocks_invalid_transitions():
    from agentic_pipeline.pipeline.states import PipelineState, can_transition

    # Can't go backwards
    assert not can_transition(PipelineState.COMPLETE, PipelineState.DETECTED)
    # Can't skip steps
    assert not can_transition(PipelineState.DETECTED, PipelineState.COMPLETE)


def test_is_terminal_state():
    from agentic_pipeline.pipeline.states import PipelineState, is_terminal_state

    assert is_terminal_state(PipelineState.COMPLETE)
    assert is_terminal_state(PipelineState.REJECTED)
    assert is_terminal_state(PipelineState.ARCHIVED)
    assert is_terminal_state(PipelineState.DUPLICATE)

    assert not is_terminal_state(PipelineState.PROCESSING)
    assert not is_terminal_state(PipelineState.PENDING_APPROVAL)


def test_pipeline_state_has_failed():
    from agentic_pipeline.pipeline.states import PipelineState
    assert hasattr(PipelineState, "FAILED")
    assert PipelineState.FAILED.value == "failed"


def test_failed_is_terminal():
    from agentic_pipeline.pipeline.states import PipelineState, is_terminal_state
    assert is_terminal_state(PipelineState.FAILED)


def test_needs_retry_can_transition_to_failed():
    from agentic_pipeline.pipeline.states import PipelineState, can_transition
    assert can_transition(PipelineState.NEEDS_RETRY, PipelineState.FAILED)


def test_failed_has_no_outgoing_transitions():
    from agentic_pipeline.pipeline.states import PipelineState, can_transition
    for state in PipelineState:
        assert not can_transition(PipelineState.FAILED, state)

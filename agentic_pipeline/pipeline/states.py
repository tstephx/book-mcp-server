"""Pipeline state definitions and transitions."""

from enum import Enum


class PipelineState(Enum):
    """States a book can be in during processing."""

    DETECTED = "detected"
    HASHING = "hashing"
    DUPLICATE = "duplicate"
    CLASSIFYING = "classifying"
    SELECTING_STRATEGY = "selecting_strategy"
    PROCESSING = "processing"
    VALIDATING = "validating"
    PENDING_APPROVAL = "pending_approval"
    NEEDS_RETRY = "needs_retry"
    APPROVED = "approved"
    EMBEDDING = "embedding"
    COMPLETE = "complete"
    REJECTED = "rejected"
    ARCHIVED = "archived"


# Valid state transitions
TRANSITIONS = {
    PipelineState.DETECTED: {PipelineState.HASHING},
    PipelineState.HASHING: {PipelineState.CLASSIFYING, PipelineState.DUPLICATE},
    PipelineState.DUPLICATE: set(),  # Terminal
    PipelineState.CLASSIFYING: {PipelineState.SELECTING_STRATEGY, PipelineState.REJECTED},
    PipelineState.SELECTING_STRATEGY: {PipelineState.PROCESSING},
    PipelineState.PROCESSING: {PipelineState.VALIDATING, PipelineState.NEEDS_RETRY, PipelineState.REJECTED},
    PipelineState.VALIDATING: {PipelineState.PENDING_APPROVAL, PipelineState.NEEDS_RETRY, PipelineState.REJECTED},
    PipelineState.PENDING_APPROVAL: {PipelineState.APPROVED, PipelineState.REJECTED, PipelineState.NEEDS_RETRY},
    PipelineState.NEEDS_RETRY: {PipelineState.PROCESSING, PipelineState.REJECTED},
    PipelineState.APPROVED: {PipelineState.EMBEDDING},
    PipelineState.EMBEDDING: {PipelineState.COMPLETE, PipelineState.REJECTED, PipelineState.NEEDS_RETRY},
    PipelineState.COMPLETE: {PipelineState.ARCHIVED},  # Can archive completed books
    PipelineState.REJECTED: {PipelineState.ARCHIVED},
    PipelineState.ARCHIVED: set(),  # Terminal
}

TERMINAL_STATES = {
    PipelineState.COMPLETE,
    PipelineState.REJECTED,
    PipelineState.ARCHIVED,
    PipelineState.DUPLICATE,
}


def can_transition(from_state: PipelineState, to_state: PipelineState) -> bool:
    """Check if a state transition is valid."""
    return to_state in TRANSITIONS.get(from_state, set())


def is_terminal_state(state: PipelineState) -> bool:
    """Check if a state is terminal (no further transitions possible)."""
    return state in TERMINAL_STATES

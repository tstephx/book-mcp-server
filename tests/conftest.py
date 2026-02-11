"""Shared test helpers."""

from agentic_pipeline.db.pipelines import PipelineRepository
from agentic_pipeline.pipeline.states import PipelineState


# Shortest valid paths to reach each state from DETECTED
_PATHS_TO_STATE = {
    PipelineState.DETECTED: [],
    PipelineState.HASHING: [PipelineState.HASHING],
    PipelineState.CLASSIFYING: [PipelineState.HASHING, PipelineState.CLASSIFYING],
    PipelineState.SELECTING_STRATEGY: [
        PipelineState.HASHING, PipelineState.CLASSIFYING,
        PipelineState.SELECTING_STRATEGY,
    ],
    PipelineState.PROCESSING: [
        PipelineState.HASHING, PipelineState.CLASSIFYING,
        PipelineState.SELECTING_STRATEGY, PipelineState.PROCESSING,
    ],
    PipelineState.VALIDATING: [
        PipelineState.HASHING, PipelineState.CLASSIFYING,
        PipelineState.SELECTING_STRATEGY, PipelineState.PROCESSING,
        PipelineState.VALIDATING,
    ],
    PipelineState.PENDING_APPROVAL: [
        PipelineState.HASHING, PipelineState.CLASSIFYING,
        PipelineState.SELECTING_STRATEGY, PipelineState.PROCESSING,
        PipelineState.VALIDATING, PipelineState.PENDING_APPROVAL,
    ],
    PipelineState.APPROVED: [
        PipelineState.HASHING, PipelineState.CLASSIFYING,
        PipelineState.SELECTING_STRATEGY, PipelineState.PROCESSING,
        PipelineState.VALIDATING, PipelineState.PENDING_APPROVAL,
        PipelineState.APPROVED,
    ],
    PipelineState.EMBEDDING: [
        PipelineState.HASHING, PipelineState.CLASSIFYING,
        PipelineState.SELECTING_STRATEGY, PipelineState.PROCESSING,
        PipelineState.VALIDATING, PipelineState.PENDING_APPROVAL,
        PipelineState.APPROVED, PipelineState.EMBEDDING,
    ],
    PipelineState.COMPLETE: [
        PipelineState.HASHING, PipelineState.CLASSIFYING,
        PipelineState.SELECTING_STRATEGY, PipelineState.PROCESSING,
        PipelineState.VALIDATING, PipelineState.PENDING_APPROVAL,
        PipelineState.APPROVED, PipelineState.EMBEDDING,
        PipelineState.COMPLETE,
    ],
    PipelineState.NEEDS_RETRY: [
        PipelineState.HASHING, PipelineState.CLASSIFYING,
        PipelineState.SELECTING_STRATEGY, PipelineState.PROCESSING,
        PipelineState.NEEDS_RETRY,
    ],
    PipelineState.REJECTED: [
        PipelineState.HASHING, PipelineState.CLASSIFYING,
        PipelineState.REJECTED,
    ],
    PipelineState.DUPLICATE: [PipelineState.HASHING, PipelineState.DUPLICATE],
}


def transition_to(repo: PipelineRepository, pipeline_id: str, target: PipelineState):
    """Transition a pipeline from DETECTED to target via valid intermediate states."""
    path = _PATHS_TO_STATE.get(target)
    if path is None:
        raise ValueError(f"No known path to {target}")
    for state in path:
        repo.update_state(pipeline_id, state)

# agentic_pipeline/orchestrator/errors.py
"""Custom error types for the orchestrator."""


class OrchestratorError(Exception):
    """Base error for orchestrator operations."""
    pass


class ProcessingError(OrchestratorError):
    """Error during book processing (text extraction, cleaning)."""

    def __init__(self, message: str, exit_code: int = None, stderr: str = None):
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr = stderr


class EmbeddingError(OrchestratorError):
    """Error during embedding generation."""

    def __init__(self, message: str, exit_code: int = None):
        super().__init__(message)
        self.exit_code = exit_code


class PipelineTimeoutError(OrchestratorError):
    """Operation exceeded timeout."""

    def __init__(self, message: str, timeout: int = None):
        super().__init__(message)
        self.timeout = timeout


class IdempotencyError(OrchestratorError):
    """Book already processed or in progress."""

    def __init__(self, message: str, existing_state: str = None):
        super().__init__(message)
        self.existing_state = existing_state

"""
Adapters for integrating external libraries with the agentic pipeline.

Imports are lazy — book_ingestion is a local-only dependency not available in CI.
"""


def __getattr__(name):
    if name == "LLMFallbackAdapter":
        from agentic_pipeline.adapters.llm_fallback_adapter import LLMFallbackAdapter

        return LLMFallbackAdapter
    if name == "ProcessingAdapter":
        from agentic_pipeline.adapters.processing_adapter import ProcessingAdapter

        return ProcessingAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LLMFallbackAdapter",
    "ProcessingAdapter",
]

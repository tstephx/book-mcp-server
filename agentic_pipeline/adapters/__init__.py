"""
Adapters for integrating external libraries with the agentic pipeline.
"""

from agentic_pipeline.adapters.llm_fallback_adapter import LLMFallbackAdapter
from agentic_pipeline.adapters.processing_adapter import ProcessingAdapter

__all__ = [
    "LLMFallbackAdapter",
    "ProcessingAdapter",
]

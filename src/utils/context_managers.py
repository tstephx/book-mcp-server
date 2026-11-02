"""Context managers for resource lifecycle management

Following MCP best practice from Chapter 6: "Context managers allow you to
allocate and release resources precisely when you want to."

These ensure proper initialization and cleanup of expensive resources.
"""

from contextlib import contextmanager
from typing import Generator
import logging

logger = logging.getLogger(__name__)


@contextmanager
def embedding_model_context() -> Generator:
    """Manage embedding model lifecycle.

    Yields an OpenAIEmbeddingGenerator that calls the OpenAI
    text-embedding-3-large API (3072-dim).

    Usage:
        with embedding_model_context() as generator:
            embedding = generator.generate("some text")

    Yields:
        OpenAIEmbeddingGenerator: Initialized embedding generator
    """
    from .openai_embeddings import OpenAIEmbeddingGenerator

    logger.debug("Initializing OpenAI embedding model context...")

    try:
        generator = OpenAIEmbeddingGenerator()
        logger.debug("OpenAI embedding model ready")
        yield generator
    except Exception as e:
        logger.error(f"Embedding model context error: {e}")
        raise
    finally:
        logger.debug("Embedding model context closed")

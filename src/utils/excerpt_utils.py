"""
Query-relevant excerpt extraction utilities

Extracts the most semantically similar paragraph to a query,
providing more relevant excerpts than static first-N-characters.
"""

import logging
import re
from typing import TYPE_CHECKING

import numpy as np

from .vector_store import cosine_similarity

if TYPE_CHECKING:
    from .openai_embeddings import OpenAIEmbeddingGenerator

logger = logging.getLogger(__name__)


def split_paragraphs(content: str, min_length: int = 50) -> list[str]:
    """Split content into paragraphs on double newlines or markdown headers

    Args:
        content: Full chapter text
        min_length: Minimum paragraph length (shorter merged with next)

    Returns:
        List of paragraph strings, filtered and merged as needed
    """
    # Split on double newlines or markdown headers
    raw_paragraphs = re.split(r'\n\n+|(?=^#{1,6}\s)', content, flags=re.MULTILINE)

    paragraphs = []
    current = ""

    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue

        # Skip pure header lines (they'll be merged with next paragraph)
        if re.match(r'^#{1,6}\s+\S', para) and '\n' not in para:
            current = para + "\n"
            continue

        # Merge with previous if current is too short
        if current:
            para = current + para
            current = ""

        if len(para) < min_length:
            current = para + " "
        else:
            paragraphs.append(para)

    # Don't lose trailing content
    if current.strip():
        if paragraphs:
            paragraphs[-1] += " " + current.strip()
        else:
            paragraphs.append(current.strip())

    return paragraphs


def extract_relevant_excerpt(
    query_embedding: np.ndarray,
    content: str,
    generator: "OpenAIEmbeddingGenerator",
    max_chars: int = 500
) -> str:
    """Extract the most query-relevant paragraph from content

    Uses semantic similarity to find the paragraph that best matches
    the query, providing more relevant excerpts than static truncation.

    Args:
        query_embedding: Pre-computed embedding for the search query
        content: Full chapter text
        generator: Embedding generator (model already loaded)
        max_chars: Maximum excerpt length

    Returns:
        Most relevant paragraph, truncated if needed with ellipsis
    """
    if not content or not content.strip():
        return "[No content]"

    paragraphs = split_paragraphs(content)

    if not paragraphs:
        # Fallback to first N chars if no paragraphs found
        excerpt = content[:max_chars].strip()
        if len(content) > max_chars:
            excerpt += "..."
        return excerpt

    # Single paragraph - no need for similarity calculation
    if len(paragraphs) == 1:
        excerpt = paragraphs[0][:max_chars].strip()
        if len(paragraphs[0]) > max_chars:
            excerpt += "..."
        return excerpt

    # Generate embeddings for all paragraphs in batch
    try:
        paragraph_embeddings = generator.generate_batch(paragraphs)
    except Exception as e:
        logger.warning(f"Failed to generate paragraph embeddings: {e}")
        # Fallback to first paragraph
        excerpt = paragraphs[0][:max_chars].strip()
        if len(paragraphs[0]) > max_chars:
            excerpt += "..."
        return excerpt

    # Find most similar paragraph
    best_idx = 0
    best_similarity = -1.0

    for i, para_embedding in enumerate(paragraph_embeddings):
        similarity = cosine_similarity(query_embedding, para_embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_idx = i

    best_paragraph = paragraphs[best_idx]

    # Truncate if needed
    if len(best_paragraph) > max_chars:
        excerpt = best_paragraph[:max_chars].strip() + "..."
    else:
        excerpt = best_paragraph

    logger.debug(
        f"Selected paragraph {best_idx + 1}/{len(paragraphs)} "
        f"(similarity: {best_similarity:.3f})"
    )

    return excerpt

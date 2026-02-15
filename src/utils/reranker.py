"""Cohere reranker with graceful fallback.

Reranks search results using Cohere's cross-encoder for more precise
relevance scoring. Falls back to original ordering if the API is
unavailable.
"""

import logging
from typing import Optional

import cohere

logger = logging.getLogger(__name__)

MODEL = "rerank-v3.5"

# Module-level lazy client
_client: Optional[cohere.ClientV2] = None


def _get_client() -> cohere.ClientV2:
    global _client
    if _client is None:
        _client = cohere.ClientV2()
    return _client


def rerank_results(
    query: str,
    results: list[dict],
    top_n: int = 10,
    enabled: bool = True,
    content_key: str = "chunk_content",
) -> list[dict]:
    """Rerank search results using Cohere cross-encoder.

    Args:
        query: The search query.
        results: List of result dicts (must have content_key field).
        top_n: Number of results to return after reranking.
        enabled: Set False to skip reranking (pass-through).
        content_key: Dict key containing the text to rerank on.

    Returns:
        Reranked (or original) results, each with optional 'rerank_score'.
    """
    if not results or not enabled:
        return results[:top_n] if results else []

    documents = [r.get(content_key, "") for r in results]

    try:
        client = _get_client()
        response = client.rerank(
            model=MODEL,
            query=query,
            documents=documents,
            top_n=top_n,
        )

        reranked = []
        for item in response.results:
            result = dict(results[item.index])
            result["rerank_score"] = item.relevance_score
            reranked.append(result)

        return reranked

    except Exception as e:
        logger.warning(f"Cohere rerank failed, using original order: {e}")
        return results[:top_n]

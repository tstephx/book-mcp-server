"""Semantic search tool — queries chunk embeddings with optional reranking."""

import logging
from typing import Optional

import numpy as np

from ..utils.openai_embeddings import OpenAIEmbeddingGenerator
from ..utils.chunk_loader import load_chunk_embeddings
from ..utils.reranker import rerank_results
from ..utils.vector_store import find_top_k
from ..schemas.tool_schemas import SemanticSearchInput

logger = logging.getLogger(__name__)

# Module-level lazy generator (avoid re-creating per call)
_generator: Optional[OpenAIEmbeddingGenerator] = None


def _get_generator() -> OpenAIEmbeddingGenerator:
    global _generator
    if _generator is None:
        _generator = OpenAIEmbeddingGenerator()
    return _generator


def register_semantic_search_tools(mcp):
    """Register semantic search tool with MCP server."""

    @mcp.tool()
    def semantic_search(
        query: str,
        limit: int = 5,
        min_similarity: float = 0.3,
        rerank: bool = True,
    ) -> dict:
        """Search books using semantic similarity

        Uses embeddings to find conceptually similar content, even if
        exact keywords don't match. Results are reranked for precision.

        Args:
            query: Search query (e.g., "container networking concepts")
            limit: Maximum results to return (1-20, default: 5)
            min_similarity: Minimum similarity score 0.0-1.0 (default: 0.3)
            rerank: Rerank results with Cohere for better precision (default: True)

        Returns:
            Dictionary with results containing book_title, chapter_title,
            chapter_number, similarity, excerpt.
        """
        try:
            try:
                validated = SemanticSearchInput(
                    query=query, limit=limit, min_similarity=min_similarity
                )
            except Exception as e:
                return {"error": f"Invalid input: {e}", "results": []}

            generator = _get_generator()
            query_embedding = generator.generate(validated.query)

            embeddings_matrix, chunk_metadata = load_chunk_embeddings()

            if embeddings_matrix is None:
                return {
                    "message": "No chunk embeddings found. Run embed-library first.",
                    "results": [],
                }

            # Over-fetch for reranking
            fetch_k = validated.limit * 3 if rerank else validated.limit
            top_results = find_top_k(
                query_embedding,
                embeddings_matrix,
                k=fetch_k,
                min_similarity=validated.min_similarity,
            )

            # Build candidate list
            candidates = []
            for idx, similarity in top_results:
                meta = chunk_metadata[idx]
                candidates.append({
                    "chapter_id": meta["chapter_id"],
                    "chunk_id": meta["chunk_id"],
                    "book_id": meta["book_id"],
                    "book_title": meta["book_title"],
                    "chapter_title": meta["chapter_title"],
                    "chapter_number": meta["chapter_number"],
                    "similarity": round(similarity, 3),
                    "chunk_content": meta["content"],
                })

            # Rerank
            if rerank and candidates:
                candidates = rerank_results(
                    validated.query,
                    candidates,
                    top_n=validated.limit,
                    content_key="chunk_content",
                )
            else:
                candidates = candidates[: validated.limit]

            # Format output
            results = []
            for r in candidates:
                results.append({
                    "book_title": r["book_title"],
                    "chapter_title": r["chapter_title"],
                    "chapter_number": r["chapter_number"],
                    "similarity": r.get("similarity", 0),
                    "rerank_score": r.get("rerank_score"),
                    "excerpt": r["chunk_content"][:500],
                })

            return {
                "query": validated.query,
                "results": results,
                "total_found": len(results),
            }

        except Exception as e:
            logger.error(f"Semantic search error: {e}", exc_info=True)
            return {"error": str(e), "results": []}

    @mcp.resource("book://semantic-context/{query}")
    async def semantic_context(query: str) -> str:
        """Provide semantic context for RAG pattern."""
        try:
            search_results = semantic_search(query, limit=3, min_similarity=0.4)
            if "error" in search_results:
                return f"Error retrieving context: {search_results['error']}"
            results = search_results.get("results", [])
            if not results:
                return f"No relevant context found for: {query}"
            parts = [f"Relevant context for '{query}':\n"]
            for i, r in enumerate(results, 1):
                parts.append(
                    f"\n[{i}] From '{r['book_title']}' - "
                    f"Chapter {r['chapter_number']}: {r['chapter_title']}\n"
                    f"(Similarity: {r['similarity']:.2f})\n"
                    f"{r['excerpt']}\n"
                )
            return "\n".join(parts)
        except Exception as e:
            logger.error(f"Semantic context error: {e}", exc_info=True)
            return f"Error: {e}"

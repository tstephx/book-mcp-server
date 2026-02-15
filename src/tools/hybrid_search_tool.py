"""Hybrid search tool combining FTS5 keyword search and semantic vector search

Uses Reciprocal Rank Fusion to merge results from both search systems,
with optional Maximal Marginal Relevance for diversity and Cohere reranking.
"""

import logging
from typing import Optional

import numpy as np

from ..utils.openai_embeddings import OpenAIEmbeddingGenerator
from ..utils.chunk_loader import load_chunk_embeddings
from ..utils.reranker import rerank_results
from ..utils.fts_search import full_text_search
from ..utils.hybrid_search import reciprocal_rank_fusion, maximal_marginal_relevance
from ..utils.vector_store import find_top_k

logger = logging.getLogger(__name__)

# Module-level lazy generator
_generator: Optional[OpenAIEmbeddingGenerator] = None


def _get_generator() -> OpenAIEmbeddingGenerator:
    global _generator
    if _generator is None:
        _generator = OpenAIEmbeddingGenerator()
    return _generator


def _best_chunk_per_chapter(
    chunk_results: list[dict],
) -> tuple[list[dict], dict[str, str]]:
    """Aggregate chunk-level semantic results to chapter level for RRF fusion.

    Takes the best-scoring chunk per chapter and returns chapter-level results
    plus a mapping of chapter_id -> best chunk content for excerpts.

    Returns:
        (chapter_results, chapter_content_map)
    """
    best_by_chapter: dict[str, dict] = {}
    content_map: dict[str, str] = {}

    for r in chunk_results:
        ch_id = r["chapter_id"]
        if ch_id not in best_by_chapter or r["similarity"] > best_by_chapter[ch_id]["similarity"]:
            best_by_chapter[ch_id] = {
                "chapter_id": ch_id,
                "book_id": r["book_id"],
                "book_title": r["book_title"],
                "chapter_number": r["chapter_number"],
                "chapter_title": r["chapter_title"],
                "similarity": r["similarity"],
            }
            content_map[ch_id] = r["chunk_content"]

    return list(best_by_chapter.values()), content_map


def register_hybrid_search_tools(mcp):
    """Register hybrid search tool with MCP server"""

    @mcp.tool()
    def hybrid_search(
        query: str,
        limit: int = 10,
        diverse: bool = False,
        min_similarity: float = 0.2,
        rerank: bool = True,
    ) -> dict:
        """Search using both keyword matching AND semantic similarity, fused together

        Combines FTS5 full-text search (exact terms, BM25 ranking) with
        semantic vector search (conceptual similarity) using Reciprocal Rank
        Fusion. Results are reranked with Cohere for precision.

        Use diverse=True to reduce redundant results from similar chapters.

        Args:
            query: Search query (e.g., "docker container networking")
            limit: Maximum results to return (1-50, default: 10)
            diverse: Apply diversity re-ranking to reduce redundancy (default: False)
            min_similarity: Minimum semantic similarity threshold (default: 0.2)
            rerank: Rerank results with Cohere for better precision (default: True)

        Returns:
            Dictionary with fused results, each showing which search systems
            found it and the combined ranking score.
        """
        try:
            if not query or not query.strip():
                return {"error": "Query cannot be empty", "results": []}

            query = query.strip()
            limit = max(1, min(50, limit))
            fetch_k = limit * 3  # Overfetch for fusion

            generator = _get_generator()
            query_embedding = generator.generate(query)

            # Load chunk embeddings
            embeddings_matrix, chunk_metadata = load_chunk_embeddings()

            if embeddings_matrix is None:
                return {
                    "error": "No chunk embeddings found. Run embed-library first.",
                    "results": []
                }

            # Run FTS search (chapter-level)
            fts_response = full_text_search(query, limit=fetch_k)
            fts_results = fts_response.get('results', [])

            # Run semantic search (chunk-level)
            semantic_top = find_top_k(
                query_embedding,
                embeddings_matrix,
                k=fetch_k,
                min_similarity=min_similarity
            )

            # Build chunk-level semantic results
            chunk_results = []
            for idx, similarity in semantic_top:
                meta = chunk_metadata[idx]
                chunk_results.append({
                    'chapter_id': meta['chapter_id'],
                    'chunk_id': meta['chunk_id'],
                    'book_id': meta['book_id'],
                    'book_title': meta['book_title'],
                    'chapter_number': meta['chapter_number'],
                    'chapter_title': meta['chapter_title'],
                    'similarity': round(similarity, 4),
                    'chunk_content': meta['content'],
                })

            # Aggregate to chapter level for RRF (best chunk per chapter)
            semantic_chapter_results, content_map = _best_chunk_per_chapter(chunk_results)

            # Fuse chapter-level results from FTS + semantic
            fused = reciprocal_rank_fusion(fts_results, semantic_chapter_results, k=60)

            # Optional diversity re-ranking via MMR
            if diverse and len(fused) > 1:
                # Build chapter-level metadata with 'id' key for MMR compat
                chapter_meta_for_mmr = [
                    {**m, "id": m["chunk_id"]}
                    for m in chunk_metadata
                ]
                fused = maximal_marginal_relevance(
                    fused,
                    embeddings_matrix,
                    chapter_meta_for_mmr,
                    query_embedding,
                    lambda_param=0.7,
                    top_k=limit
                )
            else:
                fused = fused[:limit]

            # Add chunk content as excerpt + rerank
            for r in fused:
                r["chunk_content"] = content_map.get(
                    r["chapter_id"], ""
                )

            if rerank and fused:
                fused = rerank_results(
                    query,
                    fused,
                    top_n=limit,
                    content_key="chunk_content",
                )

            # Count overlap
            fts_ids = {r['chapter_id'] for r in fts_results}
            sem_ids = {r['chapter_id'] for r in semantic_chapter_results}
            in_both = len(fts_ids & sem_ids)

            # Build output
            results = []
            for r in fused:
                results.append({
                    'chapter_id': r['chapter_id'],
                    'book_id': r['book_id'],
                    'book_title': r['book_title'],
                    'chapter_number': r['chapter_number'],
                    'chapter_title': r['chapter_title'],
                    'excerpt': r.get('chunk_content', '')[:500],
                    'rrf_score': round(r['rrf_score'], 6),
                    'rerank_score': r.get('rerank_score'),
                    'fts_rank': r.get('fts_rank'),
                    'semantic_sim': r.get('semantic_sim'),
                    'sources': r['sources'],
                })

            return {
                "query": query,
                "results": results,
                "total_found": len(results),
                "diverse": diverse,
                "fusion_stats": {
                    "fts_candidates": len(fts_results),
                    "semantic_candidates": len(semantic_chapter_results),
                    "in_both": in_both,
                }
            }

        except Exception as e:
            logger.error(f"Hybrid search error: {e}", exc_info=True)
            return {"error": str(e), "results": []}

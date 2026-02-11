"""Hybrid search tool combining FTS5 keyword search and semantic vector search

Uses Reciprocal Rank Fusion to merge results from both search systems,
with optional Maximal Marginal Relevance for diversity.
"""

import logging

import numpy as np

from ..utils.context_managers import embedding_model_context
from ..utils.embedding_loader import load_chapter_embeddings
from ..utils.excerpt_utils import extract_relevant_excerpt
from ..utils.file_utils import read_chapter_content, get_chapter_excerpt
from ..utils.fts_search import full_text_search
from ..utils.hybrid_search import reciprocal_rank_fusion, maximal_marginal_relevance
from ..utils.vector_store import find_top_k

logger = logging.getLogger(__name__)


def register_hybrid_search_tools(mcp):
    """Register hybrid search tool with MCP server"""

    @mcp.tool()
    def hybrid_search(
        query: str,
        limit: int = 10,
        diverse: bool = False,
        min_similarity: float = 0.2
    ) -> dict:
        """Search using both keyword matching AND semantic similarity, fused together

        Combines FTS5 full-text search (exact terms, BM25 ranking) with
        semantic vector search (conceptual similarity) using Reciprocal Rank
        Fusion. This finds results that either system alone would miss.

        Use diverse=True to reduce redundant results from similar chapters.

        Args:
            query: Search query (e.g., "docker container networking")
            limit: Maximum results to return (1-50, default: 10)
            diverse: Apply diversity re-ranking to reduce redundancy (default: False)
            min_similarity: Minimum semantic similarity threshold (default: 0.2)

        Returns:
            Dictionary with fused results, each showing which search systems
            found it and the combined ranking score.

        Examples:
            hybrid_search("docker networking")
            hybrid_search("error handling patterns", limit=5, diverse=True)
            hybrid_search("kubernetes deployment", min_similarity=0.3)
        """
        try:
            if not query or not query.strip():
                return {"error": "Query cannot be empty", "results": []}

            query = query.strip()
            limit = max(1, min(50, limit))
            fetch_k = limit * 3  # Overfetch for fusion

            with embedding_model_context() as generator:
                # Generate query embedding
                query_embedding = generator.generate(query)

                # Load embeddings from cache or database
                embeddings_matrix, chapter_metadata = load_chapter_embeddings()

                if embeddings_matrix is None:
                    return {
                        "error": "No embeddings found. Run refresh_embeddings() first.",
                        "results": []
                    }

                # Run FTS search
                fts_response = full_text_search(query, limit=fetch_k)
                fts_results = fts_response.get('results', [])

                # Run semantic search
                semantic_top = find_top_k(
                    query_embedding,
                    embeddings_matrix,
                    k=fetch_k,
                    min_similarity=min_similarity
                )

                # Map semantic indices to result dicts
                semantic_results = []
                for idx, similarity in semantic_top:
                    meta = chapter_metadata[idx]
                    semantic_results.append({
                        'chapter_id': meta['id'],
                        'book_id': meta['book_id'],
                        'book_title': meta['book_title'],
                        'chapter_number': meta['chapter_number'],
                        'chapter_title': meta['chapter_title'],
                        'similarity': round(similarity, 4),
                    })

                # Fuse results
                fused = reciprocal_rank_fusion(fts_results, semantic_results, k=60)

                # Optional diversity re-ranking
                if diverse and len(fused) > 1:
                    fused = maximal_marginal_relevance(
                        fused,
                        embeddings_matrix,
                        chapter_metadata,
                        query_embedding,
                        lambda_param=0.7,
                        top_k=limit
                    )
                else:
                    fused = fused[:limit]

                # Count overlap
                fts_ids = {r['chapter_id'] for r in fts_results}
                sem_ids = {r['chapter_id'] for r in semantic_results}
                in_both = len(fts_ids & sem_ids)

                # Build file_path lookup from metadata for excerpt extraction
                file_path_lookup = {m['id']: m['file_path'] for m in chapter_metadata}

                # Add excerpts
                results = []
                for r in fused:
                    file_path = file_path_lookup.get(r['chapter_id'])
                    excerpt = ""
                    if file_path:
                        try:
                            content = read_chapter_content(file_path)
                            excerpt = extract_relevant_excerpt(
                                query_embedding, content, generator, max_chars=500
                            )
                        except Exception as e:
                            logger.warning(f"Excerpt extraction failed for {r['chapter_id']}: {e}")
                            try:
                                excerpt = get_chapter_excerpt(file_path, max_chars=300)
                            except Exception:
                                excerpt = ""

                    results.append({
                        'chapter_id': r['chapter_id'],
                        'book_id': r['book_id'],
                        'book_title': r['book_title'],
                        'chapter_number': r['chapter_number'],
                        'chapter_title': r['chapter_title'],
                        'excerpt': excerpt,
                        'rrf_score': round(r['rrf_score'], 6),
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
                    "semantic_candidates": len(semantic_results),
                    "in_both": in_both,
                }
            }

        except Exception as e:
            logger.error(f"Hybrid search error: {e}", exc_info=True)
            return {"error": str(e), "results": []}

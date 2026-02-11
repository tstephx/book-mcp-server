"""
Hybrid Search: Reciprocal Rank Fusion + Maximal Marginal Relevance

Combines FTS5 keyword search and semantic vector search into a single
ranked result set. MMR provides diversity to reduce redundant results.
"""

import logging

import numpy as np

from .vector_store import batch_cosine_similarity

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    fts_results: list[dict],
    semantic_results: list[dict],
    k: int = 60
) -> list[dict]:
    """Fuse FTS and semantic search results using Reciprocal Rank Fusion.

    RRF assigns each result a score based on its rank position in each
    result list, then sums across lists. This avoids the scale mismatch
    between BM25 scores and cosine similarities.

    Formula: rrf_score = sum(1 / (k + rank_position)) for each list

    Args:
        fts_results: FTS results with keys: chapter_id, book_id, book_title,
            chapter_number, chapter_title, rank (BM25 score, higher=better)
        semantic_results: Semantic results with keys: chapter_id, book_id,
            book_title, chapter_number, chapter_title, similarity (0-1)
        k: Smoothing constant (default: 60). Higher k reduces the impact
            of high rankings in individual lists.

    Returns:
        Fused results sorted by rrf_score descending, each containing:
        chapter_id, book_id, book_title, chapter_number, chapter_title,
        rrf_score, fts_rank, semantic_sim, sources
    """
    # Build lookup by chapter_id
    chapter_data: dict[str, dict] = {}

    # Process FTS results (already sorted by rank — higher BM25 = better match)
    for position, result in enumerate(fts_results, start=1):
        cid = result['chapter_id']
        rrf_contribution = 1.0 / (k + position)

        chapter_data[cid] = {
            'chapter_id': cid,
            'book_id': result.get('book_id', ''),
            'book_title': result.get('book_title', ''),
            'chapter_number': result.get('chapter_number', 0),
            'chapter_title': result.get('chapter_title', ''),
            'rrf_score': rrf_contribution,
            'fts_rank': result.get('rank'),
            'semantic_sim': None,
            'sources': ['fts'],
        }

    # Process semantic results (already sorted by similarity descending)
    for position, result in enumerate(semantic_results, start=1):
        cid = result['chapter_id']
        rrf_contribution = 1.0 / (k + position)

        if cid in chapter_data:
            # Chapter found in both — add semantic contribution
            chapter_data[cid]['rrf_score'] += rrf_contribution
            chapter_data[cid]['semantic_sim'] = result.get('similarity')
            chapter_data[cid]['sources'].append('semantic')
        else:
            chapter_data[cid] = {
                'chapter_id': cid,
                'book_id': result.get('book_id', ''),
                'book_title': result.get('book_title', ''),
                'chapter_number': result.get('chapter_number', 0),
                'chapter_title': result.get('chapter_title', ''),
                'rrf_score': rrf_contribution,
                'fts_rank': None,
                'semantic_sim': result.get('similarity'),
                'sources': ['semantic'],
            }

    # Sort by rrf_score descending
    fused = sorted(chapter_data.values(), key=lambda x: x['rrf_score'], reverse=True)

    logger.debug(
        f"RRF fusion: {len(fts_results)} FTS + {len(semantic_results)} semantic "
        f"= {len(fused)} unique results"
    )

    return fused


def maximal_marginal_relevance(
    results: list[dict],
    embeddings_matrix: np.ndarray,
    metadata: list[dict],
    query_embedding: np.ndarray,
    lambda_param: float = 0.7,
    top_k: int = 10
) -> list[dict]:
    """Re-rank results using Maximal Marginal Relevance for diversity.

    Iteratively selects results that balance relevance to the query with
    novelty relative to already-selected results.

    MMR = lambda * sim(query, candidate) - (1-lambda) * max(sim(candidate, selected))

    Args:
        results: Pre-ranked results from RRF (must have chapter_id)
        embeddings_matrix: Full chapter embeddings matrix (n_chapters x dim)
        metadata: Chapter metadata list aligned with embeddings_matrix rows.
            Each dict must have 'id' key matching chapter_id.
        query_embedding: Query vector (1D)
        lambda_param: Balance between relevance (1.0) and diversity (0.0).
            Default 0.7 favors relevance slightly.
        top_k: Number of results to select

    Returns:
        Re-ranked subset of results, length min(top_k, len(results))
    """
    if not results:
        return []

    if len(results) == 1:
        return results[:1]

    top_k = min(top_k, len(results))

    # Build index mapping: chapter_id -> row index in embeddings_matrix
    metadata_index = {}
    for idx, meta in enumerate(metadata):
        metadata_index[meta['id']] = idx

    # Filter to results that have embeddings
    candidate_indices = []
    candidate_results = []
    for r in results:
        emb_idx = metadata_index.get(r['chapter_id'])
        if emb_idx is not None:
            candidate_indices.append(emb_idx)
            candidate_results.append(r)

    if not candidate_results:
        return results[:top_k]

    # Precompute query-to-candidate similarities
    candidate_embeddings = embeddings_matrix[candidate_indices]
    query_sims = batch_cosine_similarity(query_embedding, candidate_embeddings)

    selected: list[int] = []  # indices into candidate_* lists
    remaining = set(range(len(candidate_results)))

    # Pre-allocate buffer for selected embeddings to avoid repeated np.vstack
    emb_dim = candidate_embeddings.shape[1]
    selected_buffer = np.empty((top_k, emb_dim), dtype=candidate_embeddings.dtype)
    n_selected = 0

    for _ in range(top_k):
        if not remaining:
            break

        best_score = -float('inf')
        best_idx = -1

        for idx in remaining:
            # Relevance: similarity to query
            relevance = float(query_sims[idx])

            # Diversity: max similarity to any already-selected result
            if n_selected > 0:
                inter_sims = batch_cosine_similarity(
                    candidate_embeddings[idx], selected_buffer[:n_selected]
                )
                max_inter_sim = float(np.max(inter_sims))
            else:
                max_inter_sim = 0.0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_inter_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx == -1:
            break

        selected.append(best_idx)
        selected_buffer[n_selected] = candidate_embeddings[best_idx]
        n_selected += 1
        remaining.discard(best_idx)

    return [candidate_results[i] for i in selected]

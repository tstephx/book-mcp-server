"""Tests for hybrid search algorithms: RRF fusion and MMR diversity"""

import numpy as np
import pytest

from src.utils.hybrid_search import reciprocal_rank_fusion, maximal_marginal_relevance


# ─── Helpers ──────────────────────────────────────────────────────────

def make_fts_result(chapter_id, rank=1.0, **kwargs):
    """Create a synthetic FTS result dict"""
    return {
        'chapter_id': chapter_id,
        'book_id': kwargs.get('book_id', f'book-{chapter_id}'),
        'book_title': kwargs.get('book_title', f'Book {chapter_id}'),
        'chapter_number': kwargs.get('chapter_number', 1),
        'chapter_title': kwargs.get('chapter_title', f'Chapter {chapter_id}'),
        'rank': rank,
    }


def make_semantic_result(chapter_id, similarity=0.5, **kwargs):
    """Create a synthetic semantic result dict"""
    return {
        'chapter_id': chapter_id,
        'book_id': kwargs.get('book_id', f'book-{chapter_id}'),
        'book_title': kwargs.get('book_title', f'Book {chapter_id}'),
        'chapter_number': kwargs.get('chapter_number', 1),
        'chapter_title': kwargs.get('chapter_title', f'Chapter {chapter_id}'),
        'similarity': similarity,
    }


# ─── RRF Tests ────────────────────────────────────────────────────────

class TestReciprocalRankFusion:

    def test_rrf_both_sources(self):
        """Chapter in both FTS and semantic should rank highest"""
        fts = [
            make_fts_result('ch-A', rank=10.0),
            make_fts_result('ch-B', rank=5.0),
        ]
        sem = [
            make_semantic_result('ch-A', similarity=0.9),
            make_semantic_result('ch-C', similarity=0.8),
        ]

        results = reciprocal_rank_fusion(fts, sem)

        # ch-A appears in both lists, so it should have the highest score
        assert results[0]['chapter_id'] == 'ch-A'
        assert results[0]['rrf_score'] > results[1]['rrf_score']

    def test_rrf_single_source_fts_only(self):
        """Chapter in only FTS still appears in results"""
        fts = [make_fts_result('ch-A', rank=10.0)]
        sem = [make_semantic_result('ch-B', similarity=0.9)]

        results = reciprocal_rank_fusion(fts, sem)

        chapter_ids = [r['chapter_id'] for r in results]
        assert 'ch-A' in chapter_ids
        assert 'ch-B' in chapter_ids

        # ch-A should have fts_rank set and semantic_sim None
        ch_a = next(r for r in results if r['chapter_id'] == 'ch-A')
        assert ch_a['fts_rank'] == 10.0
        assert ch_a['semantic_sim'] is None

    def test_rrf_single_source_semantic_only(self):
        """Chapter in only semantic still appears in results"""
        fts = [make_fts_result('ch-A')]
        sem = [make_semantic_result('ch-B', similarity=0.9)]

        results = reciprocal_rank_fusion(fts, sem)

        ch_b = next(r for r in results if r['chapter_id'] == 'ch-B')
        assert ch_b['fts_rank'] is None
        assert ch_b['semantic_sim'] == 0.9

    def test_rrf_empty_fts(self):
        """Empty FTS results handled gracefully"""
        sem = [make_semantic_result('ch-A', similarity=0.9)]
        results = reciprocal_rank_fusion([], sem)

        assert len(results) == 1
        assert results[0]['chapter_id'] == 'ch-A'
        assert results[0]['sources'] == ['semantic']

    def test_rrf_empty_semantic(self):
        """Empty semantic results handled gracefully"""
        fts = [make_fts_result('ch-A', rank=10.0)]
        results = reciprocal_rank_fusion(fts, [])

        assert len(results) == 1
        assert results[0]['chapter_id'] == 'ch-A'
        assert results[0]['sources'] == ['fts']

    def test_rrf_both_empty(self):
        """Both empty returns empty"""
        results = reciprocal_rank_fusion([], [])
        assert results == []

    def test_rrf_k_parameter(self):
        """Different k values change relative scoring"""
        fts = [make_fts_result('ch-A')]
        sem = [make_semantic_result('ch-B')]

        results_k1 = reciprocal_rank_fusion(fts, sem, k=1)
        results_k60 = reciprocal_rank_fusion(fts, sem, k=60)

        # With k=1, rank 1 gets 1/(1+1) = 0.5
        # With k=60, rank 1 gets 1/(60+1) ≈ 0.0164
        score_k1 = results_k1[0]['rrf_score']
        score_k60 = results_k60[0]['rrf_score']

        assert score_k1 > score_k60

    def test_rrf_ordering(self):
        """Results sorted by rrf_score descending"""
        fts = [
            make_fts_result('ch-A', rank=10.0),
            make_fts_result('ch-B', rank=5.0),
            make_fts_result('ch-C', rank=3.0),
        ]
        sem = [
            make_semantic_result('ch-C', similarity=0.95),
            make_semantic_result('ch-A', similarity=0.85),
        ]

        results = reciprocal_rank_fusion(fts, sem)
        scores = [r['rrf_score'] for r in results]

        # Verify descending order
        assert scores == sorted(scores, reverse=True)

    def test_rrf_sources_field(self):
        """Sources field correctly reflects which systems found each result"""
        fts = [make_fts_result('ch-A'), make_fts_result('ch-B')]
        sem = [make_semantic_result('ch-A'), make_semantic_result('ch-C')]

        results = reciprocal_rank_fusion(fts, sem)

        ch_a = next(r for r in results if r['chapter_id'] == 'ch-A')
        ch_b = next(r for r in results if r['chapter_id'] == 'ch-B')
        ch_c = next(r for r in results if r['chapter_id'] == 'ch-C')

        assert sorted(ch_a['sources']) == ['fts', 'semantic']
        assert ch_b['sources'] == ['fts']
        assert ch_c['sources'] == ['semantic']

    def test_rrf_preserves_metadata(self):
        """Metadata from first source encountered is preserved"""
        fts = [make_fts_result('ch-A', book_id='book-1', book_title='My Book',
                               chapter_number=3, chapter_title='Chapter Three')]
        sem = []

        results = reciprocal_rank_fusion(fts, sem)

        assert results[0]['book_id'] == 'book-1'
        assert results[0]['book_title'] == 'My Book'
        assert results[0]['chapter_number'] == 3
        assert results[0]['chapter_title'] == 'Chapter Three'


# ─── MMR Tests ────────────────────────────────────────────────────────

class TestMaximalMarginalRelevance:
    """Tests for MMR diversity re-ranking"""

    @pytest.fixture
    def embedding_dim(self):
        return 8

    @pytest.fixture
    def diverse_setup(self, embedding_dim):
        """Create embeddings with clearly separated clusters

        Cluster A: ch-0, ch-1 (direction [1,0,0,...])
        Cluster B: ch-2, ch-3 (direction [0,1,0,...])
        Unique:    ch-4       (direction [0,0,1,...])
        Query:     close to cluster A
        """
        # Use orthogonal directions for clear cluster separation
        base_a = np.zeros(embedding_dim, dtype=np.float32)
        base_a[0] = 1.0

        base_b = np.zeros(embedding_dim, dtype=np.float32)
        base_b[1] = 1.0

        unique = np.zeros(embedding_dim, dtype=np.float32)
        unique[2] = 1.0

        np.random.seed(42)
        noise = 0.05  # Small noise to keep cluster members very similar

        embeddings = np.vstack([
            base_a + noise * np.random.randn(embedding_dim),  # ch-0
            base_a + noise * np.random.randn(embedding_dim),  # ch-1 (similar to ch-0)
            base_b + noise * np.random.randn(embedding_dim),  # ch-2
            base_b + noise * np.random.randn(embedding_dim),  # ch-3 (similar to ch-2)
            unique + noise * np.random.randn(embedding_dim),  # ch-4
        ]).astype(np.float32)

        metadata = [{'id': f'ch-{i}'} for i in range(5)]

        # Query is close to cluster A
        query = (base_a + 0.01 * np.random.randn(embedding_dim)).astype(np.float32)

        results = [
            {'chapter_id': f'ch-{i}', 'rrf_score': 1.0 / (60 + i + 1)}
            for i in range(5)
        ]

        return embeddings, metadata, query, results

    def test_mmr_reduces_redundancy(self, diverse_setup):
        """MMR should avoid selecting both items from a similar cluster"""
        embeddings, metadata, query, results = diverse_setup

        # lambda=0.3 gives diversity (0.7) more weight than relevance (0.3),
        # so selecting a near-duplicate gets heavily penalized
        mmr_results = maximal_marginal_relevance(
            results, embeddings, metadata, query,
            lambda_param=0.3, top_k=3
        )

        selected_ids = [r['chapter_id'] for r in mmr_results]

        # Should NOT select both ch-0 and ch-1 (nearly identical)
        cluster_a_count = sum(1 for cid in selected_ids if cid in ('ch-0', 'ch-1'))
        assert cluster_a_count <= 1, f"Selected both from cluster A: {selected_ids}"

    def test_mmr_lambda_1_pure_relevance(self, diverse_setup):
        """lambda=1.0 should produce same order as pure relevance (no diversity)"""
        embeddings, metadata, query, results = diverse_setup

        mmr_results = maximal_marginal_relevance(
            results, embeddings, metadata, query,
            lambda_param=1.0, top_k=5
        )

        # With lambda=1.0, diversity term is 0, so order is purely by
        # query similarity. The first selected should be the most similar to query.
        # Since query is close to cluster A (ch-0, ch-1), those should come first.
        first_id = mmr_results[0]['chapter_id']
        assert first_id in ('ch-0', 'ch-1')

    def test_mmr_lambda_0_max_diversity(self, diverse_setup):
        """lambda=0.0 should maximize diversity (spread across clusters)"""
        embeddings, metadata, query, results = diverse_setup

        mmr_results = maximal_marginal_relevance(
            results, embeddings, metadata, query,
            lambda_param=0.0, top_k=3
        )

        selected_ids = [r['chapter_id'] for r in mmr_results]

        # With max diversity, should pick from different clusters
        # Should not pick both from cluster A or both from cluster B
        cluster_a = sum(1 for cid in selected_ids if cid in ('ch-0', 'ch-1'))
        cluster_b = sum(1 for cid in selected_ids if cid in ('ch-2', 'ch-3'))
        assert cluster_a <= 1
        assert cluster_b <= 1

    def test_mmr_single_result(self):
        """Single result returns it unchanged"""
        embeddings = np.random.randn(1, 8).astype(np.float32)
        metadata = [{'id': 'ch-0'}]
        query = np.random.randn(8).astype(np.float32)
        results = [{'chapter_id': 'ch-0', 'rrf_score': 0.5}]

        mmr_results = maximal_marginal_relevance(
            results, embeddings, metadata, query, top_k=5
        )

        assert len(mmr_results) == 1
        assert mmr_results[0]['chapter_id'] == 'ch-0'

    def test_mmr_empty_results(self):
        """Empty results returns empty"""
        embeddings = np.random.randn(3, 8).astype(np.float32)
        metadata = [{'id': f'ch-{i}'} for i in range(3)]
        query = np.random.randn(8).astype(np.float32)

        mmr_results = maximal_marginal_relevance(
            [], embeddings, metadata, query, top_k=5
        )

        assert mmr_results == []

    def test_mmr_top_k_limits_output(self, diverse_setup):
        """top_k parameter limits the number of returned results"""
        embeddings, metadata, query, results = diverse_setup

        mmr_results = maximal_marginal_relevance(
            results, embeddings, metadata, query, top_k=2
        )

        assert len(mmr_results) == 2

    def test_mmr_missing_embeddings_graceful(self):
        """Results with chapter_ids not in metadata are handled gracefully"""
        embeddings = np.random.randn(2, 8).astype(np.float32)
        metadata = [{'id': 'ch-0'}, {'id': 'ch-1'}]
        query = np.random.randn(8).astype(np.float32)

        # ch-99 is not in metadata
        results = [
            {'chapter_id': 'ch-0', 'rrf_score': 0.5},
            {'chapter_id': 'ch-99', 'rrf_score': 0.4},
            {'chapter_id': 'ch-1', 'rrf_score': 0.3},
        ]

        mmr_results = maximal_marginal_relevance(
            results, embeddings, metadata, query, top_k=3
        )

        # ch-99 should be skipped since it has no embedding
        selected_ids = [r['chapter_id'] for r in mmr_results]
        assert 'ch-99' not in selected_ids
        assert len(mmr_results) == 2

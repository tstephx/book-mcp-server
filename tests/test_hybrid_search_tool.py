"""Integration tests for the hybrid search MCP tool

These tests mock the embedding generator and database to test the tool's
orchestration logic without requiring a real library database.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ─── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def mock_chunk_embeddings():
    """Create mock chunk embeddings matrix and metadata"""
    np.random.seed(42)
    n_chunks = 10
    dim = 1536

    matrix = np.random.randn(n_chunks, dim).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / norms

    metadata = [
        {
            'chunk_id': f'ch-{i}:0',
            'chapter_id': f'ch-{i}',
            'book_id': f'book-{i % 3}',
            'book_title': f'Book {i % 3}',
            'chapter_title': f'Chapter {i}',
            'chapter_number': i + 1,
            'chunk_index': 0,
            'content': f'Content of chunk {i}',
            'file_path': f'/tmp/books/ch-{i}.md',
        }
        for i in range(n_chunks)
    ]

    return matrix, metadata


@pytest.fixture
def mock_generator():
    """Mock OpenAI embedding generator"""
    gen = MagicMock()
    np.random.seed(123)
    gen.generate.return_value = np.random.randn(1536).astype(np.float32)
    return gen


@pytest.fixture
def mock_fts_results():
    """Sample FTS results"""
    return {
        'query': 'docker networking',
        'results': [
            {
                'chapter_id': 'ch-0',
                'book_id': 'book-0',
                'book_title': 'Book 0',
                'chapter_number': 1,
                'chapter_title': 'Chapter 0',
                'rank': 5.2,
                'excerpt': 'Docker **networking** concepts...',
            },
            {
                'chapter_id': 'ch-3',
                'book_id': 'book-0',
                'book_title': 'Book 0',
                'chapter_number': 4,
                'chapter_title': 'Chapter 3',
                'rank': 3.1,
                'excerpt': 'Container **networking** basics...',
            },
        ],
        'total_found': 2,
    }


def _capture_hybrid_search():
    """Register and capture the hybrid_search function."""
    from src.tools.hybrid_search_tool import register_hybrid_search_tools

    mcp = MagicMock()
    captured_fn = {}

    def capture_tool():
        def decorator(fn):
            captured_fn['hybrid_search'] = fn
            return fn
        return decorator

    mcp.tool = capture_tool
    register_hybrid_search_tools(mcp)
    return captured_fn['hybrid_search']


# ─── Tests ────────────────────────────────────────────────────────────

class TestHybridSearchTool:

    def test_returns_results(self, mock_chunk_embeddings, mock_generator, mock_fts_results):
        """Basic happy path: returns fused results"""
        matrix, metadata = mock_chunk_embeddings

        with patch('src.tools.hybrid_search_tool._get_generator') as mock_get_gen, \
             patch('src.tools.hybrid_search_tool.load_chunk_embeddings') as mock_load, \
             patch('src.tools.hybrid_search_tool.full_text_search') as mock_fts, \
             patch('src.tools.hybrid_search_tool.rerank_results') as mock_rerank:

            mock_get_gen.return_value = mock_generator
            mock_load.return_value = (matrix, metadata)
            mock_fts.return_value = mock_fts_results

            # rerank passes through with chunk_content + rerank_score
            def passthrough_rerank(query, results, top_n=10, content_key="chunk_content"):
                for r in results:
                    r['rerank_score'] = 0.8
                return results
            mock_rerank.side_effect = passthrough_rerank

            fn = _capture_hybrid_search()
            result = fn("docker networking", limit=5)

            assert 'results' in result
            assert 'fusion_stats' in result
            assert result['query'] == 'docker networking'
            assert len(result['results']) > 0
            assert result['fusion_stats']['fts_candidates'] == 2

    def test_diverse_flag(self, mock_chunk_embeddings, mock_generator, mock_fts_results):
        """diverse=True should apply MMR re-ranking"""
        matrix, metadata = mock_chunk_embeddings

        with patch('src.tools.hybrid_search_tool._get_generator') as mock_get_gen, \
             patch('src.tools.hybrid_search_tool.load_chunk_embeddings') as mock_load, \
             patch('src.tools.hybrid_search_tool.full_text_search') as mock_fts, \
             patch('src.tools.hybrid_search_tool.rerank_results') as mock_rerank, \
             patch('src.tools.hybrid_search_tool.find_top_k') as mock_topk, \
             patch('src.tools.hybrid_search_tool.maximal_marginal_relevance') as mock_mmr:

            mock_get_gen.return_value = mock_generator
            mock_load.return_value = (matrix, metadata)
            mock_fts.return_value = mock_fts_results
            # Return multiple semantic hits so fusion produces >1 results
            mock_topk.return_value = [(0, 0.9), (3, 0.85), (5, 0.7)]

            mock_mmr.return_value = [{
                'chapter_id': 'ch-0',
                'book_id': 'book-0',
                'book_title': 'Book 0',
                'chapter_number': 1,
                'chapter_title': 'Chapter 0',
                'rrf_score': 0.03,
                'fts_rank': 5.2,
                'semantic_sim': 0.8,
                'sources': ['fts', 'semantic'],
            }]

            def passthrough_rerank(query, results, top_n=10, content_key="chunk_content"):
                for r in results:
                    r['rerank_score'] = 0.8
                return results
            mock_rerank.side_effect = passthrough_rerank

            fn = _capture_hybrid_search()
            result = fn("docker networking", diverse=True, limit=5)

            assert result['diverse'] is True
            mock_mmr.assert_called_once()

    def test_empty_query(self):
        """Empty query returns error"""
        fn = _capture_hybrid_search()
        result = fn("", limit=5)
        assert 'error' in result
        assert result['results'] == []

    def test_no_embeddings(self, mock_generator):
        """Should return error when no embeddings loaded"""
        with patch('src.tools.hybrid_search_tool._get_generator') as mock_get_gen, \
             patch('src.tools.hybrid_search_tool.load_chunk_embeddings') as mock_load:

            mock_get_gen.return_value = mock_generator
            mock_load.return_value = (None, None)

            fn = _capture_hybrid_search()
            result = fn("docker networking")
            assert 'error' in result

    def test_fts_only_results(self, mock_chunk_embeddings, mock_generator):
        """When semantic search finds nothing, FTS results still come through"""
        matrix, metadata = mock_chunk_embeddings

        orthogonal = np.zeros(1536, dtype=np.float32)
        orthogonal[0] = 1.0
        mock_generator.generate.return_value = orthogonal

        fts_results = {
            'query': 'obscure keyword',
            'results': [{
                'chapter_id': 'ch-0',
                'book_id': 'book-0',
                'book_title': 'Book 0',
                'chapter_number': 1,
                'chapter_title': 'Chapter 0',
                'rank': 2.0,
                'excerpt': '...obscure keyword...',
            }],
            'total_found': 1,
        }

        with patch('src.tools.hybrid_search_tool._get_generator') as mock_get_gen, \
             patch('src.tools.hybrid_search_tool.load_chunk_embeddings') as mock_load, \
             patch('src.tools.hybrid_search_tool.full_text_search') as mock_fts, \
             patch('src.tools.hybrid_search_tool.rerank_results') as mock_rerank:

            mock_get_gen.return_value = mock_generator
            mock_load.return_value = (matrix, metadata)
            mock_fts.return_value = fts_results

            def passthrough_rerank(query, results, top_n=10, content_key="chunk_content"):
                for r in results:
                    r['rerank_score'] = 0.5
                return results
            mock_rerank.side_effect = passthrough_rerank

            fn = _capture_hybrid_search()
            result = fn("obscure keyword", min_similarity=0.9)

            assert len(result['results']) >= 1
            ids = [r['chapter_id'] for r in result['results']]
            assert 'ch-0' in ids

    def test_fusion_stats_overlap(self, mock_chunk_embeddings, mock_generator, mock_fts_results):
        """fusion_stats.in_both should count overlap correctly"""
        matrix, metadata = mock_chunk_embeddings

        with patch('src.tools.hybrid_search_tool._get_generator') as mock_get_gen, \
             patch('src.tools.hybrid_search_tool.load_chunk_embeddings') as mock_load, \
             patch('src.tools.hybrid_search_tool.full_text_search') as mock_fts, \
             patch('src.tools.hybrid_search_tool.rerank_results') as mock_rerank:

            mock_get_gen.return_value = mock_generator
            mock_load.return_value = (matrix, metadata)
            mock_fts.return_value = mock_fts_results

            def passthrough_rerank(query, results, top_n=10, content_key="chunk_content"):
                for r in results:
                    r['rerank_score'] = 0.7
                return results
            mock_rerank.side_effect = passthrough_rerank

            fn = _capture_hybrid_search()
            result = fn("docker networking", limit=10)

            stats = result['fusion_stats']
            assert 'fts_candidates' in stats
            assert 'semantic_candidates' in stats
            assert 'in_both' in stats
            assert isinstance(stats['in_both'], int)

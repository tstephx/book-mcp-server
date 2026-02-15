"""Tests that hybrid_search uses chunks and reranking."""

from unittest.mock import MagicMock, patch

import numpy as np


def test_hybrid_search_uses_chunks():
    """hybrid_search should call load_chunk_embeddings."""
    with patch("src.tools.hybrid_search_tool.load_chunk_embeddings") as mock_chunks, \
         patch("src.tools.hybrid_search_tool._get_generator") as mock_get_gen, \
         patch("src.tools.hybrid_search_tool.rerank_results") as mock_rerank, \
         patch("src.tools.hybrid_search_tool.find_top_k") as mock_topk, \
         patch("src.tools.hybrid_search_tool.full_text_search") as mock_fts, \
         patch("src.tools.hybrid_search_tool.reciprocal_rank_fusion") as mock_rrf:

        mock_gen = MagicMock()
        mock_gen.generate.return_value = np.zeros(1536)
        mock_get_gen.return_value = mock_gen

        mock_chunks.return_value = (
            np.zeros((1, 1536)),
            [{"chunk_id": "c1:0", "id": "c1:0", "book_id": "b1", "book_title": "Book",
              "chapter_title": "Ch1", "chapter_number": 1, "content": "text",
              "chunk_index": 0, "chapter_id": "c1", "file_path": "c1.md"}],
        )
        mock_topk.return_value = [(0, 0.9)]
        mock_fts.return_value = {"results": []}
        mock_rrf.return_value = [
            {"chapter_id": "c1", "book_id": "b1", "book_title": "Book",
             "chapter_title": "Ch1", "chapter_number": 1,
             "rrf_score": 0.5, "sources": ["semantic"]},
        ]
        mock_rerank.return_value = [
            {"chapter_id": "c1", "book_id": "b1", "book_title": "Book",
             "chapter_title": "Ch1", "chapter_number": 1,
             "rrf_score": 0.5, "sources": ["semantic"],
             "chunk_content": "text", "rerank_score": 0.9},
        ]

        from src.tools.hybrid_search_tool import register_hybrid_search_tools
        mcp = MagicMock()
        captured = {}

        def capture_tool():
            def decorator(fn):
                captured["hybrid_search"] = fn
                return fn
            return decorator

        mcp.tool = capture_tool
        register_hybrid_search_tools(mcp)

        result = captured["hybrid_search"]("test query", limit=5)

        mock_chunks.assert_called_once()
        assert len(result["results"]) == 1
        assert result["results"][0]["excerpt"] == "text"

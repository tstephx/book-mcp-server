"""Tests for get_topic_coverage result limiting."""
from unittest.mock import patch, MagicMock
import numpy as np


def _make_chunk_metadata(n):
    """Generate n fake chunks across n different chapters/books."""
    return [
        {
            "chunk_id": f"c{i}:0",
            "book_id": f"book-{i}",
            "book_title": f"Book {i}",
            "author": "Author",
            "chapter_id": f"chapter-{i}",
            "chapter_title": f"Chapter {i}",
            "chapter_number": 1,
            "word_count": 500,
            "excerpt": "some text",
        }
        for i in range(n)
    ]


def test_get_topic_coverage_respects_limit():
    """get_topic_coverage should return at most `limit` books."""
    n = 50
    meta = _make_chunk_metadata(n)
    embeddings = np.random.rand(n, 3072).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    with patch("src.tools.discovery_tools._load_chunk_data") as mock_data, \
         patch("src.tools.discovery_tools.embedding_model_context") as mock_ctx, \
         patch("src.tools.discovery_tools.cosine_similarity") as mock_sim:

        mock_data.return_value = (embeddings, meta)
        mock_gen = MagicMock()
        mock_gen.generate.return_value = np.random.rand(3072)
        mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_gen)
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
        mock_sim.return_value = 0.9  # all chunks pass threshold

        from src.tools.discovery_tools import register_discovery_tools
        mcp = MagicMock()
        captured = {}

        def tool_decorator(func):
            captured[func.__name__] = func
            return func

        mcp.tool.return_value = tool_decorator
        register_discovery_tools(mcp)

        result = captured["get_topic_coverage"]("python", limit=5)
        assert isinstance(result, dict)
        assert "coverage_by_book" in result
        assert len(result["coverage_by_book"]) <= 5


def test_get_topic_coverage_default_limit_is_20():
    """Default limit should be 20 books."""
    n = 40
    meta = _make_chunk_metadata(n)
    embeddings = np.random.rand(n, 3072).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    with patch("src.tools.discovery_tools._load_chunk_data") as mock_data, \
         patch("src.tools.discovery_tools.embedding_model_context") as mock_ctx, \
         patch("src.tools.discovery_tools.cosine_similarity") as mock_sim:

        mock_data.return_value = (embeddings, meta)
        mock_gen = MagicMock()
        mock_gen.generate.return_value = np.random.rand(3072)
        mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_gen)
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
        mock_sim.return_value = 0.9

        from src.tools.discovery_tools import register_discovery_tools
        mcp = MagicMock()
        captured = {}

        def tool_decorator(func):
            captured[func.__name__] = func
            return func

        mcp.tool.return_value = tool_decorator
        register_discovery_tools(mcp)

        result = captured["get_topic_coverage"]("python")  # no limit arg
        assert isinstance(result, dict)
        assert "coverage_by_book" in result
        assert len(result["coverage_by_book"]) <= 20


def test_get_topic_coverage_limit_zero_returns_one():
    """limit=0 should be treated as limit=1 (minimum)."""
    n = 10
    meta = _make_chunk_metadata(n)
    embeddings = np.random.rand(n, 3072).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    with patch("src.tools.discovery_tools._load_chunk_data") as mock_data, \
         patch("src.tools.discovery_tools.embedding_model_context") as mock_ctx, \
         patch("src.tools.discovery_tools.cosine_similarity") as mock_sim:

        mock_data.return_value = (embeddings, meta)
        mock_gen = MagicMock()
        mock_gen.generate.return_value = np.random.rand(3072)
        mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_gen)
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
        mock_sim.return_value = 0.9

        from src.tools.discovery_tools import register_discovery_tools
        mcp = MagicMock()
        captured = {}

        def tool_decorator(func):
            captured[func.__name__] = func
            return func

        mcp.tool.return_value = tool_decorator
        register_discovery_tools(mcp)

        result = captured["get_topic_coverage"]("python", limit=0)
        assert isinstance(result, dict)
        assert "coverage_by_book" in result
        assert len(result["coverage_by_book"]) >= 1  # minimum 1

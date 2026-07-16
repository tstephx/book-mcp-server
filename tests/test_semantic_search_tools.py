"""Tests that semantic_search uses chunks and reranking."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def test_semantic_search_uses_chunks():
    """semantic_search should call load_chunk_embeddings, not load_chapter_embeddings."""
    with (
        patch("src.tools.semantic_search_tools.load_chunk_embeddings") as mock_chunks,
        patch("src.tools.semantic_search_tools._get_generator") as mock_get_gen,
        patch("src.tools.semantic_search_tools.rerank_results") as mock_rerank,
        patch("src.tools.semantic_search_tools.find_top_k") as mock_topk,
    ):
        mock_gen = MagicMock()
        mock_gen.generate.return_value = np.zeros(1536)
        mock_get_gen.return_value = mock_gen

        mock_chunks.return_value = (
            np.zeros((2, 1536)),
            [
                {
                    "chunk_id": "c1:0",
                    "book_id": "b1",
                    "book_title": "Book",
                    "chapter_title": "Ch1",
                    "chapter_number": 1,
                    "content": "text",
                    "chunk_index": 0,
                    "chapter_id": "c1",
                    "file_path": "c1.md",
                },
                {
                    "chunk_id": "c2:0",
                    "book_id": "b2",
                    "book_title": "Book2",
                    "chapter_title": "Ch2",
                    "chapter_number": 2,
                    "content": "other",
                    "chunk_index": 0,
                    "chapter_id": "c2",
                    "file_path": "c2.md",
                },
            ],
        )
        mock_topk.return_value = [(0, 0.9)]
        mock_rerank.return_value = [
            {
                "chunk_id": "c1:0",
                "book_title": "Book",
                "chapter_title": "Ch1",
                "chapter_number": 1,
                "content": "text",
                "rerank_score": 0.95,
                "excerpt": "text",
                "similarity": 0.9,
            },
        ]

        from src.tools.semantic_search_tools import register_semantic_search_tools

        mcp = MagicMock()
        captured = {}

        def capture_tool():
            def decorator(fn):
                captured["semantic_search"] = fn
                return fn

            return decorator

        mcp.tool = capture_tool
        # Also capture the resource decorator
        mcp.resource = lambda uri: lambda fn: fn
        register_semantic_search_tools(mcp)

        result = captured["semantic_search"]("test query", limit=5)

        mock_chunks.assert_called_once()
        assert result["results"][0]["excerpt"] == "text"


class FakeMcp:
    def __init__(self):
        self.tools = {}
        self.resources = {}

    def tool(self):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn

        return deco

    def resource(self, pattern):
        def deco(fn):
            self.resources[pattern] = fn
            return fn

        return deco


@pytest.fixture
def semantic_tool(monkeypatch):
    from src.tools import semantic_search_tools as sst

    matrix = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=np.float32)
    meta = [
        {
            "chunk_id": "k1",
            "chapter_id": "ch1",
            "book_id": "b1",
            "book_title": "B",
            "chapter_title": "C1",
            "chapter_number": 1,
            "content": "chunk one text",
        },
        {
            "chunk_id": "k2",
            "chapter_id": "ch1",
            "book_id": "b1",
            "book_title": "B",
            "chapter_title": "C1",
            "chapter_number": 1,
            "content": "chunk two text",
        },
        {
            "chunk_id": "k3",
            "chapter_id": "ch2",
            "book_id": "b1",
            "book_title": "B",
            "chapter_title": "C2",
            "chapter_number": 2,
            "content": "chunk three text",
        },
    ]
    monkeypatch.setattr(sst, "load_chunk_embeddings", lambda: (matrix, meta))

    class FakeGen:
        def generate(self, q):
            return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(sst, "_get_generator", lambda: FakeGen())

    mcp = FakeMcp()
    sst.register_semantic_search_tools(mcp)
    return mcp.tools["semantic_search"]


def test_adjacent_overlapping_chunks_do_not_crowd_results(semantic_tool):
    out = semantic_tool("query", limit=5, min_similarity=0.0, rerank=False)
    titles = [(r["chapter_title"],) for r in out["results"]]
    assert len(titles) == len(set(titles)), f"duplicate chapters in results: {titles}"
    assert len(out["results"]) == 2  # ch1 (best chunk) + ch2

"""Tests for Cohere reranker with graceful fallback."""

from unittest.mock import MagicMock, patch

import src.utils.reranker as reranker_module
from src.utils.reranker import rerank_results


class TestReranker:
    def setup_method(self):
        """Reset the module-level client cache before each test."""
        reranker_module._client = None

    def _make_results(self, n: int) -> list[dict]:
        """Build dummy search results."""
        return [
            {
                "chapter_id": f"ch_{i}",
                "book_title": f"Book {i}",
                "chapter_title": f"Chapter {i}",
                "chunk_content": f"Content about topic {i}. " * 50,
                "rrf_score": 1.0 / (i + 1),
            }
            for i in range(n)
        ]

    def test_rerank_reorders_results(self):
        """Reranker reorders results based on Cohere scores."""
        results = self._make_results(3)

        mock_response = MagicMock()
        mock_response.results = [
            MagicMock(index=2, relevance_score=0.9),
            MagicMock(index=0, relevance_score=0.5),
            MagicMock(index=1, relevance_score=0.1),
        ]

        with patch("src.utils.reranker.cohere") as mock_cohere:
            mock_cohere.ClientV2.return_value.rerank.return_value = mock_response
            reranked = rerank_results("test query", results, top_n=3)

        assert reranked[0]["chapter_id"] == "ch_2"
        assert reranked[1]["chapter_id"] == "ch_0"
        assert "rerank_score" in reranked[0]

    def test_fallback_on_api_error(self):
        """Returns original results if Cohere API fails."""
        results = self._make_results(3)

        with patch("src.utils.reranker.cohere") as mock_cohere:
            mock_cohere.ClientV2.return_value.rerank.side_effect = Exception("API down")
            reranked = rerank_results("test query", results, top_n=3)

        assert [r["chapter_id"] for r in reranked] == ["ch_0", "ch_1", "ch_2"]
        assert "rerank_score" not in reranked[0]

    def test_empty_results(self):
        """Empty input returns empty output."""
        assert rerank_results("query", [], top_n=5) == []

    def test_top_n_limits_output(self):
        """top_n limits the number of returned results."""
        results = self._make_results(10)

        mock_response = MagicMock()
        mock_response.results = [
            MagicMock(index=i, relevance_score=1.0 - i * 0.1)
            for i in range(5)
        ]

        with patch("src.utils.reranker.cohere") as mock_cohere:
            mock_cohere.ClientV2.return_value.rerank.return_value = mock_response
            reranked = rerank_results("query", results, top_n=5)

        assert len(reranked) == 5

    def test_disabled_returns_original(self):
        """rerank=False passes results through unchanged."""
        results = self._make_results(3)
        reranked = rerank_results("query", results, top_n=3, enabled=False)
        assert reranked == results

    def test_disabled_does_not_truncate(self):
        """enabled=False returns all results, ignoring top_n."""
        results = self._make_results(10)
        reranked = rerank_results("query", results, top_n=3, enabled=False)
        assert len(reranked) == 10

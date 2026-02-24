"""Tests for OpenAI embedding wrapper."""

from unittest.mock import MagicMock, patch

import numpy as np

from src.utils.openai_embeddings import OpenAIEmbeddingGenerator


class TestOpenAIEmbeddingGenerator:
    def test_generate_single(self):
        """Generate embedding for a single text."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]

        with patch("src.utils.openai_embeddings.openai") as mock_openai:
            mock_openai.OpenAI.return_value.embeddings.create.return_value = mock_response
            gen = OpenAIEmbeddingGenerator()
            result = gen.generate("hello world")

        assert isinstance(result, np.ndarray)
        assert result.shape == (1536,)

    def test_generate_batch(self):
        """Generate embeddings for multiple texts."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536),
            MagicMock(embedding=[0.2] * 1536),
        ]

        with patch("src.utils.openai_embeddings.openai") as mock_openai:
            mock_openai.OpenAI.return_value.embeddings.create.return_value = mock_response
            gen = OpenAIEmbeddingGenerator()
            result = gen.generate_batch(["hello", "world"])

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 1536)

    def test_generate_empty_raises(self):
        """Empty text raises ValueError."""
        import pytest

        gen = OpenAIEmbeddingGenerator.__new__(OpenAIEmbeddingGenerator)
        gen._client = MagicMock()

        with pytest.raises(ValueError):
            gen.generate("")

    def test_large_batch_splits(self):
        """Batches larger than max_batch_size are split into sub-batches."""
        embeddings = [[0.1] * 1536] * 5
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=e) for e in embeddings]

        with patch("src.utils.openai_embeddings.openai") as mock_openai:
            client = mock_openai.OpenAI.return_value
            client.embeddings.create.return_value = mock_response

            gen = OpenAIEmbeddingGenerator(max_batch_size=5)
            texts = [f"text {i}" for i in range(10)]
            result = gen.generate_batch(texts)

        assert client.embeddings.create.call_count == 2
        assert result.shape == (10, 1536)

    def test_large_token_batch_splits(self):
        """Batches exceeding MAX_BATCH_TOKENS are split even if item count is under max_batch_size."""
        import tiktoken
        from src.utils.openai_embeddings import MAX_BATCH_TOKENS, MODEL

        enc = tiktoken.encoding_for_model(MODEL)
        # Build a text that is exactly 1000 tokens when re-encoded
        token_ids = list(range(1000))
        long_text = enc.decode(token_ids)
        actual_tokens = len(enc.encode(long_text))
        # Scale to produce > MAX_BATCH_TOKENS total
        count = (MAX_BATCH_TOKENS // actual_tokens) + 10
        texts = [long_text] * count

        # Each API call returns one embedding per input text
        def make_response(n):
            r = MagicMock()
            r.data = [MagicMock(embedding=[0.1] * 1536) for _ in range(n)]
            return r

        with patch("src.utils.openai_embeddings.openai") as mock_openai:
            client = mock_openai.OpenAI.return_value
            client.embeddings.create.side_effect = lambda model, input: make_response(len(input))

            gen = OpenAIEmbeddingGenerator(max_batch_size=2048)  # item limit won't trigger
            result = gen.generate_batch(texts)

        # Should have split into at least 2 API calls
        assert client.embeddings.create.call_count >= 2
        assert result.shape == (count, 1536)

    def test_dimension_property(self):
        """Dimension reports 3072 (text-embedding-3-large)."""
        with patch("src.utils.openai_embeddings.openai") as mock_openai:
            gen = OpenAIEmbeddingGenerator()
        assert gen.dimension == 3072

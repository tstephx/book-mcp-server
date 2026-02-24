"""OpenAI embedding generation for semantic search.

Wraps the OpenAI embeddings API with batching support. Drop-in
replacement for the local EmbeddingGenerator interface.
"""

import logging

import numpy as np
import openai
import tiktoken

logger = logging.getLogger(__name__)

MODEL = "text-embedding-3-large"
DIMENSIONS = 3072
MAX_TOKENS = 8191  # text-embedding-3-large per-text token limit
MAX_BATCH_TOKENS = 250_000  # OpenAI allows 300k; use 250k as safety margin

# Lazy-loaded tokenizer
_encoding = None


def _get_encoding():
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.encoding_for_model(MODEL)
    return _encoding


class OpenAIEmbeddingGenerator:
    """Generate embeddings using OpenAI text-embedding-3-large.

    Matches the interface of the local EmbeddingGenerator:
    generate(text) -> ndarray, generate_batch(texts) -> ndarray.
    """

    def __init__(self, max_batch_size: int = 2048):
        self._client = openai.OpenAI()
        self._max_batch_size = max_batch_size

    @staticmethod
    def _truncate(text: str) -> tuple[str, int]:
        """Truncate text to stay within the model's token limit.

        Returns (truncated_text, token_count) to avoid re-encoding in callers.
        """
        enc = _get_encoding()
        tokens = enc.encode(text)
        if len(tokens) <= MAX_TOKENS:
            return text, len(tokens)
        logger.warning(f"Truncating text from {len(tokens)} to {MAX_TOKENS} tokens")
        return enc.decode(tokens[:MAX_TOKENS]), MAX_TOKENS

    def generate(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")

        truncated, _ = self._truncate(text)
        response = self._client.embeddings.create(
            model=MODEL,
            input=[truncated],
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def generate_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        if not texts:
            raise ValueError("Cannot generate embeddings for empty list")

        all_embeddings: list[np.ndarray] = []
        enc = _get_encoding()

        current_batch: list[str] = []
        current_tokens = 0

        def _flush(batch: list[str]) -> list[np.ndarray]:
            response = self._client.embeddings.create(model=MODEL, input=batch)
            return [np.array(item.embedding, dtype=np.float32) for item in response.data]

        for text in texts:
            truncated, token_count = self._truncate(text)

            needs_token_flush = current_tokens + token_count > MAX_BATCH_TOKENS
            needs_size_flush = len(current_batch) >= self._max_batch_size

            if current_batch and (needs_token_flush or needs_size_flush):
                all_embeddings.extend(_flush(current_batch))
                current_batch = []
                current_tokens = 0

            current_batch.append(truncated)
            current_tokens += token_count

        if current_batch:
            all_embeddings.extend(_flush(current_batch))

        return np.vstack(all_embeddings)

    @property
    def dimension(self) -> int:
        return DIMENSIONS

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
MAX_TOKENS = 8191  # text-embedding-3-large limit

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
    def _truncate(text: str) -> str:
        """Truncate text to stay within the model's token limit."""
        enc = _get_encoding()
        tokens = enc.encode(text)
        if len(tokens) <= MAX_TOKENS:
            return text
        logger.warning(f"Truncating text from {len(tokens)} to {MAX_TOKENS} tokens")
        return enc.decode(tokens[:MAX_TOKENS])

    def generate(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")

        response = self._client.embeddings.create(
            model=MODEL,
            input=[self._truncate(text)],
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def generate_batch(self, texts: list[str], batch_size: int = 0) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        if not texts:
            raise ValueError("Cannot generate embeddings for empty list")

        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(texts), self._max_batch_size):
            batch = [self._truncate(t) for t in texts[i : i + self._max_batch_size]]
            response = self._client.embeddings.create(
                model=MODEL,
                input=batch,
            )
            batch_embeddings = [
                np.array(item.embedding, dtype=np.float32)
                for item in response.data
            ]
            all_embeddings.extend(batch_embeddings)

        return np.vstack(all_embeddings)

    @property
    def dimension(self) -> int:
        return DIMENSIONS

"""Embedding generation utilities for semantic search

This module handles text-to-vector conversion using sentence transformers.
Follows MCP best practices: single responsibility, clean separation.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generate embeddings for semantic search
    
    Uses sentence-transformers for efficient, high-quality embeddings.
    Model is loaded once and reused (singleton pattern).
    """
    
    _instance: Optional['EmbeddingGenerator'] = None
    _model: Optional[SentenceTransformer] = None
    
    def __new__(cls):
        """Singleton pattern - one model instance for efficiency"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize embedding model (lazy loading)"""
        if self._model is None:
            logger.info("Loading embedding model...")
            # Use a lightweight, fast model optimized for semantic search
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully")
    
    def generate(self, text: str) -> np.ndarray:
        """Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            384-dimensional embedding vector
            
        Raises:
            ValueError: If text is empty or invalid
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")
        
        # Truncate very long texts to avoid memory issues
        max_length = 512  # tokens
        if len(text) > max_length * 4:  # rough char estimate
            text = text[:max_length * 4]
            logger.warning(f"Text truncated to {max_length * 4} characters")
        
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding
    
    def generate_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for multiple texts efficiently
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings, shape (len(texts), 384)
            
        Raises:
            ValueError: If texts list is empty
        """
        if not texts:
            raise ValueError("Cannot generate embeddings for empty text list")
        
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True
        )
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        return 384  # all-MiniLM-L6-v2 outputs 384-dim vectors

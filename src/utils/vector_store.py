"""Vector similarity utilities for semantic search

This module handles vector similarity calculations using cosine similarity.
Follows MCP best practices: focused functionality, clear interface.
"""

import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Similarity score between -1 and 1 (1 = identical)
        
    Raises:
        ValueError: If vectors have different dimensions
    """
    if vec1.shape != vec2.shape:
        raise ValueError(f"Vector dimensions don't match: {vec1.shape} vs {vec2.shape}")
    
    # Cosine similarity = dot product / (norm1 * norm2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))

def batch_cosine_similarity(query_vec: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity between query and multiple vectors efficiently
    
    Args:
        query_vec: Query vector (1D array)
        vectors: Matrix of vectors to compare against (2D array)
        
    Returns:
        Array of similarity scores
        
    Raises:
        ValueError: If dimensions don't match
    """
    if len(query_vec.shape) != 1:
        raise ValueError(f"Query vector must be 1D, got shape {query_vec.shape}")
    
    if len(vectors.shape) != 2:
        raise ValueError(f"Vectors must be 2D matrix, got shape {vectors.shape}")
    
    if query_vec.shape[0] != vectors.shape[1]:
        raise ValueError(
            f"Dimension mismatch: query {query_vec.shape[0]} vs vectors {vectors.shape[1]}"
        )
    
    # Normalize query vector
    query_norm = query_vec / np.linalg.norm(query_vec)
    
    # Normalize all vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    vectors_norm = vectors / norms
    
    # Compute all similarities at once
    similarities = np.dot(vectors_norm, query_norm)
    
    return similarities

def find_top_k(
    query_vec: np.ndarray,
    vectors: np.ndarray,
    k: int = 10,
    min_similarity: float = 0.0
) -> List[Tuple[int, float]]:
    """Find top K most similar vectors
    
    Args:
        query_vec: Query vector
        vectors: Matrix of vectors to search
        k: Number of top results to return
        min_similarity: Minimum similarity threshold (0.0 to 1.0)
        
    Returns:
        List of (index, similarity) tuples, sorted by similarity (descending)
        
    Raises:
        ValueError: If k is invalid or dimensions don't match
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    
    if k > vectors.shape[0]:
        k = vectors.shape[0]
        logger.warning(f"k reduced to {k} (total number of vectors)")
    
    # Calculate all similarities
    similarities = batch_cosine_similarity(query_vec, vectors)
    
    # Filter by minimum similarity
    mask = similarities >= min_similarity
    filtered_indices = np.where(mask)[0]
    filtered_similarities = similarities[filtered_indices]
    
    if len(filtered_similarities) == 0:
        return []
    
    # Get top k
    top_k_indices = np.argsort(filtered_similarities)[::-1][:k]
    
    results = [
        (int(filtered_indices[i]), float(filtered_similarities[i]))
        for i in top_k_indices
    ]
    
    return results

"""Semantic search tool for finding relevant book content

Follows MCP best practices:
- Single responsibility (semantic search only)
- Clean separation from keyword search
- Comprehensive error handling
- Centralized schemas (MCP Chapter 6 best practice)
"""

import logging
from typing import Optional
import numpy as np

from ..utils.vector_store import find_top_k
from ..utils.context_managers import embedding_model_context
from ..utils.file_utils import read_chapter_content, get_chapter_excerpt
from ..utils.excerpt_utils import extract_relevant_excerpt
from ..utils.cache import get_cache
from ..utils.embedding_loader import load_chapter_embeddings
from ..schemas.tool_schemas import SemanticSearchInput

logger = logging.getLogger(__name__)

def register_semantic_search_tools(mcp):
    """Register semantic search tool with MCP server
    
    Args:
        mcp: FastMCP server instance
    """
    
    @mcp.tool()
    def semantic_search(
        query: str,
        limit: int = 5,
        min_similarity: float = 0.3
    ) -> dict:
        """Search books using semantic similarity
        
        Uses embeddings to find conceptually similar content, even if
        exact keywords don't match. Great for finding related topics,
        similar concepts, or thematic connections.
        
        Args:
            query: Search query (e.g., "container networking concepts")
            limit: Maximum results to return (1-20, default: 5)
            min_similarity: Minimum similarity score 0.0-1.0 (default: 0.3)
            
        Returns:
            Dictionary with results, each containing:
            - book_title: Title of the book
            - chapter_title: Chapter title
            - chapter_number: Chapter number
            - similarity: Similarity score (0.0-1.0, higher is better)
            - excerpt: Relevant text excerpt
            
        Examples:
            semantic_search("docker networking", limit=3)
            semantic_search("leadership principles", min_similarity=0.5)
        """
        try:
            # Validate inputs using centralized schema (MCP best practice)
            try:
                validated = SemanticSearchInput(
                    query=query,
                    limit=limit,
                    min_similarity=min_similarity
                )
            except Exception as e:
                logger.warning(f"Validation error: {e}")
                return {
                    "error": f"Invalid input: {str(e)}",
                    "results": []
                }
            
            logger.info(f"Semantic search: '{validated.query}' (limit={validated.limit}, min_sim={validated.min_similarity})")

            # Keep embedding model loaded for both query and excerpt extraction
            with embedding_model_context() as generator:
                # Generate query embedding
                query_embedding = generator.generate(validated.query)

                # Load embeddings from cache or database
                embeddings_matrix, chapter_metadata = load_chapter_embeddings()

                if embeddings_matrix is None:
                    return {
                        "message": "No embeddings found. Run generate_embeddings.py first.",
                        "results": []
                    }

                # Find top K most similar
                top_results = find_top_k(
                    query_embedding,
                    embeddings_matrix,
                    k=validated.limit,
                    min_similarity=validated.min_similarity
                )

                # Build response with query-relevant excerpts (model still loaded)
                results = []
                for idx, similarity in top_results:
                    metadata = chapter_metadata[idx]

                    # Extract query-relevant excerpt using semantic similarity
                    try:
                        content = read_chapter_content(metadata['file_path'])
                        excerpt = extract_relevant_excerpt(
                            query_embedding, content, generator, max_chars=500
                        )
                    except Exception as e:
                        logger.warning(f"Excerpt extraction failed: {e}")
                        excerpt = get_chapter_excerpt(metadata['file_path'], max_chars=300)

                    results.append({
                        'book_title': metadata['book_title'],
                        'chapter_title': metadata['chapter_title'],
                        'chapter_number': metadata['chapter_number'],
                        'similarity': round(similarity, 3),
                        'excerpt': excerpt
                    })

            logger.info(f"Found {len(results)} results for '{validated.query}'")
            
            return {
                "query": validated.query,
                "results": results,
                "total_found": len(results)
            }
            
        except Exception as e:
            logger.error(f"Semantic search error: {e}", exc_info=True)
            return {
                "error": str(e),
                "results": []
            }
    
    @mcp.resource("book://semantic-context/{query}")
    async def semantic_context(query: str) -> str:
        """Provide semantic context for RAG pattern
        
        This resource automatically injects relevant context into the LLM's
        prompt when the query URI is referenced. This enables retrieval-
        augmented generation without explicit tool calls.
        
        Args:
            query: Search query for finding relevant context
            
        Returns:
            Formatted text containing top 3 most relevant passages
            
        Usage:
            - LLM can automatically use this for context injection
            - Combines semantic search with prompt augmentation
            - Follows MCP RAG best practices
        """
        try:
            # Use semantic search tool to find relevant content
            search_results = semantic_search(query, limit=3, min_similarity=0.4)
            
            if 'error' in search_results:
                return f"Error retrieving context: {search_results['error']}"
            
            results = search_results.get('results', [])
            
            if not results:
                return f"No relevant context found for: {query}"
            
            # Format results as context
            context_parts = [f"Relevant context for '{query}':\n"]
            
            for i, result in enumerate(results, 1):
                context_parts.append(
                    f"\n[{i}] From '{result['book_title']}' - "
                    f"Chapter {result['chapter_number']}: {result['chapter_title']}\n"
                    f"(Similarity: {result['similarity']:.2f})\n"
                    f"{result['excerpt']}\n"
                )
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Semantic context error: {e}", exc_info=True)
            return f"Error: {str(e)}"

"""Context managers for resource lifecycle management

Following MCP best practice from Chapter 6: "Context managers allow you to 
allocate and release resources precisely when you want to."

These ensure proper initialization and cleanup of expensive resources.
"""

from contextlib import contextmanager
from typing import Generator, Optional
import logging

logger = logging.getLogger(__name__)

@contextmanager
def embedding_model_context() -> Generator:
    """Manage embedding model lifecycle
    
    Ensures proper initialization and cleanup of the embedding model.
    The model is expensive (~90MB) so proper lifecycle management is important.
    
    Usage:
        with embedding_model_context() as generator:
            embedding = generator.generate("some text")
    
    Yields:
        EmbeddingGenerator: Initialized embedding generator
        
    Example:
        with embedding_model_context() as gen:
            query_emb = gen.generate("docker networking")
            results = search_similar(query_emb)
    """
    from .embeddings import EmbeddingGenerator
    
    logger.debug("Initializing embedding model context...")
    generator = None
    
    try:
        generator = EmbeddingGenerator()
        logger.debug("Embedding model ready")
        yield generator
    except Exception as e:
        logger.error(f"Embedding model context error: {e}")
        raise
    finally:
        # Cleanup: In future, could release model from memory if needed
        logger.debug("Embedding model context closed")
        # Note: Singleton pattern means model stays in memory
        # This is intentional for performance

@contextmanager
def database_transaction() -> Generator:
    """Manage database transaction lifecycle
    
    Ensures proper commit/rollback of database operations.
    Use this for write operations that need transactional guarantees.
    
    Usage:
        with database_transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT ...")
            # Automatically commits on success
            # Automatically rolls back on error
    
    Yields:
        sqlite3.Connection: Database connection with transaction
        
    Raises:
        Exception: Any database error (after rollback)
        
    Example:
        with database_transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE chapters SET embedding = ? WHERE id = ?", 
                          (embedding_bytes, chapter_id))
            # Auto-commit if successful
    """
    from ..database import get_db_connection
    
    logger.debug("Starting database transaction...")
    
    with get_db_connection() as conn:
        try:
            yield conn
            conn.commit()
            logger.debug("Transaction committed")
        except Exception as e:
            conn.rollback()
            logger.error(f"Transaction rolled back: {e}")
            raise

@contextmanager
def batch_processing_context(batch_size: int = 32, 
                             description: str = "Processing") -> Generator:
    """Manage batch processing operations
    
    Provides progress tracking and proper cleanup for batch operations.
    Useful for processing large numbers of items (embeddings, etc.)
    
    Args:
        batch_size: Number of items per batch
        description: Description for logging
        
    Yields:
        dict: Context with batch_size and progress tracking
        
    Example:
        with batch_processing_context(batch_size=32, description="Embeddings") as ctx:
            for i, batch in enumerate(batches):
                process_batch(batch)
                ctx['processed'] += len(batch)
    """
    import time
    
    start_time = time.time()
    context = {
        'batch_size': batch_size,
        'processed': 0,
        'start_time': start_time
    }
    
    logger.info(f"Starting {description} (batch_size={batch_size})")
    
    try:
        yield context
    finally:
        elapsed = time.time() - start_time
        rate = context['processed'] / elapsed if elapsed > 0 else 0
        logger.info(
            f"{description} complete: {context['processed']} items "
            f"in {elapsed:.1f}s ({rate:.1f} items/s)"
        )

@contextmanager
def error_context(operation: str, 
                  default_value: Optional[any] = None,
                  raise_error: bool = True) -> Generator:
    """Wrap operations with consistent error handling
    
    Provides consistent error logging and optional graceful degradation.
    
    Args:
        operation: Description of operation for logging
        default_value: Value to return on error (if raise_error=False)
        raise_error: Whether to raise exception or return default
        
    Yields:
        dict: Context for storing operation results
        
    Example:
        with error_context("semantic search", default_value=[]) as ctx:
            ctx['result'] = perform_search()
        
        results = ctx.get('result', [])
    """
    context = {}
    
    try:
        yield context
    except Exception as e:
        logger.error(f"Error in {operation}: {e}", exc_info=True)
        
        if raise_error:
            raise
        else:
            logger.warning(f"Returning default value for {operation}")
            context['result'] = default_value

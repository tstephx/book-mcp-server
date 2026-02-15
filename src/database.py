"""
Database connection management with context managers
Follows MCP best practices for resource management
"""

import logging
import sqlite3
from contextlib import contextmanager
from typing import Generator
from pathlib import Path

from .config import Config

logger = logging.getLogger(__name__)

class DatabaseError(Exception):
    """Custom exception for database errors"""
    pass

@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager for database connections
    
    Usage:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM books")
    
    Yields:
        sqlite3.Connection: Database connection with Row factory
    
    Raises:
        DatabaseError: If connection fails
    """
    conn = None
    try:
        conn = sqlite3.connect(str(Config.DB_PATH), timeout=10)
        conn.row_factory = sqlite3.Row
        yield conn
    except sqlite3.Error as e:
        raise DatabaseError(f"Database connection error: {str(e)}")
    finally:
        if conn:
            conn.close()

def execute_query(query: str, params: tuple = ()) -> list[sqlite3.Row]:
    """
    Execute a SELECT query and return results
    
    Args:
        query: SQL query string
        params: Query parameters (optional)
    
    Returns:
        List of Row objects
    
    Raises:
        DatabaseError: If query fails
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
    except sqlite3.Error as e:
        raise DatabaseError(f"Query execution failed: {str(e)}")

def execute_single(query: str, params: tuple = ()) -> sqlite3.Row | None:
    """
    Execute a SELECT query and return single result
    
    Args:
        query: SQL query string
        params: Query parameters (optional)
    
    Returns:
        Single Row object or None
    
    Raises:
        DatabaseError: If query fails
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchone()
    except sqlite3.Error as e:
        raise DatabaseError(f"Query execution failed: {str(e)}")

def check_database_health() -> dict:
    """
    Check database health and return stats
    
    Returns:
        Dictionary with database statistics
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Count books
            cursor.execute("SELECT COUNT(*) as count FROM books")
            book_count = cursor.fetchone()['count']
            
            # Count chapters
            cursor.execute("SELECT COUNT(*) as count FROM chapters")
            chapter_count = cursor.fetchone()['count']
            
            # Total word count
            cursor.execute("SELECT SUM(word_count) as total FROM books")
            total_words = cursor.fetchone()['total'] or 0
            
            return {
                "status": "healthy",
                "books": book_count,
                "chapters": chapter_count,
                "total_words": total_words
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


def _add_column_if_missing(cursor, table: str, column: str, col_type: str) -> bool:
    """Add a column to a table if it doesn't exist. Returns True if added."""
    try:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        return True
    except sqlite3.OperationalError:
        return False


def ensure_library_schema() -> None:
    """Ensure all library-side tables and columns exist.

    Idempotent — safe to call on every server startup.
    Uses CREATE TABLE IF NOT EXISTS for tables and
    ALTER TABLE ADD COLUMN (with try/except) for columns.

    Manages:
    - chapters_fts (FTS5 virtual table, created empty)
    - chapter_summaries + embedding columns
    - reading_progress
    - bookmarks
    - Tracking columns on chapters table (content_hash, file_mtime, embedding_updated_at)
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        created = []

        # --- FTS5 virtual table (empty — rebuild_fts_index() populates it) ---
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chapters_fts'"
        )
        if not cursor.fetchone():
            cursor.execute("""
                CREATE VIRTUAL TABLE chapters_fts USING fts5(
                    chapter_id,
                    title,
                    content,
                    tokenize='porter unicode61'
                )
            """)
            created.append("chapters_fts")

        # --- Chapter summaries ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chapter_summaries (
                chapter_id TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                summary_type TEXT DEFAULT 'extractive',
                word_count INTEGER,
                generated_at TEXT,
                FOREIGN KEY (chapter_id) REFERENCES chapters(id) ON DELETE CASCADE
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_summaries_type ON chapter_summaries(summary_type)"
        )

        # Embedding columns on chapter_summaries
        if _add_column_if_missing(cursor, "chapter_summaries", "embedding", "BLOB"):
            created.append("chapter_summaries.embedding")
        if _add_column_if_missing(cursor, "chapter_summaries", "embedding_model", "TEXT"):
            created.append("chapter_summaries.embedding_model")

        # --- Reading progress ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reading_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id TEXT NOT NULL,
                chapter_number INTEGER NOT NULL,
                status TEXT DEFAULT 'unread',
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                notes TEXT,
                FOREIGN KEY (book_id) REFERENCES books(id) ON DELETE CASCADE,
                UNIQUE(book_id, chapter_number)
            )
        """)

        # --- Bookmarks ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bookmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id TEXT NOT NULL,
                chapter_number INTEGER NOT NULL,
                position INTEGER DEFAULT 0,
                title TEXT,
                note TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (book_id) REFERENCES books(id) ON DELETE CASCADE
            )
        """)

        # --- Chunks for sub-chapter retrieval ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                chapter_id TEXT NOT NULL,
                book_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                word_count INTEGER NOT NULL,
                embedding BLOB,
                embedding_model TEXT,
                content_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chapter_id) REFERENCES chapters(id)
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_chapter ON chunks(chapter_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_book ON chunks(book_id)"
        )

        # --- Tracking columns on chapters (owned by book-ingestion, but we add columns) ---
        if _add_column_if_missing(cursor, "chapters", "content_hash", "TEXT"):
            created.append("chapters.content_hash")
        if _add_column_if_missing(cursor, "chapters", "file_mtime", "REAL"):
            created.append("chapters.file_mtime")
        if _add_column_if_missing(cursor, "chapters", "embedding_updated_at", "TEXT"):
            created.append("chapters.embedding_updated_at")

        conn.commit()

    if created:
        logger.info(f"Library schema updated: {', '.join(created)}")
    else:
        logger.debug("Library schema up to date")

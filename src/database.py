"""
Database connection management with context managers
Follows MCP best practices for resource management
"""

import sqlite3
from contextlib import contextmanager
from typing import Generator
from pathlib import Path

from .config import Config

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
        conn = sqlite3.connect(str(Config.DB_PATH))
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

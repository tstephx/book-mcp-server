"""Shared database connection context manager for agentic_pipeline.

Provides guaranteed connection cleanup via context manager pattern,
replacing the per-class _connect() / try/finally boilerplate.
"""

import sqlite3
from contextlib import contextmanager
from typing import Generator, Optional

from .config import get_db_path


@contextmanager
def get_pipeline_db(db_path: Optional[str] = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for pipeline database connections.

    Args:
        db_path: Optional database path override. Uses get_db_path() if None.

    Yields:
        sqlite3.Connection with row_factory set to sqlite3.Row

    Example:
        with get_pipeline_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM processing_pipelines")
    """
    if db_path is None:
        db_path = str(get_db_path())

    conn = None
    try:
        conn = sqlite3.connect(str(db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        if conn:
            conn.close()

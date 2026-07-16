"""Read the library-wide data version used for cache coherence.

`library_meta.data_version` is bumped inside bulk-mutation transactions
(e.g. `agentic-pipeline rechunk --swap`). In-memory caches store the
version they loaded under and self-invalidate when it changes.
"""

import sqlite3
from typing import Optional

from ..database import DatabaseError, get_db_connection


def get_data_version() -> Optional[int]:
    """Current data_version, or None on a pre-migration DB (no self-check).

    Note: `get_db_connection()`'s context manager re-raises any
    `sqlite3.Error` encountered inside its `with`-body as `DatabaseError`
    (see `src/database.py`), so a missing `library_meta` table surfaces as
    `DatabaseError`, not the original `sqlite3.OperationalError`. Catch both.
    """
    try:
        with get_db_connection() as conn:
            row = conn.execute("SELECT data_version FROM library_meta WHERE id = 1").fetchone()
            return row["data_version"] if row else None
    except (sqlite3.OperationalError, DatabaseError):
        return None

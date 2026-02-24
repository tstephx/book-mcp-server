# Incremental Embedding Updates Design

**Status: Implemented**

## Overview

Track content changes using file mtime and content hashes to detect modifications. Only regenerate embeddings for new or changed chapters, avoiding expensive full reindexing.

## Problem

Current embedding generation (`generate_embeddings.py`) has two modes:
- No `--force`: Only generates for chapters without embeddings
- With `--force`: Regenerates ALL embeddings

Neither detects content changes. If a chapter file is modified, its embedding becomes stale but won't be regenerated unless `--force` is used.

## Solution

Add change tracking columns to the chapters table:
1. `file_mtime` - Fast detection via filesystem metadata
2. `content_hash` - Definitive verification when mtime changes
3. `embedding_updated_at` - Audit trail

Detection flow:
```
For each chapter:
  1. Get current file mtime (O(1) syscall)
  2. If mtime > stored_mtime OR stored_mtime is NULL:
     a. Read file content
     b. Compute SHA-256 hash
     c. If hash != stored_hash → needs update
  3. If mtime unchanged → skip (embedding still valid)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Incremental Embedding Updates              │
├─────────────────────────────────────────────────────────────┤
│  Schema Additions (chapters table):                         │
│    + content_hash TEXT                                      │
│    + file_mtime REAL                                        │
│    + embedding_updated_at TEXT                              │
│                                                             │
│  Detection (mtime as fast path, hash as verification):      │
│    file.mtime > stored_mtime? → read & hash → compare       │
│                                                             │
│  Update Flow:                                               │
│    find_chapters_needing_update()                           │
│         ↓                                                   │
│    batch process (50 at a time)                             │
│         ↓                                                   │
│    update: embedding + hash + mtime + timestamp             │
│         ↓                                                   │
│    invalidate embeddings cache                              │
└─────────────────────────────────────────────────────────────┘
```

## API Design

### New Module: `src/utils/embedding_sync.py`

```python
def compute_content_hash(content: str) -> str:
    """SHA-256 hash of chapter content"""

def find_chapters_needing_update(force: bool = False) -> list[dict]:
    """Find chapters where file changed or no embedding exists

    Args:
        force: If True, return all chapters (ignore tracking)

    Returns:
        List of: {id, file_path, title, chapter_number, reason}
        Reason: 'new' | 'modified' | 'no_hash' | 'forced'
    """

def update_embeddings_incremental(
    batch_size: int = 50,
    force: bool = False,
    progress_callback: Callable = None
) -> dict:
    """Update only changed/new embeddings

    Returns: {status, updated, skipped, errors, duration_seconds}
    """
```

### New MCP Tool

```python
@mcp.tool()
def refresh_embeddings(force: bool = False) -> dict:
    """Refresh embeddings for changed chapters

    Detects content changes using file modification times and content
    hashes. Only regenerates embeddings for new or modified chapters.

    Args:
        force: If True, regenerate ALL embeddings (ignore change detection)

    Returns:
        Dictionary with:
        - status: 'updated' | 'no_updates_needed' | 'error'
        - updated: Number of embeddings regenerated
        - skipped: Number of unchanged chapters
        - errors: Number of failed chapters
        - duration_seconds: Time taken

    Examples:
        refresh_embeddings()  # Only update changed chapters
        refresh_embeddings(force=True)  # Regenerate everything
    """
```

## Schema Migration

```sql
-- Add tracking columns
ALTER TABLE chapters ADD COLUMN content_hash TEXT;
ALTER TABLE chapters ADD COLUMN file_mtime REAL;
ALTER TABLE chapters ADD COLUMN embedding_updated_at TEXT;

-- Create index for efficient queries
CREATE INDEX IF NOT EXISTS idx_chapters_needs_update
ON chapters(file_mtime, content_hash);
```

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `migrations/add_embedding_tracking.py` | Create | Schema migration |
| `src/utils/embedding_sync.py` | Create | Core sync logic |
| `src/server.py` | Modify | Register refresh_embeddings tool |

## Usage

### Via MCP Tool (Claude Desktop)

```
refresh_embeddings()  # Check for changes and update
refresh_embeddings(force=True)  # Full regeneration
```

### Via Command Line

```bash
# Run migration first (one-time)
python migrations/add_embedding_tracking.py

# Then use via MCP or import directly
python -c "from src.utils.embedding_sync import update_embeddings_incremental; print(update_embeddings_incremental())"
```

## Performance Expectations

| Scenario | Time |
|----------|------|
| Check 268 chapters (no changes) | ~100ms (mtime checks only) |
| Update 1 changed chapter | ~2s (model load + encode) |
| Update 10 changed chapters | ~3s (batch processing) |
| Full regeneration (268 chapters) | ~30s |

## Cache Invalidation

After any embeddings are updated:
1. Call `cache.invalidate_embeddings()`
2. Next search will reload fresh embeddings from database

# Caching Design for Book MCP Server

**Status: Implemented**

## Overview

Two-tier in-memory caching system to improve performance by avoiding repeated file reads and embedding deserialization.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    LibraryCache                         │
├─────────────────────────────────────────────────────────┤
│  Tier 1: Embeddings Matrix (in-memory, load once)      │
│  - All chapter embeddings as numpy matrix               │
│  - Associated metadata (book_title, chapter_title, etc) │
│  - Invalidate only when DB changes                      │
│                                                         │
│  Tier 2: Chapter Content (in-memory, TTL-based)        │
│  - file_path → (content, mtime, expires_at)            │
│  - Check file mtime for invalidation                    │
│  - Default TTL: 1 hour                                  │
└─────────────────────────────────────────────────────────┘
```

## Design Decisions

### Why in-memory only (no SQLite persistence)?

- MCP tools are called with varying parameters - cache hit rate for search results is low
- Real performance wins come from data layer caching (files, embeddings)
- Simpler implementation, no serialization complexity
- Cache warms quickly on first use

### Why not cache search results?

- `semantic_search("docker", limit=5)` vs `semantic_search("docker", limit=10)` are different cache keys
- Embedding matrix cache already eliminates the expensive part (DB query + deserialization)
- Chapter content cache handles excerpt generation

### TTL Strategy

| Cache Tier | TTL | Invalidation |
|------------|-----|--------------|
| Embeddings | Infinite | Manual (when books added/removed) |
| Chapters | 1 hour | File mtime check |

## API Design

```python
from src.utils.cache import cache

# Chapter content
content = cache.get_chapter(file_path)  # Returns None if miss/expired
cache.set_chapter(file_path, content, mtime)

# Embeddings matrix (tuple of matrix + metadata list)
result = cache.get_embeddings()  # Returns (np.ndarray, list[dict]) or None
cache.set_embeddings(matrix, metadata)
cache.invalidate_embeddings()

# Monitoring
stats = cache.stats()  # {"chapters_cached": 12, "embeddings_loaded": True, ...}
```

## Integration Points

### 1. `src/utils/cache.py` (new)

LibraryCache class with:
- Thread-safe dictionary storage
- TTL expiration logic
- File mtime validation
- Memory usage estimation

### 2. `src/utils/file_utils.py` (modify)

```python
def read_chapter_content(file_path: str | Path) -> str:
    # Check cache first
    cached = cache.get_chapter(file_path)
    if cached is not None:
        return cached

    # Read file (existing logic)
    content = ...

    # Store in cache
    cache.set_chapter(file_path, content, mtime)
    return content
```

### 3. `src/tools/semantic_search_tool.py` (modify)

```python
def semantic_search(...):
    # Check embeddings cache
    cached = cache.get_embeddings()
    if cached:
        embeddings_matrix, chapter_metadata = cached
    else:
        # Load from DB (existing logic)
        ...
        cache.set_embeddings(embeddings_matrix, chapter_metadata)
```

### 4. `src/tools/discovery_tools.py` (modify)

Same pattern as semantic_search for `find_related_content` and `get_topic_coverage`.

### 5. `src/config.py` (modify)

```python
# Cache settings
ENABLE_CACHING: bool = os.getenv("ENABLE_CACHING", "false").lower() == "true"
CACHE_CHAPTER_TTL: int = int(os.getenv("CACHE_CHAPTER_TTL", "3600"))  # 1 hour
```

### 6. New MCP tool: `get_cache_stats`

```python
@mcp.tool()
def get_cache_stats() -> dict:
    """Get cache statistics for monitoring"""
    return cache.stats()
```

## File Changes Summary

| File | Action |
|------|--------|
| `src/utils/cache.py` | Create |
| `src/utils/file_utils.py` | Modify |
| `src/tools/semantic_search_tool.py` | Modify |
| `src/tools/discovery_tools.py` | Modify |
| `src/config.py` | Modify |
| `src/server.py` | Modify (register cache stats tool) |

## Performance Results

Tested with caching enabled (`ENABLE_CACHING=true`):

| Operation | Without Cache | With Cache | Speedup |
|-----------|--------------|------------|---------|
| Chapter read | 0.19ms | 0.03ms | **6x** |
| Embeddings load (268 chapters) | 17ms | 0.01ms | **1921x** |

## Usage

Enable caching by setting the environment variable:

```bash
export ENABLE_CACHING=true
```

Or in your MCP client configuration.

### MCP Tools

- `get_cache_stats()` - View cache statistics
- `clear_cache(cache_type="all")` - Clear caches ("chapters", "embeddings", or "all")

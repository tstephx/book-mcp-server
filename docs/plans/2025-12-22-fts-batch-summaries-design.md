# Full-Text Search, Batch Operations, and Chapter Summaries Design

**Status: Implemented**

## Overview

Three complementary features to enhance library search and content access:

1. **Full-Text Search (FTS5)**: Fast keyword/phrase search with BM25 ranking
2. **Batch Operations**: Cross-library search and aggregate statistics
3. **Chapter Summaries**: Extractive summarization for quick content overview

## 1. Full-Text Search (FTS5)

### Problem

Semantic search finds conceptually similar content but misses exact term matches. Users need both capabilities for effective research.

### Solution

SQLite FTS5 virtual table with Porter stemming and Unicode support.

```
┌─────────────────────────────────────────────────────────────┐
│                    Full-Text Search                         │
├─────────────────────────────────────────────────────────────┤
│  Virtual Table: chapters_fts                                │
│    - chapter_id (TEXT)                                      │
│    - title (TEXT)                                           │
│    - content (TEXT, full chapter content)                   │
│    - tokenize='porter unicode61'                            │
│                                                             │
│  Capabilities:                                              │
│    - Phrase search: "async await"                           │
│    - Boolean: python AND async                              │
│    - Prefix: python*                                        │
│    - Negation: python NOT java                              │
│    - BM25 ranking for relevance                             │
│    - Snippet extraction with highlighting                   │
└─────────────────────────────────────────────────────────────┘
```

### Files

| File | Description |
|------|-------------|
| `migrations/add_fts_and_summaries.py` | Creates FTS5 table and indexes content |
| `src/utils/fts_search.py` | Search implementation with query escaping |
| `src/server.py` | `text_search` tool registration |

### MCP Tool

```python
text_search(
    query: str,           # FTS5 query
    limit: int = 10,      # Max results (1-50)
    book_id: str = None   # Optional book filter
) -> dict
```

## 2. Batch Operations

### Problem

Searching one book at a time is tedious. Users need to search across the entire library and get aggregate statistics.

### Solution

Batch semantic search with per-book result limits and comprehensive library statistics.

```
┌─────────────────────────────────────────────────────────────┐
│                    Batch Operations                         │
├─────────────────────────────────────────────────────────────┤
│  Batch Semantic Search:                                     │
│    - Searches ALL books with single query                   │
│    - Limits results per book (default: 5)                   │
│    - Groups results by book for easy navigation             │
│    - Uses cached embeddings for speed                       │
│                                                             │
│  Library Statistics:                                        │
│    - Total books, chapters, words                           │
│    - Embedding coverage percentage                          │
│    - FTS index status                                       │
│    - Per-book breakdown                                     │
└─────────────────────────────────────────────────────────────┘
```

### Files

| File | Description |
|------|-------------|
| `src/utils/batch_ops.py` | Batch search and statistics |
| `src/server.py` | `search_all_books`, `get_library_stats` tools |

### MCP Tools

```python
search_all_books(
    query: str,                  # Semantic search query
    max_per_book: int = 5,       # Results per book
    min_similarity: float = 0.3  # Similarity threshold
) -> dict

get_library_stats() -> dict
```

## 3. Chapter Summaries

### Problem

Reading full chapters takes time. Users need quick overviews to decide what to read in depth.

### Solution

Extractive summarization using position-based and content-based heuristics.

```
┌─────────────────────────────────────────────────────────────┐
│                    Chapter Summaries                        │
├─────────────────────────────────────────────────────────────┤
│  Extraction Strategy:                                       │
│    1. Take first paragraph as intro                         │
│    2. Score remaining sentences by:                         │
│       - Position (earlier = higher)                         │
│       - Length (prefer medium 50-200 chars)                 │
│       - Content markers (definitions, conclusions)          │
│    3. Select top-scoring sentences                          │
│    4. Re-order by original position                         │
│                                                             │
│  Storage: chapter_summaries table                           │
│    - Cached on first generation                             │
│    - Force regeneration available                           │
└─────────────────────────────────────────────────────────────┘
```

### Schema

```sql
CREATE TABLE chapter_summaries (
    chapter_id TEXT PRIMARY KEY,
    summary TEXT NOT NULL,
    summary_type TEXT DEFAULT 'extractive',
    word_count INTEGER,
    generated_at TEXT,
    FOREIGN KEY (chapter_id) REFERENCES chapters(id)
);
```

### Files

| File | Description |
|------|-------------|
| `migrations/add_fts_and_summaries.py` | Creates summaries table |
| `src/utils/summaries.py` | Extraction logic and caching |
| `src/server.py` | `get_summary`, `summarize_book` tools |

### MCP Tools

```python
get_summary(
    chapter_id: str,     # Chapter to summarize
    force: bool = False  # Regenerate even if cached
) -> dict

summarize_book(
    book_id: str,        # Book to summarize
    force: bool = False  # Regenerate all
) -> dict
```

## Usage Examples

### Full-Text Search

```
# Find exact phrases
text_search("dependency injection")

# Boolean operators
text_search("docker AND networking")

# Prefix matching
text_search("async*")

# Filter to specific book
text_search("chapter", book_id="clean-architecture")
```

### Batch Operations

```
# Search across all books
search_all_books("error handling patterns")

# Limit results per book
search_all_books("testing", max_per_book=2)

# Get library overview
get_library_stats()
```

### Chapter Summaries

```
# Get or generate summary
get_summary("clean-arch-ch5")

# Force regeneration
get_summary("clean-arch-ch5", force=True)

# Summarize entire book
summarize_book("clean-architecture")
```

## Performance

| Operation | Time |
|-----------|------|
| FTS search (10 results) | ~10ms |
| Batch search (17 books) | ~50ms |
| Library statistics | ~5ms |
| Summary generation | ~100ms |
| Summary retrieval (cached) | <1ms |

## Migration

Run once to set up:

```bash
python migrations/add_fts_and_summaries.py --books-dir /path/to/books
```

This creates:
- `chapters_fts` virtual table with all content indexed
- `chapter_summaries` table (empty, populated on demand)

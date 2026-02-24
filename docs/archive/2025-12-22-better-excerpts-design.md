# Better Excerpts Design (Query-Relevant Passages)

**Status: Implemented**

## Overview

Replace static excerpts (first N characters) with dynamically extracted query-relevant passages using semantic similarity.

## Problem

Current excerpt generation takes the first 300 characters of chapter content, which often includes headers, metadata, or introductory text that may not be relevant to the user's query.

## Solution

Use the same embedding model already loaded for query processing to:
1. Split chapter content into paragraphs
2. Generate embeddings for each paragraph (batch)
3. Find the paragraph most similar to the query embedding
4. Return that paragraph as the excerpt

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    semantic_search()                         │
├─────────────────────────────────────────────────────────────┤
│  with embedding_model_context() as generator:               │
│    query_embedding = generator.generate(query)              │
│    top_results = find_top_k(...)                            │
│                                                             │
│    for result in top_results:                               │
│      content = read_chapter_content(file_path)              │
│      excerpt = extract_relevant_excerpt(                    │
│          query_embedding,   # Already computed              │
│          content,           # Chapter text                  │
│          generator,         # Model still loaded            │
│          max_chars=500                                      │
│      )                                                      │
└─────────────────────────────────────────────────────────────┘
```

## Design Decisions

### Why semantic search only (not all excerpt generation)?

- `get_chapter_excerpt()` is used in many places without query context
- Semantic excerpts only make sense when there's a query to match against
- Avoids performance overhead in non-search contexts

### Why batch with query embedding?

- Model is already loaded for query embedding
- Reusing model avoids re-initialization overhead
- Batch encoding ~100 paragraphs takes ~50-100ms

### Why pass query_embedding instead of query string?

- Already computed at the start of semantic_search
- Avoids redundant embedding generation
- Cleaner API - function does one thing (similarity matching)

## API Design

### New Module: `src/utils/excerpt_utils.py`

```python
def split_paragraphs(content: str, min_length: int = 50) -> list[str]:
    """Split content into paragraphs on double newlines or headers

    Args:
        content: Full chapter text
        min_length: Minimum paragraph length (shorter merged with next)

    Returns:
        List of paragraph strings
    """

def extract_relevant_excerpt(
    query_embedding: np.ndarray,
    content: str,
    generator: EmbeddingGenerator,
    max_chars: int = 500
) -> str:
    """Extract the most query-relevant paragraph from content

    Args:
        query_embedding: Pre-computed embedding for the search query
        content: Full chapter text
        generator: Embedding generator (model already loaded)
        max_chars: Maximum excerpt length

    Returns:
        Most relevant paragraph, truncated if needed
    """
```

## Integration Changes

### `src/tools/semantic_search_tool.py`

Move results loop inside `embedding_model_context()` block:

```python
# BEFORE
with embedding_model_context() as generator:
    query_embedding = generator.generate(validated.query)

# ... load embeddings ...
# ... find top_k ...

results = []
for idx, similarity in top_results:
    metadata = chapter_metadata[idx]
    excerpt = get_chapter_excerpt(metadata['file_path'], max_chars=300)
    results.append({...})

# AFTER
with embedding_model_context() as generator:
    query_embedding = generator.generate(validated.query)

    # ... load embeddings (same) ...
    # ... find top_k (same) ...

    # Build results WITH model still loaded
    results = []
    for idx, similarity in top_results:
        metadata = chapter_metadata[idx]
        content = read_chapter_content(metadata['file_path'])
        excerpt = extract_relevant_excerpt(
            query_embedding, content, generator, max_chars=500
        )
        results.append({...})
```

## Performance Expectations

| Operation | Time |
|-----------|------|
| Split paragraphs | <1ms |
| Batch encode ~20 paragraphs | ~20-50ms |
| Cosine similarity | <1ms |
| **Total per result** | ~25-55ms |
| **5 results** | ~125-275ms |

With embeddings cache, total semantic_search time should remain under 500ms.

## File Changes Summary

| File | Action |
|------|--------|
| `src/utils/excerpt_utils.py` | Create |
| `src/tools/semantic_search_tool.py` | Modify |

## Fallback Behavior

If excerpt extraction fails for any reason:
- Fall back to `get_chapter_excerpt()` (first N chars)
- Log warning but don't fail the search

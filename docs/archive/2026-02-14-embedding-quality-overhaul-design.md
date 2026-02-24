<!-- project: book-mcp-server -->

# Embedding Quality Overhaul — Design

**Date:** 2026-02-14
**Goal:** Improve search quality across the board by upgrading the embedding model, adding sub-chapter chunking, and introducing a reranking step.

## Architecture

Three layers, each independently valuable:

```
Search Query
  │
  ▼
Layer 3: Reranker (Cohere rerank-v3.5)
  Takes top-N candidates → precise reordering on actual text
  │
  ▼
Layer 2: Retrieval (existing RRF + MMR)
  FTS5 keyword + semantic vector → fused results
  │
  ▼
Layer 1: Embeddings (new)
  OpenAI text-embedding-3-small on sub-chapter chunks (~500 words)
  Stored in `chunks` table
```

The unit of retrieval shifts from **chapter** (~2,000-10,000 words) to **chunk** (~500 words). Chapters remain the organizational unit for reading/navigation — chunks are only for search.

**What stays the same:**
- FTS5 keyword search (searches full chapter text)
- Hybrid search RRF fusion logic
- MMR diversity re-ranking
- All non-search MCP tools (reading, learning, progress tracking)

## 1. Chunking Strategy

### New `chunks` table

```sql
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,          -- "{chapter_id}:{chunk_index}"
    chapter_id TEXT NOT NULL,
    book_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    word_count INTEGER NOT NULL,
    embedding BLOB,
    embedding_model TEXT,
    content_hash TEXT,
    created_at TEXT,
    FOREIGN KEY (chapter_id) REFERENCES chapters(id)
);
```

### Chunking algorithm — semantic paragraph grouping

1. Split chapter into paragraphs (by `\n\n`)
2. Greedily group consecutive paragraphs until hitting ~500 words
3. Overlap: include the last paragraph of the previous chunk as the first paragraph of the next (avoids losing context at boundaries)
4. Short chapters (<600 words) become a single chunk

### Why 500 words

- OpenAI `text-embedding-3-small` has an 8,191 token limit (~6,000 words) — plenty of headroom
- 500 words is the sweet spot in RAG literature — specific enough for precise retrieval, long enough for coherent meaning
- With ~3,400 chapters averaging ~1,500 words, expect roughly **~10,000-15,000 chunks**

## 2. Embedding Model Upgrade

**Switch from `all-MiniLM-L6-v2` (local) to `text-embedding-3-small` (OpenAI API).**

| | Current | New |
|---|---|---|
| Model | all-MiniLM-L6-v2 | text-embedding-3-small |
| Dimensions | 384 | 1536 |
| Max tokens | 512 | 8,191 |
| Quality (MTEB) | ~49 | ~62 |
| Latency | ~5ms local | ~100ms API call |
| Cost | Free | $0.02/1M tokens |

### Implementation

- New `src/utils/openai_embeddings.py` — thin wrapper around `openai.embeddings.create()`, handles batching (OpenAI supports up to 2,048 inputs per call)
- `EmbeddingGenerator` interface stays the same: `generate(text) -> ndarray`, `generate_batch(texts) -> ndarray`
- Query embeddings also switch to OpenAI (must use same model for queries and documents)
- Local `sentence-transformers` dependency becomes optional

### Storage impact

- Embeddings go from 384-float (1.5KB) to 1536-float (6KB) per chunk
- ~15,000 chunks = ~90MB of embedding data in the DB (fine for SQLite)
- Cache: 15,000 x 1536 = ~92MB in-memory numpy matrix (reasonable for local MCP server)

### Cost estimate

~15,000 chunks x ~700 tokens avg = ~10.5M tokens = **~$0.21** for the full library.

## 3. Reranking

**Add Cohere `rerank-v3.5` as a post-retrieval reranker.**

The retrieval layer (RRF fusion) casts a wide net — ~30-50 candidates. A cross-encoder reranker does a full query-document attention pass for much more precise relevance scores.

### Search flow

```
Query → FTS5 + Semantic → RRF fusion (top 30) → Reranker (top 30 → reordered) → Return top N
```

- Reranking runs on **chunk content** (actual text, not just embeddings)
- Only `hybrid_search` and `semantic_search` get reranking
- `rerank: bool = True` parameter on search tools to allow disabling for speed
- Estimated cost: ~$1/month with heavy use

### Fallback

If Cohere API is unavailable, skip reranking and return RRF-ordered results (graceful degradation).

## 4. Migration Plan

### Step 1: Schema migration
- Add `chunks` table via `agentic_pipeline/db/migrations.py`

### Step 2: Chunk generation (offline CLI)
- Read each chapter's content from disk
- Run paragraph-grouping chunker
- Insert rows into `chunks` table (no embeddings yet)

### Step 3: Embedding generation (offline CLI)
- Batch all chunks through OpenAI `text-embedding-3-small`
- Store embeddings in `chunks.embedding`
- Also regenerate `chapters.embedding` with OpenAI for backward compat

### Step 4: Update search tools
- `semantic_search` and `hybrid_search` query `chunks` instead of `chapters`
- Results map chunks back to parent chapter for display
- Excerpts come directly from chunk content (no more runtime excerpt extraction)

### Step 5: Update cache
- `LibraryCache` gets a chunk embeddings tier (replaces chapter embeddings tier)
- `load_chapter_embeddings()` → `load_chunk_embeddings()`

### Pipeline integration for new books
- After `ProcessingAdapter.process_book()` writes chapters, a new step chunks them and generates embeddings via OpenAI
- Replaces the current local `generate_embeddings()` in the EMBEDDING state

### CLI commands

```bash
agentic-pipeline chunk-library              # Generate chunks for all books
agentic-pipeline embed-library --model openai  # Generate embeddings
```

## 5. Error Handling

| Failure | Behavior |
|---|---|
| OpenAI API down during embedding | Retry 3x with backoff, fail to NEEDS_RETRY |
| OpenAI API down during search query | Skip semantic, return FTS-only results |
| Cohere API down during rerank | Skip reranking, return RRF-ordered results |
| Chunk has no embedding | Excluded from semantic search, still found via FTS |
| Chapter too short to chunk | Single chunk = full chapter content |

## 6. Testing

- **Chunker unit tests** — paragraph grouping, overlap, edge cases
- **OpenAI embedding wrapper tests** — mock API, batching, error handling
- **Reranker tests** — mock Cohere API, fallback behavior
- **Search integration tests** — end-to-end with test DB, chunk→chapter mapping
- **Migration tests** — `chunks` table creation, chunker output verification

## Not In Scope (YAGNI)

- No multi-model support / A-B testing framework
- No embedding versioning / rollback (just re-embed if needed)
- No async/parallel embedding calls (sequential batches are fast enough)
- No chunk-level FTS index (FTS5 stays at chapter level)

## Files Affected

### New files
- `src/utils/openai_embeddings.py` — OpenAI embedding wrapper
- `src/utils/chunker.py` — paragraph-grouping chunker
- `src/utils/reranker.py` — Cohere reranker with fallback
- `src/utils/chunk_loader.py` — load chunk embeddings from DB/cache

### Modified files
- `agentic_pipeline/db/migrations.py` — `chunks` table schema
- `agentic_pipeline/adapters/processing_adapter.py` — chunking + OpenAI embeddings in EMBEDDING state
- `src/tools/semantic_search_tool.py` — query chunks, add rerank param
- `src/tools/hybrid_search_tool.py` — query chunks, add rerank param
- `src/utils/cache.py` — chunk embeddings cache tier
- `src/utils/embedding_loader.py` — load chunks instead of chapters
- `src/utils/embeddings.py` — delegate to OpenAI wrapper
- `agentic_pipeline/cli.py` — `chunk-library` and `embed-library` commands

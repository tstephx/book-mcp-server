# MCP Tools Reference

## Agentic Pipeline MCP Server (`agentic_mcp_server.py`)

Registered tools exposed to Claude Desktop via FastMCP. Implementations in `agentic_pipeline/mcp_server.py`.

### Approval

| Tool | Signature | Description |
|------|-----------|-------------|
| `pending_books` | `(sort_by="priority") → dict` | Get all books awaiting review. Returns `{pending_count, books[]}` |
| `approve` | `(pipeline_id, actor="mcp:claude") → dict` | Approve a book. Returns immediately; embedding runs in background. Note: underlying `approve_book()` accepts `adjustments` dict but MCP wrapper does not expose it |
| `reject` | `(pipeline_id, reason, retry=False, actor="mcp:claude") → dict` | Reject a book. `retry=True` queues for reprocessing |
| `rollback` | `(pipeline_id, reason, actor="mcp:claude") → dict` | Remove an approved book from the library |

### Processing

| Tool | Signature | Description |
|------|-----------|-------------|
| `process` | `(book_path: str) → dict` | Process a book file (`.epub`/`.pdf` only). Returns `{pipeline_id, state, book_type, confidence}` |
| `status` | `(pipeline_id: str) → dict` | Get pipeline state, book_type, confidence, approved_by |
| `reingest` | `(book_id: str) → dict` | Reprocess existing book through full pipeline (requires original source file). Returns `{new_pipeline_id, old_pipeline_id}` |

### Health & Monitoring

| Tool | Signature | Description |
|------|-----------|-------------|
| `health` | `() → dict` | Active/queued/stuck counts, alerts, untracked_books count |
| `stuck` | `() → list` | Pipelines stuck longer than expected in their current state |

### Batch Operations

| Tool | Signature | Description |
|------|-----------|-------------|
| `batch_approve` | `(min_confidence?, book_type?, max_count=50, execute=False) → dict` | Approve matching books. `execute=False` previews only |
| `batch_reject` | `(reason, book_type?, max_confidence?, max_count=50, execute=False) → dict` | Reject matching books. `execute=False` previews only |

### Audit

| Tool | Signature | Description |
|------|-----------|-------------|
| `audit` | `(book_id?, actor?, action?, last_days=7, limit=100) → list` | Query approval audit trail |

### Autonomy

| Tool | Signature | Description |
|------|-----------|-------------|
| `autonomy_status` | `() → dict` | Current mode, escape hatch status, 30-day metrics |
| `set_autonomy` | `(mode: str) → dict` | Change mode: `supervised`/`partial`/`confident` |
| `escape_hatch` | `(reason: str) → dict` | Emergency: immediately revert to supervised mode |
| `autonomy_readiness` | `() → dict` | Check if ready to advance to next autonomy level |

### Library Management

| Tool | Signature | Description |
|------|-----------|-------------|
| `backfill` | `(dry_run=True) → dict` | Register legacy books in audit trail. `dry_run=True` previews |
| `validate_library` | `() → dict` | Check all books for quality issues (missing chapters, embeddings, low word count) |

---

## Book Library MCP Server (`server.py` → `src/tools/`)

Read-only tools for Claude Desktop. Implementations in `src/tools/*.py`.

### Search

| Tool | File | Description |
|------|------|-------------|
| `text_search` | `server.py` | Keyword/FTS search across all books |
| `search_titles` | `search_tools.py` | Search book titles by query |
| `semantic_search` | `semantic_search_tools.py` | Vector similarity search (requires OpenAI) |
| `hybrid_search` | `hybrid_search_tools.py` | RRF combining FTS + semantic (best quality) |
| `search_all_books` | `server.py` | Cross-library search |

### Discovery

| Tool | File | Description |
|------|------|-------------|
| `get_topic_coverage` | `discovery_tools.py` | Which books cover a topic and how well |
| `find_related_content` | `discovery_tools.py` | Find content related to a query across chapters |
| `extract_code_examples` | `discovery_tools.py` | Find code examples by language/topic |

### Reading

| Tool | File | Description |
|------|------|-------------|
| `list_books` | `book_tools.py` | All books with metadata |
| `get_book_info` | `book_tools.py` | Book details: title, author, chapters, tags |
| `get_table_of_contents` | `book_tools.py` | Chapter list for a book |
| `get_chapter` | `chapter_tools.py` | Full chapter content (auto-splits large chapters) |
| `get_section` | `chapter_tools.py` | Specific section within a chapter |
| `list_sections` | `chapter_tools.py` | Section index for a chapter |

### Learning

| Tool | File | Description |
|------|------|-------------|
| `teach_concept` | `learning_tools.py` | Explain a concept using library sources |
| `generate_learning_path` | `project_learning_tools.py` | Phased learning plan for a project goal |
| `create_study_guide` | `export_tools.py` | Study guide for a chapter |

### Planning

| Tool | File | Description |
|------|------|-------------|
| `list_project_templates` | `project_learning_tools.py` | Available project type templates |
| `generate_implementation_plan` | `project_planning_tools.py` | Phased implementation plan for a goal |
| `list_implementation_templates` | `project_planning_tools.py` | Available implementation templates |
| `get_phase_prompts` | `project_planning_tools.py` | AI prompts for a specific project phase |
| `generate_brd` | `project_planning_tools.py` | Business Requirements Document |
| `generate_wireframe_brief` | `project_planning_tools.py` | Wireframe spec for UI/UX |
| `list_architecture_templates` | `project_planning_tools.py` | Available architecture patterns |
| `analyze_project` | `project_planning_tools.py` | Analyze a project against best practices |

### Progress Tracking

| Tool | File | Description |
|------|------|-------------|
| `mark_as_reading` | `reading_tools.py` | Mark book as currently reading |
| `mark_as_read` | `reading_tools.py` | Mark book as completed |
| `get_reading_progress` | `reading_tools.py` | Reading status across all books |
| `add_bookmark` | `reading_tools.py` | Save a bookmark with note |
| `get_bookmarks` | `reading_tools.py` | List bookmarks (optionally filtered by book) |
| `remove_bookmark` | `reading_tools.py` | Delete a bookmark |

### Export

| Tool | File | Description |
|------|------|-------------|
| `export_chapter_to_markdown` | `export_tools.py` | Export chapter as clean markdown |

### Library Management

| Tool | File | Description |
|------|------|-------------|
| `library_status` | `server.py` | Overall library health check |
| `get_library_statistics` | `analytics_tools.py` | Books, chapters, word counts, coverage |
| `get_library_stats` | `server.py` | Lightweight stats: totals, embedding coverage, per-book breakdown |
| `find_duplicate_coverage` | `analytics_tools.py` | Topics covered by multiple books |
| `get_author_insights` | `analytics_tools.py` | Stats for a specific author |
| `audit_chapter_quality` | `audit_tools.py` | Chapter quality metrics |
| `get_cache_stats` | `server.py` | Cache hit/miss rates, memory usage, cached counts |
| `clear_cache` | `server.py` | Clear caches (`cache_type`: `chapters`/`embeddings`/`summary_embeddings`/`all`) |
| `get_summary` | `server.py` | Summary for a specific chapter (`chapter_id`, `force`) |
| `summarize_book` | `server.py` | Generate/fetch book summary |
| `refresh_embeddings` | `server.py` | Regenerate chapter embeddings |
| `generate_summary_embeddings` | `server.py` | Generate summary-level embeddings |

---

## Book ID Resolution (`src/utils/validators.py`)

All tools accepting `book_id` also accept partial title slugs via `resolve_book_id()`:
1. UUID exact match (fast path)
2. Fuzzy LIKE title match
3. Returns "did-you-mean" error if ambiguous

Do not call the old `validate_book_id()` directly.

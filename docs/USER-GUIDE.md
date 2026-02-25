# Book MCP Server User Guide

A comprehensive guide to using your personal book library with Claude.

## Quick Start

### Connect to Claude Desktop

Add to your Claude Desktop config:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "book-library": {
      "command": "/path/to/book-mcp-server/.venv/bin/python",
      "args": ["/path/to/book-mcp-server/server.py"],
      "env": {
        "BOOK_DB_PATH": "/path/to/library.db",
        "BOOKS_DIR": "/path/to/books",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

Restart Claude Desktop. Claude now has access to your entire book library.

---

## Capabilities Overview

### Search & Discovery Tools

| Tool | Description | Example Query |
|------|-------------|---------------|
| `semantic_search` | AI-powered concept matching using embeddings | "error handling patterns" |
| `hybrid_search` | RRF fusion of keyword + semantic search | "dependency injection" |
| `text_search` | Fast keyword/phrase search (FTS5) | `"dependency injection"` |
| `search_titles` | Simple keyword search in titles/authors | "python docker" |
| `search_all_books` | Search entire library grouped by book | "async programming" |
| `get_topic_coverage` | Find which books cover a topic | "machine learning" |
| `extract_code_examples` | Extract code samples from a book | "API design" |
| `find_related_content` | Find thematically similar chapters | "microservices" |

### Reading & Navigation Tools

| Tool | Description |
|------|-------------|
| `list_books` | List all books with metadata |
| `get_book_info` | Detailed book info including chapters |
| `get_table_of_contents` | Formatted chapter listing |
| `get_chapter` | Read full chapter content |
| `get_section` | Read section of large chapters |
| `list_sections` | List sections if chapter is split |

### Learning & Study Tools

| Tool | Description |
|------|-------------|
| `teach_concept` | Explain technical concepts with business analogies |
| `generate_learning_path` | Create personalized learning paths from your books |
| `create_study_guide` | Generate study guide from a chapter (includes flashcards and key concepts) |

### Project Planning Tools

| Tool | Description |
|------|-------------|
| `generate_learning_path` | Learning plan for VPS, web apps, ML projects |
| `list_project_templates` | View available project templates and example goals |
| `generate_implementation_plan` | PM-ready plans with phases, risks, gates |
| `generate_brd` | Business requirements document from a project goal |
| `generate_wireframe_brief` | Architecture brief / wireframe outline |
| `analyze_project` | Project complexity and recommendation analysis |
| `list_implementation_templates` | View available implementation templates |
| `get_phase_prompts` | Per-phase prompts for a given project goal |
| `list_architecture_templates` | View available architecture templates |

### Reading Progress Tools

| Tool | Description |
|------|-------------|
| `mark_as_reading` | Mark a chapter as currently reading |
| `mark_as_read` | Mark a chapter as completed |
| `get_reading_progress` | See read/unread status for a book |
| `add_bookmark` | Save bookmarks with notes |
| `get_bookmarks` | List all bookmarks in a book |
| `remove_bookmark` | Remove a bookmark by ID |
| `export_chapter_to_markdown` | Export chapter as clean markdown |

### Library Management Tools

| Tool | Description |
|------|-------------|
| `library_status` | Full library overview with pipeline summary |
| `get_library_statistics` | Comprehensive stats (books, chapters, words, topics) |
| `get_library_stats` | Aggregate statistics snapshot |
| `find_duplicate_coverage` | Identify topics covered redundantly across books |
| `get_author_insights` | Per-author analysis across your library |
| `audit_chapter_quality` | Audit chapter quality by severity (all/warning/bad) |
| `get_summary` | Get extractive summary for a chapter |
| `summarize_book` | Generate summaries for an entire book |
| `refresh_embeddings` | Update embeddings for changed chapters |
| `generate_summary_embeddings` | Generate embeddings for chapter summaries |
| `get_cache_stats` | Monitor caching performance |
| `clear_cache` | Clear cached data |

---

## Search Syntax

### Full-Text Search (FTS5)

```
text_search("docker")                    # Simple keyword
text_search('"dependency injection"')    # Exact phrase (quotes)
text_search("python AND async")          # Boolean AND
text_search("docker NOT kubernetes")     # Boolean NOT
text_search("docker OR podman")          # Boolean OR
text_search("micro*")                    # Prefix matching
```

### Semantic Search

Semantic search finds conceptually related content even without exact keyword matches:

```
semantic_search("how to handle errors gracefully")
semantic_search("container orchestration", limit=5, min_similarity=0.5)
```

---

## Example Workflows

### Learn a New Technology

**Ask Claude:**
> "Search my books for Kubernetes content and create a learning path for me"

**What happens:**
1. Semantic search finds all Kubernetes-related chapters
2. Learning path tool creates a phased plan
3. References specific chapters from YOUR books

### Build a Project

**Ask Claude:**
> "I want to set up a VPS. Create a learning path and implementation plan."

**What happens:**
1. `generate_learning_path("Build a VPS on Hetzner")` creates a phased learning plan
2. `generate_implementation_plan("Build a VPS on Hetzner")` creates a PM-ready plan
3. Pulls relevant content from your DevOps/Linux books

### Compare Perspectives

**Ask Claude:**
> "How do my different books explain microservices architecture? Compare them."

**What happens:**
1. `find_related_content` finds thematically related chapters across books
2. `get_topic_coverage("microservices")` shows which books cover it
3. Synthesizes different perspectives

### Study a Chapter

**Ask Claude:**
> "Create flashcards and a study guide for the Docker networking chapter"

**What happens:**
1. Finds the relevant chapter via search
2. `create_study_guide()` creates structured summary with flashcards and key concepts

### Research a Topic

**Ask Claude:**
> "What do my books say about error handling in Python?"

**What happens:**
1. `semantic_search("error handling Python")` finds relevant sections
2. `get_topic_coverage("error handling")` shows which books cover it
3. Synthesizes findings across your library

---

## MCP Resources

Resources provide automatic context injection:

| Resource URI | Description |
|--------------|-------------|
| `library://catalog` | Complete library overview |
| `book://catalog` | Simple book list |
| `book://{book_id}/metadata` | Rich book metadata |
| `book://semantic-context/{query}` | Top 3 semantically relevant passages |
| `collection://list` | List all reading collections |
| `collection://{name}` | Books in curated collection |

> **Note:** Named collections (`collection://{name}`) are not yet implemented. The resource URIs are defined but collection membership is not configured.

---

## CLI Commands

### Setup & Info

```bash
# Show version
agentic-pipeline version

# Initialize the database (run once)
agentic-pipeline init
```

### Pipeline Management

```bash
# Process a single book
agentic-pipeline process /path/to/book.epub

# Show status of a specific pipeline
agentic-pipeline status <pipeline_id>

# Retry books in NEEDS_RETRY state
agentic-pipeline retry
agentic-pipeline retry --max-attempts 5

# Reprocess a book through the full pipeline (archives old record, creates new)
agentic-pipeline reingest <book_id>

# Check pipeline health
agentic-pipeline health
agentic-pipeline health --json

# Check stuck pipelines
agentic-pipeline stuck
agentic-pipeline stuck --recover
```

### Approval

```bash
# See books pending approval
agentic-pipeline pending

# Approve a book (triggers embedding inline)
agentic-pipeline approve <pipeline_id>

# Reject a book
agentic-pipeline reject <pipeline_id> --reason "Too short"
agentic-pipeline reject <pipeline_id> --reason "Poor quality" --retry
```

### Batch Operations

```bash
# Dry-run batch approval
agentic-pipeline batch-approve --min-confidence 0.9

# Execute batch approval (filtered by type and confidence)
agentic-pipeline batch-approve --min-confidence 0.9 --book-type technical_tutorial --execute

# Batch reject low-quality books
agentic-pipeline batch-reject --max-confidence 0.5 --reason "Low quality" --execute
```

### Autonomy Management

```bash
# Check current autonomy mode and 30-day metrics
agentic-pipeline autonomy status

# Enable partial autonomy (auto-approve high confidence books)
agentic-pipeline autonomy enable partial

# Enable confident autonomy (calibrated per-type thresholds)
agentic-pipeline autonomy enable confident

# Disable autonomy (revert to supervised without escape hatch)
agentic-pipeline autonomy disable

# Resume autonomy after escape hatch
agentic-pipeline autonomy resume

# Emergency revert to supervised mode (sets escape hatch flag)
agentic-pipeline escape-hatch "Unusual errors detected"
```

### Spot-Check Reviews

```bash
# List books selected for spot-check review
agentic-pipeline spot-check --list
```

### Library Maintenance

```bash
# Show combined library + pipeline status dashboard
agentic-pipeline library-status
agentic-pipeline library-status --json

# Register legacy books that have no pipeline record
agentic-pipeline backfill --dry-run
agentic-pipeline backfill --execute

# Check library books for quality issues (missing chapters, embeddings)
agentic-pipeline validate
agentic-pipeline validate --json

# Audit extraction quality (flags books with poor chapter extraction)
agentic-pipeline audit-quality
agentic-pipeline audit-quality --json

# Generate chunks for all unchunked chapters
agentic-pipeline chunk-library --dry-run
agentic-pipeline chunk-library

# Generate OpenAI embeddings for all unembedded chunks
agentic-pipeline embed-library --dry-run
agentic-pipeline embed-library --batch-size 50

# Re-queue books that fail quality checks (deletes and reprocesses)
agentic-pipeline reprocess --flagged
agentic-pipeline reprocess --flagged --execute
```

### Inspection & Audit

```bash
# View audit trail
agentic-pipeline audit --last 50
agentic-pipeline audit --actor human:cli --action approve
agentic-pipeline audit --book-id <id>

# List available processing strategies
agentic-pipeline strategies

# Test-classify book text (pass text or a file path)
agentic-pipeline classify --text "Chapter 1: Introduction to Docker..."
agentic-pipeline classify --text /path/to/sample.txt --provider anthropic
```

### Monitoring

```bash
# View audit trail
agentic-pipeline audit --last 50
agentic-pipeline audit --actor human:cli --action approve

# JSON output for monitoring systems
agentic-pipeline health --json
```

---

## Database Schema

### Primary Tables

**books**
- `id` - Book identifier
- `title`, `author` - Metadata
- `word_count` - Total words
- `processing_status` - pending/complete/error

**chapters**
- `id`, `book_id`, `chapter_number`
- `title`, `word_count`
- `embedding` - Vector for semantic search
- `content_hash` - For incremental updates

**reading_progress**
- `book_id`, `chapter_number`
- `status` - unread/reading/read
- `notes` - Personal annotations

**bookmarks**
- `book_id`, `chapter_number`
- `position`, `title`, `note`

**chapters_fts** (Virtual)
- Full-text search index using FTS5
- Porter stemming + unicode support

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BOOK_DB_PATH` | `../book-ingestion-python/data/library.db` | Database path |
| `BOOKS_DIR` | `../book-ingestion-python/data/books` | Books directory |
| `MAX_SEARCH_RESULTS` | `10` | Default search limit |
| `MAX_CHAPTER_SIZE` | `100000` | Max chapter size (bytes) |
| `ENABLE_CACHING` | `false` | Enable chapter caching |
| `CACHE_CHAPTER_TTL` | `3600` | Cache TTL (seconds) |
| `DEBUG` | `false` | Enable debug logging |

---

## Tips for Maximum Value

### 1. Use Semantic Search First
Semantic search finds conceptually related content even without exact keywords. Start broad, then narrow down.

### 2. Create Learning Paths
Don't just read randomly - use `generate_learning_path()` to create structured plans based on YOUR books.

### 3. Cross-Reference Books
Use `get_topic_coverage()` and `find_related_content()` to get multiple perspectives on complex topics.

### 4. Track Progress
Use `mark_as_read()` and `add_bookmark()` to track your learning journey.

### 5. Export for Reference
Use `export_chapter_to_markdown()` to create personal reference documents from key chapters.

### 6. Ask Claude Naturally
You don't need to know tool names. Just ask:
- "What do my books say about X?"
- "Create a study plan for learning Y"
- "Compare how different books explain Z"

---

## Data Flow

```
Ebooks Folder
    ↓
Agentic Pipeline
├─ Classifier → Determine book type
├─ Strategy Selector → Choose processing config
├─ Processor → Extract chapters
├─ Validator → Check quality
├─ Approval → Human review or auto-approve
└─ Embeddings → Generate search vectors
    ↓
Book Library Database
├─ Books (metadata)
├─ Chapters (content + embeddings)
├─ FTS Index (keyword search)
└─ Reading Progress
    ↓
MCP Server
├─ Query tools
├─ Reading tools
├─ Learning tools
└─ Export tools
    ↓
Claude Desktop
```

---

## Troubleshooting

### Books Not Showing Up
```bash
# Check processing status
agentic-pipeline health

# Retry failed books
agentic-pipeline retry
```

### Search Not Finding Content

Ask Claude: "refresh embeddings" — this calls the `refresh_embeddings` MCP tool directly.

### Claude Not Connecting
1. Check Claude Desktop config JSON syntax
2. Verify paths are absolute
3. Restart Claude Desktop completely
4. Check server runs: `python server.py`

---

## Quick Reference Card

| Task | Ask Claude |
|------|-----------|
| Find content | "Search my books for [topic]" |
| Learn something | "Create a learning path for [goal]" |
| Compare books | "Compare how my books explain [topic]" |
| Study chapter | "Create a study guide for [book] chapter [N]" |
| Plan project | "Create an implementation plan for [project type]" |
| Track reading | "Mark chapter [N] of [book] as read" |
| Export content | "Export [book] chapter [N] as markdown" |
| Library stats | "Show my library statistics" |

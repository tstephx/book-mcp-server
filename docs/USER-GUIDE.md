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
      "command": "python",
      "args": ["/path/to/book-mcp-server/server.py"],
      "env": {
        "BOOK_DB_PATH": "/path/to/book-ingestion-python/data/library.db"
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
| `text_search` | Fast keyword/phrase search (FTS5) | `"dependency injection"` |
| `search_books` | Simple keyword search in titles/authors | "python docker" |
| `search_all_books` | Search entire library grouped by book | "async programming" |
| `topic_coverage` | Find which books cover a topic | "machine learning" |
| `cross_book_comparison` | Compare how books approach a topic | "API design" |
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
| `learning_path` | Create personalized learning paths from your books |
| `create_study_guide` | Generate study guide from a chapter |
| `generate_flashcards` | Create flashcard questions from content |
| `extract_key_concepts` | Pull out key definitions and concepts |

### Project Planning Tools

| Tool | Description |
|------|-------------|
| `generate_project_learning_path` | Learning plan for VPS, web apps, ML projects |
| `create_implementation_plan` | PM-ready plans with phases, risks, gates |

### Reading Progress Tools

| Tool | Description |
|------|-------------|
| `mark_as_read` | Track which chapters you've completed |
| `get_reading_progress` | See read/unread status for a book |
| `add_bookmark` | Save bookmarks with notes |
| `list_bookmarks` | List all bookmarks in a book |
| `export_chapter` | Export chapter as clean markdown |
| `export_book` | Export entire book as markdown |

### Library Management Tools

| Tool | Description |
|------|-------------|
| `get_library_statistics` | Comprehensive stats (books, chapters, words, topics) |
| `book_analysis` | Per-book analysis with reading time estimates |
| `refresh_embeddings` | Update embeddings for changed chapters |
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
1. `generate_project_learning_path("vps")` creates 5-8 phase learning plan
2. `create_implementation_plan("vps", "personal")` creates PM-ready plan
3. Pulls relevant content from your DevOps/Linux books

### Compare Perspectives

**Ask Claude:**
> "How do my different books explain microservices architecture? Compare them."

**What happens:**
1. `cross_book_comparison("microservices")` analyzes multiple books
2. Creates side-by-side comparison of approaches
3. Synthesizes different perspectives

### Study a Chapter

**Ask Claude:**
> "Create flashcards and a study guide for the Docker networking chapter"

**What happens:**
1. Finds the relevant chapter via search
2. `generate_flashcards()` creates study questions
3. `create_study_guide()` creates structured summary

### Research a Topic

**Ask Claude:**
> "What do my books say about error handling in Python?"

**What happens:**
1. `semantic_search("error handling Python")` finds relevant sections
2. `topic_coverage("error handling")` shows which books cover it
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

### Available Collections

- `python-essentials` - Core Python books
- `devops-stack` - DevOps and infrastructure
- `ai-ml-track` - AI/ML learning path
- `data-engineering` - Data pipeline books
- `architecture` - Software architecture
- `web-development` - Web dev resources

---

## CLI Commands

### Pipeline Management

```bash
# Check pipeline health
agentic-pipeline health

# See pending approvals
agentic-pipeline pending

# Process a new book
agentic-pipeline process /path/to/book.epub

# Retry failed books
agentic-pipeline retry

# Check stuck pipelines
agentic-pipeline stuck
agentic-pipeline stuck --recover
```

### Batch Operations

```bash
# Dry-run batch approval (high confidence books)
agentic-pipeline batch-approve --min-confidence 0.9

# Execute batch approval
agentic-pipeline batch-approve --min-confidence 0.9 --execute

# Batch reject low-quality
agentic-pipeline batch-reject --max-confidence 0.5 -r "Low quality" --execute
```

### Autonomy Management

```bash
# Check current autonomy mode
agentic-pipeline autonomy status

# Enable partial autonomy (auto-approve high confidence)
agentic-pipeline autonomy enable partial

# Enable confident autonomy (auto-approve + skip validation)
agentic-pipeline autonomy enable confident

# Emergency revert to supervised mode
agentic-pipeline autonomy escape-hatch
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
Don't just read randomly - use `learning_path()` to create structured plans based on YOUR books.

### 3. Cross-Reference Books
Use `cross_book_comparison()` to get multiple perspectives on complex topics.

### 4. Track Progress
Use `mark_as_read()` and `add_bookmark()` to track your learning journey.

### 5. Export for Reference
Use `export_chapter()` to create personal reference documents from key chapters.

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
```bash
# Refresh embeddings
python -c "from src.tools.maintenance import refresh_embeddings; refresh_embeddings(force=True)"
```

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

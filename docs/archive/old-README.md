# 📚 Book Library MCP Server (Production-Ready)

**Production-ready** FastMCP server for accessing your processed book library through Claude Desktop.

---

## 🚀 Quick Start (3 Steps!)

### Step 1: Install

```bash
cd /Users/taylorstephens/_Projects/book-mcp-server
./setup.sh
```

### Step 2: Activate & Test

```bash
# IMPORTANT: Activate virtual environment
source venv/bin/activate

# Run tests
python test_server.py
```

Expected output:
```
✅ All tests passed!
```

### Step 3: Configure Claude Desktop

**📝 EDIT THIS FILE:**
```bash
open -a TextEdit ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

**If file doesn't exist, create it:**
```bash
mkdir -p ~/Library/Application\ Support/Claude/
touch ~/Library/Application\ Support/Claude/claude_desktop_config.json
open -a TextEdit ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

---

## ⚙️ Claude Desktop Configuration

**COPY THIS INTO THE CONFIG FILE:**

```json
{
  "mcpServers": {
    "book-library": {
      "command": "/Users/taylorstephens/_Projects/book-mcp-server/venv/bin/python",
      "args": [
        "-m",
        "src.server"
      ],
      "cwd": "/Users/taylorstephens/_Projects/book-mcp-server"
    }
  }
}
```

**If you already have other MCP servers, add `book-library` to existing config:**

```json
{
  "mcpServers": {
    "your-existing-server": {
      ...existing config...
    },
    "book-library": {
      "command": "/Users/taylorstephens/_Projects/book-mcp-server/venv/bin/python",
      "args": ["-m", "src.server"],
      "cwd": "/Users/taylorstephens/_Projects/book-mcp-server"
    }
  }
}
```

**Then restart Claude Desktop:**
1. **Quit completely** (Cmd+Q, not just close window)
2. Wait 5 seconds
3. Reopen Claude Desktop
4. Look for 🔌 MCP indicator in the interface

---

## ✅ Verify It's Working

In Claude Desktop, try asking:

> **"What books do I have?"**

You should get a list of your 7 books!

---

## 🎯 What You Can Do

### Available Tools

| Tool | Example Query |
|------|---------------|
| `list_books` | "What books do I have?" |
| `get_book_info` | "Tell me about the Docker book" |
| `get_chapter` | "Read chapter 1 of Ubuntu Linux Bible" |
| `search_books` | "Search for systemd" |
| `get_table_of_contents` | "Show me the TOC for Node.js book" |

### Example Queries

**Browse Library:**
- "What books do I have?"
- "Show me all my books"
- "How many chapters do I have?"

**Get Information:**
- "Tell me about the Docker book"
- "What's in the systemd book?"
- "Show me the table of contents for Node.js"

**Read Chapters:**
- "Read chapter 1 of Docker Basics"
- "Show me chapter 5 of Learn Docker in a Month of Lunches"
- "Get the introduction to Ubuntu Linux Bible"

**Search Content:**
- "Search for container networking"
- "Find chapters about systemd services"
- "What chapters mention Docker Compose?"

**Ask Questions:**
- "What does my Docker book say about images?"
- "Explain systemd units based on my books"
- "How do I configure Docker networks?"

---

## 📊 Your Library

- **7 books** (620,000 words)
- **84 chapters**
- Topics: Docker, Linux, Node.js, systemd, networking

**Books:**
1. Ubuntu Linux Bible (229K words)
2. Learn Docker in a Month of Lunches (121K words)
3. Node.js for Beginners (86K words)
4. systemd for Linux SysAdmins (81K words)
5. Docker Basics (41K words)
6. Deep Dive Into Linux Networking (32K words)
7. Linux Coding & Programming Manual (30K words)

---

## 🐛 Troubleshooting

### "No module named 'mcp'" error?

```bash
# Activate virtual environment
source venv/bin/activate

# Should see (venv) in prompt
```

### MCP server not showing in Claude Desktop?

**Check 1:** Config file location correct?
```bash
ls -la ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

**Check 2:** Paths are absolute (not relative)?
```bash
# Verify these exist:
ls -la /Users/taylorstephens/_Projects/book-mcp-server/venv/bin/python
ls -la /Users/taylorstephens/_Projects/book-mcp-server/src/server.py
```

**Check 3:** Restarted Claude Desktop completely?
- Use Cmd+Q (not just close window)
- Wait 5 seconds  
- Reopen

**Check 4:** Look at Claude Desktop logs
```bash
# Open Console.app
open /Applications/Utilities/Console.app

# Filter for: Claude
# Look for Python or MCP errors
```

### Tests failing?

**Database not found:**
```bash
# Check database exists
ls -la /Users/taylorstephens/_Projects/book-ingestion-python/data/library.db
```

**Books directory not found:**
```bash
# Check books directory exists
ls -la /Users/taylorstephens/_Projects/book-ingestion-python/data/books/
```

---

## 🏗️ Architecture (For Developers)

This server follows **all MCP best practices**:

### ✨ Best Practices Implemented

1. **Modular Architecture** ✅
   - Separate modules for tools, database, config
   - Clean separation of concerns

2. **Error Handling** ✅
   - Try/catch in every function
   - Custom exception types
   - Graceful error messages

3. **Input Validation** ✅
   - UUID format validation
   - Business logic checks
   - Range validation

4. **Configuration Management** ✅
   - Environment-based config
   - Validation on startup
   - Easy customization

5. **Connection Management** ✅
   - Context managers for database
   - Automatic cleanup
   - No connection leaks

6. **Logging** ✅
   - Comprehensive logging
   - stderr output (MCP standard)
   - Debug mode support

### 📁 Project Structure

```
book-mcp-server/
├── src/
│   ├── server.py              # Main entry point
│   ├── config.py              # Environment config
│   ├── database.py            # Connection management
│   ├── tools/
│   │   ├── book_tools.py     # Book operations
│   │   ├── chapter_tools.py  # Chapter operations
│   │   └── search_tools.py   # Search operations
│   └── utils/
│       ├── logging.py        # Logging setup
│       └── validators.py     # Input validation
├── test_server.py            # Test suite
├── requirements.txt          # Dependencies
└── setup.sh                  # Setup script
```

---

## ⚙️ Configuration Options

You can customize the server with environment variables:

```bash
# Server
export MCP_SERVER_NAME="book-library"
export MCP_SERVER_VERSION="1.0.0"
export DEBUG="false"

# Paths
export BOOK_DB_PATH="/path/to/library.db"
export BOOKS_DIR="/path/to/books"

# Limits
export MAX_SEARCH_RESULTS="20"
export MAX_CHAPTER_SIZE="200000"

# Features
export ENABLE_LOGGING="true"
```

---

## 🚀 Features & Capabilities

### Current: Production-Ready Server ✅
- ✅ All MCP best practices
- ✅ Comprehensive error handling
- ✅ Full logging & validation
- ✅ **Semantic search with RAG** ✨ NEW!
- ✅ Keyword search
- ✅ Book catalog & metadata
- ✅ Chapter retrieval

### Future Enhancements
- 🔜 Hybrid search (keyword + semantic)
- 🔜 ELI5 explanations
- 🔜 Highlight/bookmark system
- 🔜 Note-taking integration
- 🔜 Query result caching

---

## 📚 Documentation

### Comprehensive Guides
Detailed documentation in the [`docs/`](docs/) folder:

- **[📖 Documentation Index](docs/README.md)** - Start here!
  - Quick reference
  - Maintenance guides
  - Technical specifications

- **[⭐ Semantic Search Guide](docs/SEMANTIC-SEARCH-COMPLETE.md)**
  - Complete implementation details
  - Usage instructions & examples
  - Performance metrics (2.1s for 272 chapters!)
  - RAG pattern explanation

- **[🏗️ Architecture Review](docs/ARCHITECTURE-REVIEW.md)**
  - MCP best practices analysis (9.5/10 score)
  - Design decisions & trade-offs
  - Scalability considerations
  - Future enhancement roadmap

### Quick Reference Files
- **QUICKSTART.md** - Fast setup guide
- **REFACTORING-SUMMARY.md** - Version history & changes
- **test_server.py** - Run comprehensive tests
- **test_semantic_setup.py** - Verify semantic search (100% coverage)

---

**Version**: 2.0.0 (Production-Ready + Semantic Search)  
**Architecture**: Modular, following MCP best practices  
**Status**: Ready for production use ✅

**Need help?** Check [`docs/`](docs/) or run `python test_server.py`

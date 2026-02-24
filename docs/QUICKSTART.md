# Book Library MCP Server — Quick Start

Connect your book library to Claude Desktop in a few minutes.

---

## Step 1: Install

```bash
cd /path/to/book-mcp-server
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Step 2: Configure Claude Desktop

Open `~/Library/Application Support/Claude/claude_desktop_config.json` and add:

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

**Required:** Use the `.venv/bin/python` path (not bare `python`). `OPENAI_API_KEY` is needed for semantic search.

---

## Step 3: Restart Claude Desktop

Quit completely (Cmd+Q), then reopen. The book-library tools will appear in Claude's tool list.

---

## Step 4: Try It

```
"What books do I have?"
"Search for Kubernetes networking"
"Create a learning path for distributed systems"
"Teach me eventual consistency from my library"
```

---

## Troubleshooting

**Server not appearing in Claude?**
- Confirm you used the full absolute `.venv/bin/python` path
- Confirm `BOOK_DB_PATH` points to an existing `.db` file
- Quit Claude fully (Cmd+Q), not just close the window

**"Module not found" error?**
```bash
source .venv/bin/activate
pip install -e .
```

**Semantic search not working?**
- Check `OPENAI_API_KEY` is set in the MCP config env block
- Verify embeddings exist: ask Claude `"refresh embeddings"` or run `python -m pytest tests/test_openai_embeddings.py`

**Check Claude's MCP logs:**
```bash
tail -f ~/Library/Logs/Claude/mcp*.log
```

---

For full tool reference, see [USER-GUIDE.md](./USER-GUIDE.md).
For contributor setup, see [CLAUDE.md](../CLAUDE.md).

# 🎉 Book Library MCP Server - Quick Start

Your MCP server is ready! Follow these 4 simple steps to connect your books to Claude Desktop.

---

## ⚡ Quick Setup (5 Minutes)

### Step 1: Install the Server

```bash
cd /path/to/book-mcp-server

# Run setup script
./setup.sh
```

**What this does:**
- Creates Python virtual environment
- Installs FastMCP
- Prepares the server

---

### Step 2: Test the Server

```bash
# Activate environment
source venv/bin/activate

# Test it works
python test_server.py
```

**Expected output:**
```
✅ Database connected
✅ list_books works
📚 Book Library
...
✅ All tests passed!
```

---

### Step 3: Configure Claude Desktop

**Open the config file:**

```bash
# Method 1: VS Code
code ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Method 2: Nano editor
nano ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Method 3: TextEdit
open -a TextEdit ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

**If the file doesn't exist, create it:**

```bash
mkdir -p ~/Library/Application\ Support/Claude/
touch ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

**Add this configuration:**

```json
{
  "mcpServers": {
    "book-library": {
      "command": "python",
      "args": [
        "/path/to/book-mcp-server/server.py"
      ],
      "env": {
        "PYTHONPATH": "/path/to/book-mcp-server/venv/lib/python3.13/site-packages"
      }
    }
  }
}
```

**If you already have MCP servers configured:**

Add the `book-library` section inside the existing `mcpServers` object:

```json
{
  "mcpServers": {
    "existing-server": {
      ...existing config...
    },
    "book-library": {
      "command": "python",
      "args": [
        "/path/to/book-mcp-server/server.py"
      ],
      "env": {
        "PYTHONPATH": "/path/to/book-mcp-server/venv/lib/python3.13/site-packages"
      }
    }
  }
}
```

---

### Step 4: Restart Claude Desktop

1. **Quit Claude Desktop completely** (Cmd+Q)
2. **Reopen Claude Desktop**
3. Look for the 🔌 MCP indicator or book-library in available tools

---

## ✅ Verify It's Working

In Claude Desktop, try asking:

> "What books do I have?"

> "List all my books"

> "Show me the table of contents for the Docker book"

**You should get responses about your 7 books!**

---

## 🎯 Example Queries to Try

### Browse Your Library
- "What books do I have in my library?"
- "How many books do I have?"
- "Show me all my Linux books"

### Search Content
- "Search for Docker networking"
- "Find chapters about systemd"
- "What chapters mention containers?"

### Read Chapters
- "Read chapter 1 of the Docker Basics book"
- "Show me chapter 3 of Learn Docker in a Month of Lunches"
- "Get the first chapter of the Ubuntu Linux Bible"

### Get Information
- "What's in the systemd book?"
- "Show me the table of contents for Node.js for Beginners"
- "Tell me about the Linux Networking book"

### Ask Questions
- "What does my Docker book say about images?"
- "Explain containers based on my books"
- "How do I use systemd according to my books?"

---

## 🐛 Troubleshooting

### MCP server not showing in Claude?

**Check 1:** Is the config file correct?
```bash
cat ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

**Check 2:** Are the paths absolute?
```bash
# Verify server exists
ls -la /path/to/book-mcp-server/server.py

# Verify Python in venv
ls -la /path/to/book-mcp-server/venv/bin/python
```

**Check 3:** Did you restart Claude Desktop completely?
- Use Cmd+Q (not just closing the window)
- Wait 5 seconds
- Reopen

### "Module not found" error?

```bash
cd /path/to/book-mcp-server
source venv/bin/activate
pip install --upgrade fastmcp
```

### Server works in terminal but not in Claude?

Check Console.app for errors:
```bash
# Open Console.app
open /Applications/Utilities/Console.app

# Filter for: Claude
# Look for Python or MCP errors
```

---

## 📊 Your Library

You have access to:

✅ **7 books** (620,000 words)  
✅ **84 chapters**  
✅ Topics: Docker, Linux, Node.js, systemd, networking

### Books:
1. Ubuntu Linux Bible (229K words, largest)
2. Learn Docker in a Month of Lunches (121K words)
3. Node.js for Beginners (86K words)
4. systemd for Linux SysAdmins (81K words)
5. Docker Basics (41K words)
6. Deep Dive Into Linux Networking (32K words)
7. Linux Coding & Programming Manual (30K words)

---

## 🚀 What's Next?

After you get this working:

### Phase 1: Use Your Library
- ✅ Ask questions about your books
- ✅ Search across all content
- ✅ Read chapters on demand

### Phase 2: Enhancements (Future)
- 🔜 Semantic search (RAG with embeddings)
- 🔜 ELI5 explanations
- 🔜 Highlight/bookmark system
- 🔜 Note-taking integration

---

## 💡 Pro Tips

1. **Be specific with book titles** - "Docker Basics" or "Ubuntu Linux Bible"
2. **Use chapter numbers** - "Chapter 3" or "the third chapter"
3. **Search is powerful** - Use keywords to find content
4. **Ask follow-ups** - Claude remembers context in the conversation

---

**Ready?** Run `./setup.sh` and let's get started! 🚀

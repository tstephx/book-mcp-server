---
name: add-mcp-tool-tests
description: Add test coverage for an MCP tool module in src/tools/. Handles the closure-based registration pattern.
disable-model-invocation: true
---

# Add MCP Tool Tests

Arguments: `<module_name>` (e.g., `analytics_tools`, `learning_tools`, `book_tools`)

## Architecture

MCP tools in `src/tools/<module>_tools.py` use a closure pattern:

```python
def register_<category>_tools(mcp: "FastMCP") -> None:
    @mcp.tool()
    def tool_name(param: str) -> str:
        # uses execute_query, execute_single from src.database
```

This means tools are **not directly importable**. Testing strategies:

1. **Test helper functions** — module-level `_helper()` functions ARE importable. Prefer this.
2. **Test via DB fixture** — mock or provide a real SQLite DB, then call the inner function indirectly by importing and invoking `register_*_tools` with a mock MCP.
3. **Test via integration** — use `execute_query` with a test DB.

## Steps

### 1. Read the module

```bash
# Get an overview of what's in the module
grep -n "^def \|^    def " src/tools/<module>_tools.py
```

Identify:
- Module-level helpers (prefixed with `_`) — directly testable
- Constants/dicts — directly testable
- Registered tools — need DB fixture or mock

### 2. Write tests for helpers first

Create or expand `tests/test_<module>_tools.py`:

```python
"""Tests for <module>_tools."""


def test_helper_function():
    """_helper should do X."""
    from src.tools.<module>_tools import _helper

    result = _helper("input")
    assert expected_condition
```

### 3. For DB-dependent tools, use a fixture

```python
import sqlite3
import pytest


@pytest.fixture
def tool_db(tmp_path):
    """Create a test DB with the tables this module queries."""
    db_path = tmp_path / "library.db"
    conn = sqlite3.connect(db_path)
    # Create ONLY the tables/columns the tool actually queries
    # Check ref/db-schema.md for column definitions
    conn.execute(
        "CREATE TABLE books (id TEXT PRIMARY KEY, title TEXT, author TEXT, word_count INTEGER)"
    )
    conn.execute(
        "CREATE TABLE chapters (id TEXT PRIMARY KEY, book_id TEXT, "
        "chapter_number INTEGER, title TEXT, word_count INTEGER, embedding BLOB)"
    )
    # Insert test data
    conn.execute("INSERT INTO books VALUES ('b1', 'Test Book', 'Author', 5000)")
    conn.commit()
    conn.close()
    return db_path


def test_tool_via_db(tool_db):
    """Test tool function with a real DB."""
    from src.config import Config

    Config.DB_PATH = tool_db
    from src.tools.<module>_tools import _helper_that_uses_db

    result = _helper_that_uses_db("b1")
    assert result is not None
```

### 4. Test edge cases

Every test file should include:
- **Empty input**: `tool("")` or `tool([])`
- **Missing data**: query returns no rows
- **Type assertions**: verify return types (`isinstance(result, dict)`)

### 5. Run and verify

```bash
.venv/bin/python -m pytest tests/test_<module>_tools.py -v
.venv/bin/python -m pytest tests/ -x -q  # full suite
```

### 6. Commit

```bash
git add tests/test_<module>_tools.py
git commit -m "test: add test coverage for <module>_tools"
```

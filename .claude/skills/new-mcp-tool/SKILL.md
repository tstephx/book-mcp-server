---
name: new-mcp-tool
description: Scaffold a new MCP tool with registration, tests, and ref doc update. Use when adding a new tool to the book-library or agentic-pipeline MCP server.
disable-model-invocation: true
---

# New MCP Tool

Arguments: `<tool_name>` `<server: library|pipeline>` `<description>`

## Architecture

**Library server** tools live in `src/tools/<category>_tools.py`. Each file exports a `register_<category>_tools(mcp)` function that decorates tools with `@mcp.tool()`. Registration happens in `src/server.py` via explicit import + call.

**Pipeline server** tools live directly in `agentic_pipeline/mcp_server.py` as module-level functions decorated with `@mcp.tool()`.

## Steps

### 1. Determine target location

- **library** — Find the best-fit category file in `src/tools/`. If no existing file matches, create a new `src/tools/<category>_tools.py` following the pattern.
- **pipeline** — Add to `agentic_pipeline/mcp_server.py`.

### 2. Read existing patterns

Read the target file to match:
- Import style (`from ..database import execute_query, execute_single, DatabaseError`)
- Error handling pattern (try/except returning error strings, not raising)
- Type hints and docstrings
- Return format (formatted strings for library tools, dicts for pipeline tools)

### 3. Write the tool function

For **library** tools:
```python
@mcp.tool()
def tool_name(param: str) -> str:
    """Tool description for Claude to read"""
    try:
        # implementation
        return formatted_result
    except DatabaseError as e:
        return f"Database error: {e}"
    except ValidationError as e:
        return str(e)
    except Exception as e:
        logger.error(f"tool_name failed: {e}")
        return f"Error: {e}"
```

For **pipeline** tools:
```python
@mcp.tool()
def tool_name(param: str) -> dict:
    """Tool description for Claude to read"""
    try:
        db_path = get_db_path()
        # implementation
        return {"success": True, ...}
    except Exception as e:
        return {"error": str(e)}
```

### 4. Register (library tools only)

If you created a new category file:
1. Add `from .tools.<category>_tools import register_<category>_tools` to `src/server.py` imports
2. Add `register_<category>_tools(mcp)` call in the registration block (~line 67-79)

### 5. Write tests

Create `tests/test_<tool_name>.py` or append to existing test file:

```python
"""Tests for <tool_name> tool."""
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest

# Minimum 3 tests: happy path, edge case, error handling
def test_<tool_name>_happy_path():
    """Basic functionality test."""
    ...

def test_<tool_name>_edge_case():
    """Empty input / no results / boundary condition."""
    ...

def test_<tool_name>_error_handling():
    """Verify errors return gracefully, not raise."""
    ...
```

### 6. Update ref doc

Append the new tool to `ref/mcp-tools.md` in the appropriate table section.

### 7. Verify

```bash
python -m pytest tests/test_<tool_name>.py -v
ruff check src/tools/<file>.py tests/test_<tool_name>.py
```

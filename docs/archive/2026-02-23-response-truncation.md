# Response Truncation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent large MCP tool responses from blowing up the LLM context window by (1) adding a `limit` param to `get_topic_coverage` and (2) adding a server-side default `max_tokens` to `get_chapter` and `get_section`.

**Architecture:** Two surgical changes. `get_topic_coverage` in `discovery_tools.py` currently returns every matching chapter with no cap — add `limit: int = 20` to cap results and sort best-first. `get_chapter`/`get_section` in `chapter_tools.py` already support `max_tokens` but it's opt-in with no default — add `Config.DEFAULT_CHAPTER_TOKENS = 8000` and apply it as the fallback when `max_tokens` is None.

**Tech Stack:** Python, pytest, unittest.mock, `src/config.py`, `src/tools/discovery_tools.py`, `src/tools/chapter_tools.py`

---

### Task 1: Add `DEFAULT_CHAPTER_TOKENS` to Config

**Files:**
- Modify: `src/config.py:28-30`

**Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_default_chapter_tokens_exists():
    from src.config import Config
    assert hasattr(Config, 'DEFAULT_CHAPTER_TOKENS')
    assert Config.DEFAULT_CHAPTER_TOKENS == 8000

def test_default_chapter_tokens_env_override():
    import os
    from importlib import reload
    import src.config as config_module
    os.environ['DEFAULT_CHAPTER_TOKENS'] = '4000'
    reload(config_module)
    assert config_module.Config.DEFAULT_CHAPTER_TOKENS == 4000
    del os.environ['DEFAULT_CHAPTER_TOKENS']
    reload(config_module)
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/taylorstephens/Dev/_Projects/book-mcp-server
.venv/bin/pytest tests/test_config.py::test_default_chapter_tokens_exists -v
```
Expected: `FAILED — AttributeError: type object 'Config' has no attribute 'DEFAULT_CHAPTER_TOKENS'`

**Step 3: Add the config constant**

In `src/config.py`, in the `# Search limits` block after line 30, add:

```python
DEFAULT_CHAPTER_TOKENS: int = int(os.getenv("DEFAULT_CHAPTER_TOKENS", "8000"))
```

Also add to the `display()` method output under `Limits:`:
```
  Default chapter tokens: {cls.DEFAULT_CHAPTER_TOKENS}
```

**Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_config.py::test_default_chapter_tokens_exists tests/test_config.py::test_default_chapter_tokens_env_override -v
```
Expected: both PASS

**Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: add DEFAULT_CHAPTER_TOKENS config constant (default 8000)"
```

---

### Task 2: Apply default `max_tokens` in `get_chapter` and `get_section`

**Files:**
- Modify: `src/tools/chapter_tools.py:305` (`get_chapter` signature)
- Modify: `src/tools/chapter_tools.py:448` (`get_section` signature)

**Context:** Both functions currently have `max_tokens: Optional[int] = None`. The fix is to default to `Config.DEFAULT_CHAPTER_TOKENS` when `None` is passed. The truncation logic already exists — we just need to ensure it always fires.

**Step 1: Write the failing test**

Add to a new file `tests/test_chapter_truncation.py`:

```python
"""Tests that get_chapter and get_section apply a default token cap."""
from unittest.mock import patch, MagicMock
from pathlib import Path


LONG_CONTENT = "word " * 40000  # ~200KB, well over 8000 tokens


def _make_chapter_path(tmp_path, content):
    chapter_file = tmp_path / "chapter.md"
    chapter_file.write_text(content)
    return chapter_file


def test_get_chapter_truncates_by_default(tmp_path):
    chapter_file = _make_chapter_path(tmp_path, LONG_CONTENT)

    with patch("src.tools.chapter_tools.execute_single") as mock_db, \
         patch("src.tools.chapter_tools._find_chapter_path") as mock_path, \
         patch("src.tools.chapter_tools.validate_book_id", return_value="book-123"), \
         patch("src.tools.chapter_tools.validate_chapter_number", return_value=1):

        mock_db.return_value = {"title": "Ch1", "file_path": str(chapter_file), "word_count": 40000}
        mock_path.return_value = (chapter_file, False)

        from src.tools.chapter_tools import register_chapter_tools
        mcp = MagicMock()
        captured = {}

        def tool_decorator(func):
            captured[func.__name__] = func
            return func

        mcp.tool.return_value = tool_decorator
        register_chapter_tools(mcp)

        result = captured["get_chapter"]("book-123", 1)  # no max_tokens arg
        assert "truncated" in result.lower() or len(result) < len(LONG_CONTENT)


def test_get_chapter_explicit_max_tokens_overrides_default(tmp_path):
    chapter_file = _make_chapter_path(tmp_path, LONG_CONTENT)

    with patch("src.tools.chapter_tools.execute_single") as mock_db, \
         patch("src.tools.chapter_tools._find_chapter_path") as mock_path, \
         patch("src.tools.chapter_tools.validate_book_id", return_value="book-123"), \
         patch("src.tools.chapter_tools.validate_chapter_number", return_value=1):

        mock_db.return_value = {"title": "Ch1", "file_path": str(chapter_file), "word_count": 40000}
        mock_path.return_value = (chapter_file, False)

        from src.tools.chapter_tools import register_chapter_tools
        mcp = MagicMock()
        captured = {}

        def tool_decorator(func):
            captured[func.__name__] = func
            return func

        mcp.tool.return_value = tool_decorator
        register_chapter_tools(mcp)

        result = captured["get_chapter"]("book-123", 1, max_tokens=500)
        # Should be much shorter than default truncation
        assert len(result) < 8000
```

**Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_chapter_truncation.py::test_get_chapter_truncates_by_default -v
```
Expected: FAIL — result is full 200KB content, assertion fails

**Step 3: Implement the fix in `get_chapter`**

In `src/tools/chapter_tools.py`, find the `get_chapter` function. The `max_tokens` parameter is used in two places (around lines 350 and 408). Change the function signature and add a default resolution at the top:

Current signature (line 305):
```python
def get_chapter(book_id: str, chapter_number: int, max_tokens: Optional[int] = None) -> str:
```

Change to apply config default — add this line right after the `try:` block opens (after `book_id = validate_book_id(...)`):
```python
# Apply server-side default if caller did not specify
if max_tokens is None:
    from src.config import Config
    max_tokens = Config.DEFAULT_CHAPTER_TOKENS
```

Do the same in `get_section` (line 448), in the same position after input validation.

**Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_chapter_truncation.py -v
```
Expected: both PASS

**Step 5: Run full test suite to check for regressions**

```bash
.venv/bin/pytest tests/ -x -q 2>&1 | tail -20
```
Expected: no new failures

**Step 6: Commit**

```bash
git add src/tools/chapter_tools.py tests/test_chapter_truncation.py
git commit -m "feat: apply DEFAULT_CHAPTER_TOKENS cap in get_chapter and get_section"
```

---

### Task 3: Add `limit` param to `get_topic_coverage`

**Files:**
- Modify: `src/tools/discovery_tools.py:176` (`get_topic_coverage` signature)
- Test: `tests/test_discovery_tools.py` (create new)

**Context:** Currently returns all matching chapters sorted by avg_similarity. With 185+ books and a broad topic like "Python", this can return 100+ chapters = 39KB. Add `limit: int = 20` to cap the final result list. The results are already sorted best-first, so we just slice before returning.

**Step 1: Write the failing test**

Create `tests/test_discovery_tools.py`:

```python
"""Tests for get_topic_coverage result limiting."""
from unittest.mock import patch, MagicMock
import numpy as np


def _make_chunk_metadata(n):
    """Generate n fake chunks across n different chapters/books."""
    return [
        {
            "chunk_id": f"c{i}:0",
            "book_id": f"book-{i}",
            "book_title": f"Book {i}",
            "author": "Author",
            "chapter_id": f"chapter-{i}",
            "chapter_title": f"Chapter {i}",
            "chapter_number": 1,
            "word_count": 500,
            "excerpt": "some text",
        }
        for i in range(n)
    ]


def test_get_topic_coverage_respects_limit():
    """get_topic_coverage should return at most `limit` books."""
    n = 50
    meta = _make_chunk_metadata(n)
    embeddings = np.random.rand(n, 3072).astype(np.float32)
    # Normalize so cosine similarity works
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    with patch("src.tools.discovery_tools._load_chunk_data") as mock_data, \
         patch("src.tools.discovery_tools.embedding_model_context") as mock_ctx, \
         patch("src.tools.discovery_tools.cosine_similarity") as mock_sim:

        mock_data.return_value = (embeddings, meta)
        mock_gen = MagicMock()
        mock_gen.generate.return_value = np.random.rand(3072)
        mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_gen)
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
        mock_sim.return_value = 0.9  # all chunks pass threshold

        from src.tools.discovery_tools import register_discovery_tools
        mcp = MagicMock()
        captured = {}

        def tool_decorator(func):
            captured[func.__name__] = func
            return func

        mcp.tool.return_value = tool_decorator
        register_discovery_tools(mcp)

        result = captured["get_topic_coverage"]("python", limit=5)
        assert "books" in result or isinstance(result, dict)
        # Should have at most 5 books
        if isinstance(result, dict) and "books" in result:
            assert len(result["books"]) <= 5


def test_get_topic_coverage_default_limit_is_20():
    """Default limit should be 20 books."""
    n = 40
    meta = _make_chunk_metadata(n)
    embeddings = np.random.rand(n, 3072).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    with patch("src.tools.discovery_tools._load_chunk_data") as mock_data, \
         patch("src.tools.discovery_tools.embedding_model_context") as mock_ctx, \
         patch("src.tools.discovery_tools.cosine_similarity") as mock_sim:

        mock_data.return_value = (embeddings, meta)
        mock_gen = MagicMock()
        mock_gen.generate.return_value = np.random.rand(3072)
        mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_gen)
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
        mock_sim.return_value = 0.9

        from src.tools.discovery_tools import register_discovery_tools
        mcp = MagicMock()
        captured = {}

        def tool_decorator(func):
            captured[func.__name__] = func
            return func

        mcp.tool.return_value = tool_decorator
        register_discovery_tools(mcp)

        result = captured["get_topic_coverage"]("python")  # no limit arg
        if isinstance(result, dict) and "books" in result:
            assert len(result["books"]) <= 20
```

**Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_discovery_tools.py::test_get_topic_coverage_respects_limit -v
```
Expected: FAIL — `get_topic_coverage` has no `limit` param, TypeError

**Step 3: Implement the fix**

In `src/tools/discovery_tools.py`, change the `get_topic_coverage` signature from:
```python
def get_topic_coverage(
    topic: str,
    min_similarity: float = 0.3,
    include_excerpts: bool = True
) -> dict:
```

To:
```python
def get_topic_coverage(
    topic: str,
    min_similarity: float = 0.3,
    include_excerpts: bool = True,
    limit: int = 20,
) -> dict:
```

Update the docstring Args section to add:
```
limit: Maximum number of books to return (default: 20). Results are sorted
       by average similarity, so highest-relevance books are returned first.
```

Then find the line that builds `sorted_books` (around line 268-272):
```python
sorted_books = sorted(
    books_coverage.values(),
    key=lambda x: x['avg_similarity'],
    reverse=True
)
```

Add the slice immediately after:
```python
sorted_books = sorted_books[:max(1, limit)]
```

**Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_discovery_tools.py -v
```
Expected: both PASS

**Step 5: Run full test suite**

```bash
.venv/bin/pytest tests/ -x -q 2>&1 | tail -20
```
Expected: no new failures

**Step 6: Commit**

```bash
git add src/tools/discovery_tools.py tests/test_discovery_tools.py
git commit -m "feat: add limit param to get_topic_coverage (default 20 books)"
```

---

### Task 4: Verify end-to-end with a manual smoke test

**Step 1: Check the MCP server starts cleanly**

```bash
cd /Users/taylorstephens/Dev/_Projects/book-mcp-server
.venv/bin/python -c "
from src.tools.discovery_tools import register_discovery_tools
from src.tools.chapter_tools import register_chapter_tools
from src.config import Config
print('DEFAULT_CHAPTER_TOKENS:', Config.DEFAULT_CHAPTER_TOKENS)
print('All imports OK')
"
```
Expected: prints `DEFAULT_CHAPTER_TOKENS: 8000` and `All imports OK`

**Step 2: Run full test suite one final time**

```bash
.venv/bin/pytest tests/ -q 2>&1 | tail -5
```
Expected: all pass, no regressions

**Step 3: Final commit if any cleanup needed, otherwise done**

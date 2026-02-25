# Slug ID Resolution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow callers to pass slug-style book IDs (e.g. `agentic-design-patterns-a-hands-on-guide`) or partial titles instead of UUIDs, and return "did you mean?" suggestions when no match is found.

**Architecture:** Two changes to `src/utils/validators.py`. First, `resolve_book_id()` — a new function that accepts any string, checks if it's a valid UUID (fast path), then falls back to fuzzy title matching in the DB (slug → spaces, LIKE query). Returns the resolved UUID or raises with a "did you mean?" error listing close matches. Second, all existing `validate_book_id()` call sites in `book_tools.py` and `chapter_tools.py` are replaced with `resolve_book_id()`. The old `validate_book_id()` stays as an internal helper used by `resolve_book_id()`.

**Tech Stack:** Python, sqlite3 (via existing `execute_query`/`execute_single`), pytest, unittest.mock

---

### Task 1: Add `resolve_book_id()` to validators

**Files:**
- Modify: `src/utils/validators.py`
- Test: `tests/test_validators.py` (create new)

**Context:** `validate_book_id()` currently rejects anything that isn't a full UUID. The new `resolve_book_id()` wraps it with a DB-backed fallback. It converts the input slug to a search query (hyphens → spaces, strip extra whitespace), runs a case-insensitive LIKE query on `books.title`, and returns the matching UUID. If multiple matches, returns the best one. If zero matches, raises `ValidationError` with a "did you mean?" message listing close candidates.

**Step 1: Write failing tests**

Create `tests/test_validators.py`:

```python
"""Tests for resolve_book_id slug resolution and did-you-mean errors."""
from unittest.mock import patch
import pytest
from src.utils.validators import resolve_book_id, ValidationError


VALID_UUID = "438ed6e4-d90f-4996-9dd3-460c45fbba67"
VALID_UUID_2 = "922e5dd7-defd-421e-92a4-dedc6bf87275"


def test_resolve_book_id_passes_valid_uuid_through():
    """A valid UUID bypasses DB lookup entirely."""
    with patch("src.utils.validators.execute_single") as mock_db:
        result = resolve_book_id(VALID_UUID)
    assert result == VALID_UUID
    mock_db.assert_not_called()


def test_resolve_book_id_slug_resolves_to_uuid():
    """A slug is converted to spaces and matched against book titles."""
    with patch("src.utils.validators.execute_query") as mock_query:
        mock_query.return_value = [
            {"id": VALID_UUID, "title": "Agentic Design Patterns A Hands On Guide"}
        ]
        result = resolve_book_id("agentic-design-patterns-a-hands-on-guide")
    assert result == VALID_UUID


def test_resolve_book_id_slug_single_match_returns_uuid():
    """Single fuzzy match returns that book's UUID."""
    with patch("src.utils.validators.execute_query") as mock_query:
        mock_query.return_value = [
            {"id": VALID_UUID_2, "title": "Arduino Programming Essentials"}
        ]
        result = resolve_book_id("arduino-programming-essentials")
    assert result == VALID_UUID_2


def test_resolve_book_id_no_match_raises_with_suggestions():
    """No match raises ValidationError with did-you-mean suggestions."""
    with patch("src.utils.validators.execute_query") as mock_query:
        # First call (exact match): returns empty
        # Second call (broad search for suggestions): returns candidates
        mock_query.side_effect = [
            [],  # no fuzzy match
            [
                {"id": VALID_UUID, "title": "Agentic Design Patterns"},
                {"id": VALID_UUID_2, "title": "Agentic AI Foundations"},
            ],
        ]
        with pytest.raises(ValidationError) as exc_info:
            resolve_book_id("agentic-something-unknown")
    assert "did you mean" in str(exc_info.value).lower()
    assert "Agentic Design Patterns" in str(exc_info.value)


def test_resolve_book_id_no_match_no_suggestions_raises_plain_error():
    """No match and no suggestions raises a plain invalid-ID error."""
    with patch("src.utils.validators.execute_query") as mock_query:
        mock_query.side_effect = [[], []]  # no matches at all
        with pytest.raises(ValidationError) as exc_info:
            resolve_book_id("completely-unknown-slug")
    assert "not found" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()


def test_resolve_book_id_empty_raises():
    """Empty string raises immediately."""
    with pytest.raises(ValidationError):
        resolve_book_id("")


def test_resolve_book_id_multiple_matches_returns_first():
    """Multiple matches returns the first (highest relevance) result."""
    with patch("src.utils.validators.execute_query") as mock_query:
        mock_query.return_value = [
            {"id": VALID_UUID, "title": "Docker Deep Dive"},
            {"id": VALID_UUID_2, "title": "Docker Networking Guide"},
        ]
        result = resolve_book_id("docker-deep-dive")
    assert result == VALID_UUID
```

**Step 2: Run to confirm all tests fail**

```bash
cd /Users/taylorstephens/Dev/_Projects/book-mcp-server
.venv/bin/pytest tests/test_validators.py -v
```
Expected: `ImportError` — `resolve_book_id` does not exist yet

**Step 3: Implement `resolve_book_id()` in `src/utils/validators.py`**

Add this import at the top of `src/utils/validators.py` (after existing imports):
```python
from typing import Optional, List
```

Add this function at the end of `src/utils/validators.py`:

```python
def resolve_book_id(book_id: str) -> str:
    """
    Resolve a book ID to a valid UUID.

    Accepts:
    - Full UUID (fast path, no DB lookup)
    - Slug-style string (hyphens converted to spaces, fuzzy title match)

    Returns the book's UUID. Raises ValidationError with a "did you mean?"
    message if no match found but candidates exist, or a plain error if no
    candidates exist at all.

    Args:
        book_id: UUID or slug-style book identifier

    Returns:
        Resolved UUID string

    Raises:
        ValidationError: If the book cannot be resolved
    """
    # Import here to avoid circular imports at module load time
    from ..database import execute_query

    if not book_id:
        raise ValidationError("Book ID cannot be empty")

    # Fast path: already a valid UUID
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    if re.match(uuid_pattern, book_id, re.IGNORECASE):
        return book_id

    # Slug fallback: convert hyphens to spaces for title matching
    search_term = book_id.replace('-', ' ').replace('_', ' ').strip()

    # Try fuzzy LIKE match against titles
    matches = execute_query(
        "SELECT id, title FROM books WHERE LOWER(title) LIKE LOWER(?) ORDER BY title LIMIT 5",
        (f"%{search_term}%",)
    )

    if matches:
        # Return the first (best) match
        return matches[0]['id']

    # No match — fetch candidates for did-you-mean using first word of slug
    first_word = search_term.split()[0] if search_term.split() else search_term
    candidates = execute_query(
        "SELECT id, title FROM books WHERE LOWER(title) LIKE LOWER(?) ORDER BY title LIMIT 3",
        (f"%{first_word}%",)
    )

    if candidates:
        suggestions = ", ".join(f'"{c["title"]}"' for c in candidates)
        raise ValidationError(
            f"Book not found for: '{book_id}'. Did you mean: {suggestions}? "
            f"Use search_titles() to find the correct book ID."
        )

    raise ValidationError(
        f"Book not found for: '{book_id}'. "
        f"Use search_titles() to find available books and their IDs."
    )
```

**Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_validators.py -v
```
Expected: all 7 PASS

**Step 5: Commit**

```bash
git add src/utils/validators.py tests/test_validators.py
git commit -m "feat: add resolve_book_id() with slug fallback and did-you-mean errors"
```

---

### Task 2: Replace `validate_book_id` with `resolve_book_id` in tool files

**Files:**
- Modify: `src/tools/book_tools.py` (2 call sites: lines 69, 143)
- Modify: `src/tools/chapter_tools.py` (3 call sites: lines 320, 468, 617)

**Context:** Every place that currently calls `validate_book_id(book_id)` must be changed to `resolve_book_id(book_id)`. The import line in each file must also be updated. `validate_book_id` remains in `validators.py` for internal use but is no longer called from tool files.

**Step 1: Write failing tests**

Add to `tests/test_validators.py` — integration-style tests that mock the DB at a higher level:

```python
def test_get_book_info_accepts_slug(tmp_path):
    """get_book_info should resolve slug IDs, not reject them."""
    with patch("src.tools.book_tools.resolve_book_id", return_value=VALID_UUID) as mock_resolve, \
         patch("src.tools.book_tools.execute_single") as mock_db, \
         patch("src.tools.book_tools.execute_query") as mock_chapters:
        mock_db.return_value = {
            "id": VALID_UUID, "title": "Docker Deep Dive", "author": "Nigel Poulton",
            "word_count": 50000, "added_date": "2024-01-01"
        }
        mock_chapters.return_value = []
        from src.tools.book_tools import register_book_tools
        from unittest.mock import MagicMock
        mcp = MagicMock()
        captured = {}
        def tool_decorator(func):
            captured[func.__name__] = func
            return func
        mcp.tool.return_value = tool_decorator
        register_book_tools(mcp)
        result = captured["get_book_info"]("docker-deep-dive")
        mock_resolve.assert_called_once_with("docker-deep-dive")
        assert "Docker" in result or VALID_UUID in result
```

**Step 2: Run to confirm test fails**

```bash
.venv/bin/pytest tests/test_validators.py::test_get_book_info_accepts_slug -v
```
Expected: FAIL — `resolve_book_id` is not called (still uses `validate_book_id`)

**Step 3: Update `src/tools/book_tools.py`**

Change import line from:
```python
from ..utils.validators import validate_book_id, ValidationError
```
To:
```python
from ..utils.validators import resolve_book_id, ValidationError
```

Change both call sites:
- Line ~69: `book_id = validate_book_id(book_id)` → `book_id = resolve_book_id(book_id)`
- Line ~143: `book_id = validate_book_id(book_id)` → `book_id = resolve_book_id(book_id)`

**Step 4: Update `src/tools/chapter_tools.py`**

Change import line from:
```python
from ..utils.validators import validate_book_id, validate_chapter_number, ValidationError
```
To:
```python
from ..utils.validators import resolve_book_id, validate_chapter_number, ValidationError
```

Change all three call sites:
- Line ~320: `book_id = validate_book_id(book_id)` → `book_id = resolve_book_id(book_id)`
- Line ~468: `book_id = validate_book_id(book_id)` → `book_id = resolve_book_id(book_id)`
- Line ~617: `book_id = validate_book_id(book_id)` → `book_id = resolve_book_id(book_id)`

**Step 5: Run all validator and chapter/book tests**

```bash
.venv/bin/pytest tests/test_validators.py tests/test_chapter_truncation.py -v
```
Expected: all PASS

**Step 6: Run full suite**

```bash
.venv/bin/pytest tests/ -x -q 2>&1 | tail -20
```
Expected: same 2 pre-existing failures, no new failures

**Step 7: Commit**

```bash
git add src/tools/book_tools.py src/tools/chapter_tools.py
git commit -m "feat: use resolve_book_id in book and chapter tools (slug support)"
```

---

### Task 3: Smoke test

**Step 1: Verify imports work cleanly**

```bash
cd /Users/taylorstephens/Dev/_Projects/book-mcp-server
.venv/bin/python -c "
from src.utils.validators import resolve_book_id, validate_book_id, ValidationError
from src.tools.book_tools import register_book_tools
from src.tools.chapter_tools import register_chapter_tools
print('All imports OK')
"
```
Expected: `All imports OK`

**Step 2: Verify slug resolution logic directly**

```bash
.venv/bin/python -c "
from unittest.mock import patch
from src.utils.validators import resolve_book_id, ValidationError

# Test UUID fast path (no DB needed)
uuid = '438ed6e4-d90f-4996-9dd3-460c45fbba67'
result = resolve_book_id(uuid)
assert result == uuid, f'UUID passthrough failed: {result}'
print('UUID fast path: OK')

# Test did-you-mean error format
with patch('src.utils.validators.execute_query') as mock:
    mock.side_effect = [[], [{'id': uuid, 'title': 'Docker Deep Dive'}]]
    try:
        resolve_book_id('docker-something-unknown')
    except ValidationError as e:
        msg = str(e)
        assert 'did you mean' in msg.lower(), f'Missing did-you-mean: {msg}'
        assert 'Docker Deep Dive' in msg, f'Missing suggestion: {msg}'
        print('Did-you-mean error: OK')

print('All smoke tests passed')
"
```
Expected: all OK messages

**Step 3: Final full test run**

```bash
.venv/bin/pytest tests/ -q 2>&1 | tail -5
```
Expected: 2 pre-existing failures only

---
status: complete
tags: [project/book-mcp-server, format/plan]
type: project
created: '2026-03-05'
modified: '2026-03-05'
---

# Backlog Cleanup Implementation Plan — COMPLETE
<!-- project: book-mcp-server -->

> **Status:** All 6 tasks completed on 2026-03-05. Commits: 876e2fd, d3801e3, f4e32b1, b4c4b1e, 660bc25, 1ca9556, 55e1b61.

**Goal:** Fix minor code bugs, add CLI for chapter title management, clean up library data quality, and add initial src/ test coverage.

**Architecture:** Four independent work streams — code bug fixes (direct edits), CLI commands (Click pattern matching existing commands), library data cleanup (SQL against shared DB), and test coverage (pytest against src/tools/). All are additive and don't require architectural changes.

**Tech Stack:** Python, Click (CLI), SQLite, pytest

---

## Scope Clarification

Two items from the original backlog are resolved and excluded:
- **DB context manager refactor** — Only `migrations.py` uses raw `sqlite3.connect`, which is by design (tables don't exist before migrations run; `get_pipeline_db()` enables foreign keys which would fail on missing tables). Not a bug.
- **Unused NEEDS_RETRY → HASHING transition** — Actually used. `_retry_one()` calls `_process_book()` which transitions to HASHING. Verified in `orchestrator.py:332,513`. Not a bug.

---

### Task 1: Fix `import re` placement in llm_fallback_adapter.py

**Files:**
- Modify: `agentic_pipeline/adapters/llm_fallback_adapter.py`

**Step 1: Move the inline import to the top of the file**

The `import re` at line 159 is inside `_call_llm()`. Move it to the module-level imports (after `import json` at line ~4).

```python
# At top of file, add to existing imports:
import re
```

Then delete the inline `import re` at line 159.

**Step 2: Run tests to verify**

Run: `python -m pytest tests/test_llm_fallback_adapter.py -v`
Expected: All pass

**Step 3: Commit**

```bash
git add agentic_pipeline/adapters/llm_fallback_adapter.py
git commit -m "fix: move inline import re to module level in llm_fallback_adapter"
```

---

### Task 2: Fix unguarded `sources[0]` access in learning_tools.py

**Files:**
- Modify: `src/tools/learning_tools.py`
- Create: `tests/test_learning_tools.py`

**Context:** Five call sites access `sources[0]` which could IndexError if `sources` is empty. Lines 414 and 426 are already behind `if sources:` guards. Lines 484 and 605 are inside `if impact_sentences:` / `if tradeoff_sentences:` — the sentences are extracted from `sources`, so if sentences exist then sources exist. Still fragile — add explicit guards.

**Step 1: Write failing tests**

```python
"""Tests for learning_tools edge cases."""

import pytest
from unittest.mock import patch, MagicMock


def test_generate_business_impact_empty_sources():
    """_generate_business_impact should return fallback when sources is empty."""
    from src.tools.learning_tools import _generate_business_impact

    result = _generate_business_impact("kubernetes", [])
    assert "kubernetes" in result.lower() or "informed decisions" in result.lower()
    # Must NOT raise IndexError


def test_generate_tradeoffs_empty_sources():
    """_generate_tradeoffs should return fallback when sources is empty."""
    from src.tools.learning_tools import _generate_tradeoffs

    result = _generate_tradeoffs("kubernetes", [])
    assert "kubernetes" in result.lower() or "tradeoffs" in result.lower()
    # Must NOT raise IndexError


def test_generate_quick_summary_empty_sources():
    """_generate_quick_summary should return fallback when sources is empty."""
    from src.tools.learning_tools import _generate_quick_summary

    result = _generate_quick_summary("kubernetes", [])
    assert "kubernetes" in result.lower()
```

**Step 2: Run tests to verify they fail (or pass — the empty-list path may already hit the fallback return)**

Run: `python -m pytest tests/test_learning_tools.py -v`

Note: The `sources[0]` at lines 484 and 605 is only reached when `impact_sentences` / `tradeoff_sentences` is non-empty, which requires `sources` to be non-empty. So these won't IndexError in practice. However, fix them anyway for defensive coding — change the attribution to use the source that actually produced the sentence.

**Step 3: Add guards to the two fragile sites**

In `_generate_business_impact` (around line 483-484):
```python
    if impact_sentences:
        attribution = f"(from *{sources[0]['book_title']}*)" if sources else ""
        return "\n\n".join(impact_sentences) + (f"\n\n{attribution}" if attribution else "")
```

In `_generate_tradeoffs` (around line 604-605):
```python
    if tradeoff_sentences:
        attribution = f"(from *{sources[0]['book_title']}*)" if sources else ""
        return "\n\n".join(tradeoff_sentences) + (f"\n\n{attribution}" if attribution else "")
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_learning_tools.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add src/tools/learning_tools.py tests/test_learning_tools.py
git commit -m "fix: guard sources[0] access in learning_tools.py"
```

---

### Task 3: Add CLI `update-title` command

**Files:**
- Modify: `agentic_pipeline/cli.py`
- Create: `tests/test_cli_update_title.py`

**Context:** Currently chapter titles can only be fixed via raw SQL. Add a proper CLI command following the existing Click pattern. The command needs to update the `chapters` table in the shared library DB (same DB as pipeline).

**Step 1: Write failing test**

```python
"""Tests for the update-title CLI command."""

import sqlite3
import pytest
from click.testing import CliRunner
from agentic_pipeline.cli import main


@pytest.fixture
def db_with_book(tmp_path):
    """Create a test DB with a book and chapters."""
    db_path = tmp_path / "library.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE books (id TEXT PRIMARY KEY, title TEXT, author TEXT, word_count INTEGER)"
    )
    conn.execute(
        "CREATE TABLE chapters (id TEXT PRIMARY KEY, book_id TEXT, chapter_number INTEGER, title TEXT, word_count INTEGER)"
    )
    conn.execute(
        "INSERT INTO books VALUES ('book-1', 'Test Book', 'Author', 1000)"
    )
    conn.execute(
        "INSERT INTO chapters VALUES ('ch-1', 'book-1', 1, 'Old Title', 500)"
    )
    conn.execute(
        "INSERT INTO chapters VALUES ('ch-2', 'book-1', 2, 'Chapter Two', 500)"
    )
    conn.commit()
    conn.close()
    return db_path


def test_update_title_by_chapter_number(db_with_book, monkeypatch):
    """update-title should update the chapter title."""
    monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(db_with_book))
    runner = CliRunner()
    result = runner.invoke(main, ["update-title", "book-1", "1", "New Title"])
    assert result.exit_code == 0
    assert "New Title" in result.output

    # Verify in DB
    conn = sqlite3.connect(db_with_book)
    row = conn.execute(
        "SELECT title FROM chapters WHERE book_id = 'book-1' AND chapter_number = 1"
    ).fetchone()
    conn.close()
    assert row[0] == "New Title"


def test_update_title_nonexistent_chapter(db_with_book, monkeypatch):
    """update-title should error for nonexistent chapter."""
    monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(db_with_book))
    runner = CliRunner()
    result = runner.invoke(main, ["update-title", "book-1", "99", "New Title"])
    assert result.exit_code == 0
    assert "not found" in result.output.lower() or "no chapter" in result.output.lower()


def test_update_title_nonexistent_book(db_with_book, monkeypatch):
    """update-title should error for nonexistent book."""
    monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(db_with_book))
    runner = CliRunner()
    result = runner.invoke(main, ["update-title", "nonexistent", "1", "New Title"])
    assert result.exit_code == 0
    assert "not found" in result.output.lower() or "no book" in result.output.lower()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli_update_title.py -v`
Expected: FAIL — `update-title` command doesn't exist

**Step 3: Implement the command**

Add to `agentic_pipeline/cli.py` after the existing commands:

```python
@main.command("update-title")
@click.argument("book_id")
@click.argument("chapter_number", type=int)
@click.argument("new_title")
def update_title(book_id: str, chapter_number: int, new_title: str):
    """Update a chapter's title.

    BOOK_ID is the book UUID or title slug.
    CHAPTER_NUMBER is the chapter number to update.
    NEW_TITLE is the new title to set.
    """
    from .db.config import get_db_path

    import sqlite3

    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        # Verify book exists
        book = conn.execute(
            "SELECT id, title FROM books WHERE id = ?", (book_id,)
        ).fetchone()
        if not book:
            # Try fuzzy match
            book = conn.execute(
                "SELECT id, title FROM books WHERE title LIKE ?",
                (f"%{book_id}%",),
            ).fetchone()
        if not book:
            console.print(f"[red]No book found matching '{book_id}'[/red]")
            return

        # Verify chapter exists
        chapter = conn.execute(
            "SELECT id, title FROM chapters WHERE book_id = ? AND chapter_number = ?",
            (book["id"], chapter_number),
        ).fetchone()
        if not chapter:
            console.print(
                f"[red]No chapter {chapter_number} found in '{book['title']}'[/red]"
            )
            return

        old_title = chapter["title"]
        conn.execute(
            "UPDATE chapters SET title = ? WHERE id = ?",
            (new_title, chapter["id"]),
        )
        conn.commit()
        console.print(f"[green]Updated chapter {chapter_number}:[/green]")
        console.print(f"  Old: {old_title}")
        console.print(f"  New: {new_title}")
    finally:
        conn.close()
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_cli_update_title.py -v`
Expected: All 3 pass

**Step 5: Run full test suite**

Run: `python -m pytest tests/ -x -q`
Expected: All pass

**Step 6: Commit**

```bash
git add agentic_pipeline/cli.py tests/test_cli_update_title.py
git commit -m "feat: add update-title CLI command for chapter title management"
```

---

### Task 4: Add CLI `library-issues` command for data quality reporting

**Files:**
- Modify: `agentic_pipeline/cli.py`
- Create: `tests/test_cli_library_issues.py`

**Context:** The library has data quality issues: empty titles, ISBN-as-titles, duplicates. Add a diagnostic command that reports all issues in one view.

**Step 1: Write failing test**

```python
"""Tests for the library-issues CLI command."""

import sqlite3
import pytest
from click.testing import CliRunner
from agentic_pipeline.cli import main


@pytest.fixture
def db_with_issues(tmp_path):
    """Create a test DB with various data quality issues."""
    db_path = tmp_path / "library.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE books (id TEXT PRIMARY KEY, title TEXT, author TEXT, word_count INTEGER)"
    )
    conn.execute(
        "CREATE TABLE chapters (id TEXT PRIMARY KEY, book_id TEXT, chapter_number INTEGER, title TEXT, word_count INTEGER)"
    )
    # Empty title
    conn.execute("INSERT INTO books VALUES ('b1', '', 'Author A', 1000)")
    # ISBN-as-title
    conn.execute("INSERT INTO books VALUES ('b2', '1394159641', 'Author B', 2000)")
    # Good book
    conn.execute("INSERT INTO books VALUES ('b3', 'Clean Code', 'Martin', 3000)")
    # Duplicate title
    conn.execute("INSERT INTO books VALUES ('b4', 'Clean Code', 'Martin', 3000)")
    conn.commit()
    conn.close()
    return db_path


def test_library_issues_finds_problems(db_with_issues, monkeypatch):
    """library-issues should report empty titles, ISBN titles, and duplicates."""
    monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(db_with_issues))
    runner = CliRunner()
    result = runner.invoke(main, ["library-issues"])
    assert result.exit_code == 0
    # Should report empty titles
    assert "empty" in result.output.lower() or "missing" in result.output.lower()
    # Should report duplicates
    assert "duplicate" in result.output.lower()


def test_library_issues_clean_db(tmp_path, monkeypatch):
    """library-issues should report no issues on a clean DB."""
    db_path = tmp_path / "library.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE books (id TEXT PRIMARY KEY, title TEXT, author TEXT, word_count INTEGER)"
    )
    conn.execute("INSERT INTO books VALUES ('b1', 'Good Book', 'Author', 1000)")
    conn.commit()
    conn.close()
    monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(db_path))
    runner = CliRunner()
    result = runner.invoke(main, ["library-issues"])
    assert result.exit_code == 0
    assert "no issues" in result.output.lower() or "clean" in result.output.lower()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli_library_issues.py -v`
Expected: FAIL — command doesn't exist

**Step 3: Implement the command**

Add to `agentic_pipeline/cli.py`:

```python
@main.command("library-issues")
def library_issues():
    """Report library data quality issues (empty titles, duplicates, etc.)."""
    from .db.config import get_db_path

    import sqlite3

    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    issues = []

    try:
        # Empty or missing titles
        empty = conn.execute(
            "SELECT id, title, author FROM books WHERE title IS NULL OR TRIM(title) = '' OR title = '(Title Missing)'"
        ).fetchall()
        if empty:
            issues.append(("Empty/Missing Titles", empty))
            console.print(f"\n[red]Empty/Missing Titles: {len(empty)}[/red]")
            for b in empty:
                console.print(f"  {b['id'][:12]}  author={b['author'] or 'unknown'}")

        # ISBN-as-title (all digits, optionally with hyphens)
        isbn = conn.execute(
            "SELECT id, title, author FROM books WHERE title GLOB '[0-9]*' AND length(title) <= 20 AND TRIM(title) != ''"
        ).fetchall()
        if isbn:
            issues.append(("ISBN/Number Titles", isbn))
            console.print(f"\n[yellow]ISBN/Number Titles: {len(isbn)}[/yellow]")
            for b in isbn:
                console.print(f"  {b['id'][:12]}  title='{b['title']}'  author={b['author'] or 'unknown'}")

        # Short titles (likely garbage)
        short = conn.execute(
            "SELECT id, title, author FROM books WHERE length(TRIM(title)) BETWEEN 1 AND 3 AND title NOT GLOB '[0-9]*'"
        ).fetchall()
        if short:
            issues.append(("Very Short Titles", short))
            console.print(f"\n[yellow]Very Short Titles (≤3 chars): {len(short)}[/yellow]")
            for b in short:
                console.print(f"  {b['id'][:12]}  title='{b['title']}'")

        # Duplicate titles
        dupes = conn.execute(
            "SELECT title, COUNT(*) as cnt FROM books WHERE TRIM(title) != '' GROUP BY title HAVING cnt > 1 ORDER BY cnt DESC"
        ).fetchall()
        if dupes:
            issues.append(("Duplicate Titles", dupes))
            console.print(f"\n[yellow]Duplicate Titles: {len(dupes)} groups[/yellow]")
            for d in dupes:
                console.print(f"  {d['cnt']}x  '{d['title'][:70]}'")

        if not issues:
            console.print("[green]No issues found — library is clean.[/green]")
        else:
            total = sum(len(items) for _, items in issues)
            console.print(f"\n[bold]{total} issues across {len(issues)} categories[/bold]")
            console.print("Use [bold]update-title[/bold] to fix chapter titles, or manual SQL for book titles.")

    finally:
        conn.close()
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_cli_library_issues.py -v`
Expected: All pass

**Step 5: Run full test suite**

Run: `python -m pytest tests/ -x -q`
Expected: All pass

**Step 6: Commit**

```bash
git add agentic_pipeline/cli.py tests/test_cli_library_issues.py
git commit -m "feat: add library-issues CLI command for data quality reporting"
```

---

### Task 5: Add initial src/ test coverage for learning_tools

**Files:**
- Expand: `tests/test_learning_tools.py` (created in Task 2)

**Context:** `src/tools/learning_tools.py` has complex logic with sentence extraction, vocabulary parsing, and depth-based formatting. Zero test coverage today. Add tests for the key helper functions that don't need a real database.

**Step 1: Add tests for helper functions**

Expand `tests/test_learning_tools.py` with:

```python
def test_split_sentences():
    """_split_sentences should split on sentence boundaries."""
    from src.tools.learning_tools import _split_sentences

    result = _split_sentences("Hello world. This is a test. Another one!")
    assert len(result) >= 2
    assert "Hello world" in result[0]


def test_extract_vocabulary_with_sources():
    """_extract_vocabulary should extract terms from source excerpts."""
    from src.tools.learning_tools import _extract_vocabulary

    sources = [
        {
            "book_title": "Test Book",
            "excerpt": "**Kubernetes** is a container orchestration platform. **Pod** refers to the smallest deployable unit.",
        }
    ]
    result = _extract_vocabulary("kubernetes", sources)
    assert isinstance(result, dict)


def test_extract_vocabulary_empty_sources():
    """_extract_vocabulary should return empty dict for no sources."""
    from src.tools.learning_tools import _extract_vocabulary

    result = _extract_vocabulary("kubernetes", [])
    assert result == {}


def test_extract_related_concepts_empty_sources():
    """_extract_related_concepts should return empty list for no sources."""
    from src.tools.learning_tools import _extract_related_concepts

    result = _extract_related_concepts("kubernetes", [])
    assert result == []


def test_generate_decisions_empty_sources():
    """_generate_decisions should return fallback for no sources."""
    from src.tools.learning_tools import _generate_decisions

    result = _generate_decisions("kubernetes", [])
    assert isinstance(result, str)
    assert len(result) > 0
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_learning_tools.py -v`
Expected: All pass

**Step 3: Commit**

```bash
git add tests/test_learning_tools.py
git commit -m "test: add initial test coverage for learning_tools helpers"
```

---

### Task 6: Add test coverage for analytics_tools

**Files:**
- Create: `tests/test_analytics_tools.py`

**Context:** `src/tools/analytics_tools.py` has complex SQL queries and data aggregation but zero test coverage. Tests need a populated test DB.

**Step 1: Write tests with DB fixture**

```python
"""Tests for analytics_tools."""

import sqlite3
import pytest
from unittest.mock import patch


@pytest.fixture
def analytics_db(tmp_path):
    """Create a test DB with books and chapters for analytics."""
    db_path = tmp_path / "library.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE books (id TEXT PRIMARY KEY, title TEXT, author TEXT, word_count INTEGER, book_type TEXT, source_file TEXT)"
    )
    conn.execute(
        "CREATE TABLE chapters (id TEXT PRIMARY KEY, book_id TEXT, chapter_number INTEGER, title TEXT, word_count INTEGER, file_path TEXT, embedding BLOB)"
    )
    # Two books by different authors
    conn.execute(
        "INSERT INTO books VALUES ('b1', 'Clean Code', 'Robert Martin', 50000, 'programming', 'clean.epub')"
    )
    conn.execute(
        "INSERT INTO books VALUES ('b2', 'Design Patterns', 'GoF', 30000, 'programming', 'patterns.epub')"
    )
    # Chapters
    for i in range(1, 4):
        conn.execute(
            "INSERT INTO chapters VALUES (?, 'b1', ?, ?, ?, ?, NULL)",
            (f"ch-b1-{i}", i, f"Chapter {i}", 5000, f"ch{i}.md"),
        )
    for i in range(1, 3):
        conn.execute(
            "INSERT INTO chapters VALUES (?, 'b2', ?, ?, ?, ?, NULL)",
            (f"ch-b2-{i}", i, f"Pattern {i}", 3000, f"p{i}.md"),
        )
    conn.commit()
    conn.close()
    return db_path


def test_get_library_statistics(analytics_db):
    """get_library_statistics should return comprehensive stats."""
    from src.tools.analytics_tools import get_library_statistics
    from src.database import execute_query
    from src.config import Config

    Config.DB_PATH = analytics_db
    result = get_library_statistics()
    assert isinstance(result, dict)
    assert result["total_books"] == 2
    assert result["total_chapters"] == 5


def test_get_author_insights(analytics_db):
    """get_author_insights should return stats for an author."""
    from src.tools.analytics_tools import get_author_insights
    from src.config import Config

    Config.DB_PATH = analytics_db
    result = get_author_insights(author="Robert Martin")
    assert isinstance(result, dict)
    assert "Robert Martin" in str(result)
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_analytics_tools.py -v`
Expected: All pass (may need Config path adjustment — adapt as needed)

**Step 3: Commit**

```bash
git add tests/test_analytics_tools.py
git commit -m "test: add initial test coverage for analytics_tools"
```

---

## Execution Order

Tasks 1-2 are quick bug fixes (10 min). Tasks 3-4 are the CLI additions (20 min). Tasks 5-6 are test coverage (15 min). All are independent and can be parallelized.

Total: ~6 commits, ~45 min estimated.

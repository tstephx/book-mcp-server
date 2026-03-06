---
name: new-cli-command
description: Scaffold a new CLI command with tests and ref doc update. Use when adding a new Click command to agentic-pipeline.
disable-model-invocation: true
---

# New CLI Command

Arguments: `<command_name>` `<description>`

## Architecture

All CLI commands live in `agentic_pipeline/cli.py` using Click. Commands are added to the `main` group. They follow a consistent pattern: lazy imports, `get_db_path()`, `sqlite3.connect` with `row_factory=sqlite3.Row`, `try/finally` for cleanup, `console.print` for rich output.

Current command count: check `ref/cli-commands.md` for the authoritative list.

## Steps

### 1. Read the insertion point

Read the end of `agentic_pipeline/cli.py` to find where to insert (before `if __name__ == "__main__":`).

### 2. Write the command

Insert before the `__main__` guard:

```python
@main.command("<command-name>")
@click.argument("arg_name")  # or @click.option("--flag")
def command_name(arg_name: str):
    """Command description shown in --help.

    ARG_NAME is the description of the argument.
    """
    from .db.config import get_db_path

    import sqlite3

    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        # Implementation here
        console.print("[green]Success message[/green]")
    finally:
        conn.close()
```

Key patterns:
- Lazy imports inside function body (keeps CLI startup fast)
- Fuzzy book matching: query by UUID first, fallback to `LIKE %term%` with ambiguity check (see `update-title` for reference)
- Always parameterized SQL (no f-strings in queries)
- Rich markup: `[green]` success, `[red]` errors, `[yellow]` warnings

### 3. Write tests

Create `tests/test_cli_<command_name>.py`:

```python
"""Tests for the <command-name> CLI command."""

import sqlite3
import pytest
from click.testing import CliRunner
from agentic_pipeline.cli import main


@pytest.fixture
def db_with_data(tmp_path):
    """Create a test DB with relevant data."""
    db_path = tmp_path / "library.db"
    conn = sqlite3.connect(db_path)
    # Create required tables (only the columns your command uses)
    conn.execute("CREATE TABLE books (id TEXT PRIMARY KEY, title TEXT, author TEXT, word_count INTEGER)")
    conn.execute("CREATE TABLE chapters (id TEXT PRIMARY KEY, book_id TEXT, chapter_number INTEGER, title TEXT, word_count INTEGER)")
    # Insert test data
    conn.execute("INSERT INTO books VALUES ('b1', 'Test Book', 'Author', 1000)")
    conn.commit()
    conn.close()
    return db_path


def test_<command_name>_happy_path(db_with_data, monkeypatch):
    """<command-name> should succeed with valid input."""
    monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(db_with_data))
    runner = CliRunner()
    result = runner.invoke(main, ["<command-name>", "arg1"])
    assert result.exit_code == 0
    assert "expected output" in result.output.lower()


def test_<command_name>_error_case(db_with_data, monkeypatch):
    """<command-name> should handle errors gracefully."""
    monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(db_with_data))
    runner = CliRunner()
    result = runner.invoke(main, ["<command-name>", "invalid"])
    assert result.exit_code == 0
    assert "not found" in result.output.lower() or "error" in result.output.lower()
```

Minimum 2 tests: happy path + error case.

### 4. Run tests

```bash
.venv/bin/python -m pytest tests/test_cli_<command_name>.py -v
.venv/bin/python -m pytest tests/ -x -q  # full suite
```

### 5. Update ref docs

Add the new command to `ref/cli-commands.md` in the appropriate category table.
Update the command count in `ref/module-map.md` if needed.

### 6. Commit

```bash
git add agentic_pipeline/cli.py tests/test_cli_<command_name>.py ref/cli-commands.md
git commit -m "feat: add <command-name> CLI command"
```

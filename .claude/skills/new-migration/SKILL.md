---
name: new-migration
description: Add a new database migration to agentic_pipeline/db/migrations.py with a matching test.
disable-model-invocation: true
---

# New Migration

Arguments: `<description of schema change>`

## Architecture

Migrations live in `agentic_pipeline/db/migrations.py`. Two patterns exist:

1. **CREATE TABLE** — Append to the `MIGRATIONS` list. These are idempotent via `IF NOT EXISTS` and run every startup.
2. **ALTER TABLE** — Add a versioned block after the MIGRATIONS loop in `run_migrations()`. Uses `_migration_applied()` / `_record_migration()` to run exactly once. Always check column existence before ALTER.

New **indexes** go in the `INDEXES` list.

## Steps

### 1. Read current state

Read `agentic_pipeline/db/migrations.py` to understand existing tables and the last migration. Read `ref/db-schema.md` for the documented schema.

### 2. Determine migration type

- **New table** → Append CREATE TABLE to `MIGRATIONS` list
- **New column on existing table** → Add versioned ALTER TABLE block
- **New index** → Append to `INDEXES` list

### 3. Write the migration

For a **new table**, append to `MIGRATIONS`:
```python
# <Description>
"""
CREATE TABLE IF NOT EXISTS <table_name> (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ...
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""",
```

For an **ALTER TABLE**, add after the existing versioned migrations in `run_migrations()`:
```python
if not _migration_applied(cursor, "<migration_name>"):
    cursor.execute("PRAGMA table_info(<table_name>)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    if "<new_column>" not in existing_columns:
        cursor.execute(
            "ALTER TABLE <table_name> ADD COLUMN <new_column> <TYPE> <DEFAULT>"
        )
    _record_migration(cursor, "<migration_name>")
```

### 4. Write test

Add to or create a test file (`tests/test_migrations.py` or `tests/test_<feature>_migrations.py`):

```python
def test_<migration_name>_creates_table():
    """Migration creates <table> with expected columns."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    try:
        run_migrations(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(<table_name>)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "<expected_column>" in columns
        conn.close()
    finally:
        db_path.unlink(missing_ok=True)
```

### 5. Update ref doc

Update `ref/db-schema.md` with the new table or column.

### 6. Verify

```bash
python -m pytest tests/test_migrations.py -v
```

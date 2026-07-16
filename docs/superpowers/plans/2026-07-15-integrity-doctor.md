# Integrity Doctor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `agentic-pipeline doctor` detects four classes of library/pipeline drift and `doctor --fix` repairs them — backup, manifest, delete orphans, archive dead pipelines, repoint sources, backfill hashes and book types — with the same checks wired into `health` and a read-only `doctor_report` MCP tool.

**Architecture:** One new module `agentic_pipeline/health/doctor.py` holds pure check functions (reads) and fix functions (writes) sharing SQL fragments so detector and repairer cannot drift apart. A thin Click command wires report/`--fix` modes with exit codes; `health` calls `run_checks()` inline; MCP exposes capped-payload reporting only.

**Tech Stack:** Python 3.12, sqlite3 (WAL, via `get_pipeline_db`), Click + rich (CLI), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-15-integrity-doctor-design.md` (15 numbered decisions — the plan implements all of them).

## Global Constraints

- Branch: `feat/integrity-doctor` (already created; spec commits on it).
- TDD: every step pair is write-failing-test → verify RED → implement → verify GREEN. Never skip the RED run.
- Every check MUST be provably able to fail: each has a seeded-violation test.
- Tests use a temp DB via `run_migrations(tmp_path / "doctor.db")` — never the live library DB.
- DB connections in product code go through `get_pipeline_db()` (project rule), except raw `sqlite3.connect` inside test seed helpers.
- The venv must be active: `source .venv/bin/activate`. Run tests with `python -m pytest`.
- A PostToolUse ruff hook strips imports that are unused *at edit time* — when adding an import for code you haven't written yet, add import and usage in the same edit, and re-check the import line if a later test hits `NameError`.
- Commit messages: prefix `feat:`/`test:`; end body with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Backup filename pattern is `<db-name>.backup-doctor-<YYYYMMDD-HHMMSS>` — the `-doctor-` marker is deliberate (spec Decision 8 prunes "doctor backups" only; an unmarked pattern could match a user's manual backups).

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `agentic_pipeline/health/doctor.py` | Create | `Finding`/`FixReport` dataclasses, 4 checks, `run_checks`, `has_violations`, backup/rotate, manifest, `apply_fixes` |
| `agentic_pipeline/health/__init__.py` | Modify | Export the public surface |
| `agentic_pipeline/cli.py` | Modify | `doctor` command (exit codes); integrity line in `health` |
| `agentic_pipeline/mcp_server.py` | Modify | `doctor_report()` (capped payload) |
| `agentic_mcp_server.py` | Modify | Register `doctor_report` in the wrapper |
| `tests/test_doctor.py` | Create | All doctor unit/CLI tests + seed helpers |
| `tests/test_mcp_server_tools.py` | Modify | Wrapper-registration assertion for `doctor_report` |

---

### Task 1: Scaffolding + orphaned-chunks check

**Files:**
- Create: `agentic_pipeline/health/doctor.py`
- Modify: `agentic_pipeline/health/__init__.py`
- Create: `tests/test_doctor.py`

**Interfaces:**
- Produces: `Finding(category: str, count: int, fixable_count: int, details: list[dict])`; category constants `CATEGORY_ORPHANED_CHUNKS/_LOST_BOOKS/_NULL_CONTENT_HASH/_NULL_BOOK_TYPE`; tuple `CATEGORIES`; `check_orphaned_chunks(db_path) -> Finding`; module-level SQL fragment `_ORPHAN_WHERE` (reused by the delete fix in Task 6). Test seed helpers `_seed_book/_seed_chapter/_seed_chunk` and fixture `db_path` used by every later task.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_doctor.py`:

```python
"""Tests for the integrity doctor."""

import sqlite3

import pytest

from agentic_pipeline.db.migrations import run_migrations


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "doctor.db"
    run_migrations(path)
    return path


# ---------------------------------------------------------------------------
# Seed helpers — raw sqlite3 is fine in tests; ids are TEXT PKs in all tables.
# ---------------------------------------------------------------------------


def _connect(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _seed_book(conn, book_id="b1", title="Live Book", book_type=None):
    conn.execute(
        "INSERT INTO books (id, title, book_type) VALUES (?, ?, ?)",
        (book_id, title, book_type),
    )


def _seed_chapter(conn, chapter_id="c1", book_id="b1", file_path="/nope.md",
                  content_hash="hash", number=1):
    conn.execute(
        """INSERT INTO chapters (id, book_id, chapter_number, title, file_path,
                                 word_count, content_hash)
           VALUES (?, ?, ?, ?, ?, 100, ?)""",
        (chapter_id, book_id, number, f"Chapter {number}", file_path, content_hash),
    )


def _seed_chunk(conn, chunk_id="k1", chapter_id="c1", book_id="b1",
                content="chunk text body", embedded=True):
    conn.execute(
        """INSERT INTO chunks (id, chapter_id, book_id, chunk_index, content,
                               word_count, embedding, embedding_model, content_hash)
           VALUES (?, ?, ?, 0, ?, 3, ?, 'test-model', 'kh')""",
        (chunk_id, chapter_id, book_id, content, b"\x00\x01" if embedded else None),
    )


class TestCheckOrphanedChunks:
    def test_flags_chunk_with_dead_chapter_id(self, db_path):
        """Seeded violation — the check MUST be able to fail."""
        from agentic_pipeline.health.doctor import (
            CATEGORY_ORPHANED_CHUNKS,
            check_orphaned_chunks,
        )

        conn = _connect(db_path)
        _seed_book(conn)
        _seed_chapter(conn)
        _seed_chunk(conn, chunk_id="ok", chapter_id="c1")          # healthy
        _seed_chunk(conn, chunk_id="orphan", chapter_id="GONE")    # dead chapter
        conn.commit(); conn.close()

        finding = check_orphaned_chunks(db_path)

        assert finding.category == CATEGORY_ORPHANED_CHUNKS
        assert finding.count == 1
        assert finding.fixable_count == 1
        assert finding.details[0]["chunk_id"] == "orphan"

    def test_flags_chunk_with_dead_book_id(self, db_path):
        from agentic_pipeline.health.doctor import check_orphaned_chunks

        conn = _connect(db_path)
        _seed_book(conn)
        _seed_chapter(conn)
        _seed_chunk(conn, chunk_id="orphan2", book_id="GONE-BOOK")
        conn.commit(); conn.close()

        assert check_orphaned_chunks(db_path).count == 1

    def test_counts_unembedded_orphans_identically(self, db_path):
        """Spec: doctor treats embedded and never-embedded orphans identically."""
        from agentic_pipeline.health.doctor import check_orphaned_chunks

        conn = _connect(db_path)
        _seed_chunk(conn, chunk_id="o1", chapter_id="GONE", book_id="GONE", embedded=True)
        _seed_chunk(conn, chunk_id="o2", chapter_id="GONE", book_id="GONE", embedded=False)
        conn.commit(); conn.close()

        assert check_orphaned_chunks(db_path).count == 2

    def test_clean_db_passes(self, db_path):
        from agentic_pipeline.health.doctor import check_orphaned_chunks

        conn = _connect(db_path)
        _seed_book(conn); _seed_chapter(conn); _seed_chunk(conn)
        conn.commit(); conn.close()

        finding = check_orphaned_chunks(db_path)
        assert finding.count == 0
        assert finding.details == []

    def test_finding_field_types(self, db_path):
        """Contract rule: assert types, not just presence."""
        from agentic_pipeline.health.doctor import check_orphaned_chunks

        finding = check_orphaned_chunks(db_path)
        assert isinstance(finding.category, str)
        assert isinstance(finding.count, int)
        assert isinstance(finding.fixable_count, int)
        assert isinstance(finding.details, list)
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `python -m pytest tests/test_doctor.py -q`
Expected: collection error `ModuleNotFoundError`/`ImportError` naming `agentic_pipeline.health.doctor` — the feature is missing, the right reason.

- [ ] **Step 3: Implement**

Create `agentic_pipeline/health/doctor.py`:

```python
"""Integrity doctor — detect and repair library/pipeline drift.

Every check is a query that CAN return violations; the fixes reuse the
checks' SQL fragments so detector and repairer cannot drift apart.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from agentic_pipeline.db.connection import get_pipeline_db

logger = logging.getLogger(__name__)

CATEGORY_ORPHANED_CHUNKS = "orphaned_chunks"
CATEGORY_LOST_BOOKS = "lost_books"
CATEGORY_NULL_CONTENT_HASH = "null_content_hash"
CATEGORY_NULL_BOOK_TYPE = "null_book_type"

CATEGORIES = (
    CATEGORY_ORPHANED_CHUNKS,
    CATEGORY_LOST_BOOKS,
    CATEGORY_NULL_CONTENT_HASH,
    CATEGORY_NULL_BOOK_TYPE,
)

# Shared by check_orphaned_chunks and the delete fix — one definition of
# "orphan" so the two cannot disagree. Embedded or not is irrelevant:
# unjoinable rows are dead weight either way.
_ORPHAN_WHERE = """
    NOT EXISTS (SELECT 1 FROM chapters ch WHERE ch.id = chunks.chapter_id)
    OR NOT EXISTS (SELECT 1 FROM books b WHERE b.id = chunks.book_id)
"""


@dataclass
class Finding:
    """One category of integrity violation."""

    category: str
    count: int
    fixable_count: int
    details: list[dict] = field(default_factory=list)


def check_orphaned_chunks(db_path) -> Finding:
    """Chunks whose chapter_id or book_id resolves to nothing."""
    with get_pipeline_db(str(db_path)) as conn:
        rows = conn.execute(
            f"""SELECT id AS chunk_id, chapter_id, book_id
                FROM chunks WHERE {_ORPHAN_WHERE}"""
        ).fetchall()
    details = [dict(r) for r in rows]
    return Finding(
        category=CATEGORY_ORPHANED_CHUNKS,
        count=len(details),
        fixable_count=len(details),  # deletion fixes every orphan
        details=details,
    )
```

Modify `agentic_pipeline/health/__init__.py` (add import and usage in the same edit — the ruff hook strips imports that look unused):

```python
"""Health monitoring package."""

from agentic_pipeline.health.monitor import HealthMonitor
from agentic_pipeline.health.stuck_detector import StuckDetector, DEFAULT_STATE_TIMEOUTS
from agentic_pipeline.health.doctor import (
    CATEGORIES,
    Finding,
    check_orphaned_chunks,
)

__all__ = [
    "HealthMonitor",
    "StuckDetector",
    "DEFAULT_STATE_TIMEOUTS",
    "CATEGORIES",
    "Finding",
    "check_orphaned_chunks",
]
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `python -m pytest tests/test_doctor.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add agentic_pipeline/health/doctor.py agentic_pipeline/health/__init__.py tests/test_doctor.py
git commit -m "feat: doctor scaffolding + orphaned-chunks check

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Lost-books check

**Files:**
- Modify: `agentic_pipeline/health/doctor.py`
- Modify: `tests/test_doctor.py`

**Interfaces:**
- Consumes: `Finding`, seed helpers, `db_path` fixture (Task 1).
- Produces: `check_lost_books(db_path) -> Finding` whose `details` dicts carry exactly: `pipeline_id` (str), `source_path` (str), `basename` (str), `chunk_count` (int), `source_available` (bool), `resolved_path` (str|None), `live_copy` (bool), `sample` (str, ≤200 chars, may be ""). Task 5's manifest and Task 6's archive/repoint consume these keys verbatim.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_doctor.py`:

```python
def _seed_complete_pipeline(db_path, pipeline_id=None, source_path="/watch/lost.epub"):
    """Create a pipeline and walk it to COMPLETE through valid transitions."""
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)
    pid = repo.create(source_path, f"hash-{pipeline_id or 'x'}")
    for state in (
        PipelineState.HASHING, PipelineState.CLASSIFYING,
        PipelineState.SELECTING_STRATEGY, PipelineState.PROCESSING,
        PipelineState.VALIDATING, PipelineState.PENDING_APPROVAL,
        PipelineState.APPROVED, PipelineState.EMBEDDING, PipelineState.COMPLETE,
    ):
        repo.update_state(pid, state)
    return pid


class TestCheckLostBooks:
    def test_flags_complete_pipeline_without_book_row(self, db_path, tmp_path):
        from agentic_pipeline.health.doctor import CATEGORY_LOST_BOOKS, check_lost_books

        src = tmp_path / "lost.epub"
        src.write_bytes(b"epub bytes")
        pid = _seed_complete_pipeline(db_path, source_path=str(src))
        conn = _connect(db_path)
        _seed_chunk(conn, chunk_id="lost-k", chapter_id="GONE", book_id=pid,
                    content="a sample of the lost book's text " * 20)
        conn.commit(); conn.close()

        finding = check_lost_books(db_path)

        assert finding.category == CATEGORY_LOST_BOOKS
        assert finding.count == 1
        d = finding.details[0]
        assert d["pipeline_id"] == pid
        assert d["basename"] == "lost.epub"
        assert d["chunk_count"] == 1
        assert d["source_available"] is True
        assert d["resolved_path"] == str(src)
        assert d["live_copy"] is False
        assert 0 < len(d["sample"]) <= 200

    def test_source_gone_book_is_flagged_unavailable(self, db_path):
        from agentic_pipeline.health.doctor import check_lost_books

        _seed_complete_pipeline(db_path, source_path="/nowhere/gone.epub")

        d = check_lost_books(db_path).details[0]
        assert d["source_available"] is False
        assert d["resolved_path"] is None
        assert d["sample"] == ""  # no chunks seeded

    def test_resolves_source_via_processed_dir(self, db_path, tmp_path, monkeypatch):
        """A moved file is found at PROCESSED_DIR/<basename>."""
        from agentic_pipeline.health.doctor import check_lost_books

        processed = tmp_path / "processed"
        processed.mkdir()
        (processed / "moved.epub").write_bytes(b"x")
        monkeypatch.setenv("PROCESSED_DIR", str(processed))
        _seed_complete_pipeline(db_path, source_path="/watch/moved.epub")

        d = check_lost_books(db_path).details[0]
        assert d["source_available"] is True
        assert d["resolved_path"] == str(processed / "moved.epub")

    def test_complete_pipeline_with_book_row_is_healthy(self, db_path):
        from agentic_pipeline.health.doctor import check_lost_books

        pid = _seed_complete_pipeline(db_path)
        conn = _connect(db_path)
        _seed_book(conn, book_id=pid)
        conn.commit(); conn.close()

        assert check_lost_books(db_path).count == 0

    def test_fixable_count_equals_count(self, db_path):
        """Archiving fixes every lost book, source or no source."""
        from agentic_pipeline.health.doctor import check_lost_books

        _seed_complete_pipeline(db_path, source_path="/nowhere/gone.epub")

        finding = check_lost_books(db_path)
        assert finding.fixable_count == finding.count == 1
```

- [ ] **Step 2: Run — verify RED**

Run: `python -m pytest tests/test_doctor.py -q -k CheckLostBooks`
Expected: `ImportError: cannot import name 'check_lost_books'`

- [ ] **Step 3: Implement**

Append to `agentic_pipeline/health/doctor.py` (also extend the module's imports: `import os` at top with the others):

```python
def _resolve_lost_source(source_path: str) -> str | None:
    """Original path, else PROCESSED_DIR/<basename>, else None.

    Basename matching is acceptable HERE (unlike resolve_source_file's
    hash-verified fallback for live books): a lost book has no live copy
    to corrupt — reingest mints a fresh record from whatever file exists.
    """
    if source_path and Path(source_path).exists():
        return source_path
    processed = os.environ.get("PROCESSED_DIR")
    if source_path and processed:
        candidate = Path(processed) / Path(source_path).name
        if candidate.exists():
            return str(candidate)
    return None


def check_lost_books(db_path) -> Finding:
    """Pipelines claiming COMPLETE while the library has no such book."""
    with get_pipeline_db(str(db_path)) as conn:
        rows = conn.execute(
            """SELECT p.id, p.source_path FROM processing_pipelines p
               WHERE p.state = 'complete'
                 AND NOT EXISTS (SELECT 1 FROM books b WHERE b.id = p.id)"""
        ).fetchall()
        details = []
        for r in rows:
            source_path = r["source_path"] or ""
            basename = Path(source_path).name if source_path else ""
            chunk_count = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE book_id = ?", (r["id"],)
            ).fetchone()[0]
            sample_row = conn.execute(
                "SELECT content FROM chunks WHERE book_id = ? ORDER BY chunk_index LIMIT 1",
                (r["id"],),
            ).fetchone()
            sample = (sample_row["content"] or "")[:200] if sample_row else ""
            live_copy = bool(
                basename
                and conn.execute(
                    """SELECT 1 FROM processing_pipelines p2
                       JOIN books b ON b.id = p2.id
                       WHERE p2.source_path LIKE ? AND p2.id != ? LIMIT 1""",
                    (f"%{basename}", r["id"]),
                ).fetchone()
            )
            resolved = _resolve_lost_source(source_path)
            details.append(
                {
                    "pipeline_id": r["id"],
                    "source_path": source_path,
                    "basename": basename,
                    "chunk_count": chunk_count,
                    "source_available": resolved is not None,
                    "resolved_path": resolved,
                    "live_copy": live_copy,
                    "sample": sample,
                }
            )
    return Finding(
        category=CATEGORY_LOST_BOOKS,
        count=len(details),
        fixable_count=len(details),  # archiving fixes every lost book
        details=details,
    )
```

- [ ] **Step 4: Run — verify GREEN**

Run: `python -m pytest tests/test_doctor.py -q`
Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add agentic_pipeline/health/doctor.py tests/test_doctor.py
git commit -m "feat: doctor lost-books check with source resolution

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Hash/type checks, `run_checks`, `has_violations`

**Files:**
- Modify: `agentic_pipeline/health/doctor.py`
- Modify: `agentic_pipeline/health/__init__.py`
- Modify: `tests/test_doctor.py`

**Interfaces:**
- Consumes: Tasks 1–2 surfaces.
- Produces: `check_null_content_hash(db_path) -> Finding` (details keys: `chapter_id`, `file_path`, `file_exists: bool`); `check_null_book_type(db_path) -> Finding` (details keys: `book_id`, `title`, `profile_book_type: str|None`, `profile_confidence: float|None`, `valid: bool`); `run_checks(db_path) -> list[Finding]` (always exactly 4, ordered as `CATEGORIES`); `has_violations(findings) -> bool`. `_is_valid_book_type(value) -> bool` (enum member, not `unknown`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_doctor.py`:

```python
import json


class TestCheckNullContentHash:
    def test_flags_null_hash_and_reports_file_availability(self, db_path, tmp_path):
        from agentic_pipeline.health.doctor import check_null_content_hash

        real = tmp_path / "ch.md"
        real.write_text("chapter body")
        conn = _connect(db_path)
        _seed_book(conn)
        _seed_chapter(conn, chapter_id="fixable", content_hash=None, file_path=str(real))
        _seed_chapter(conn, chapter_id="unfixable", content_hash=None,
                      file_path="/nope.md", number=2)
        _seed_chapter(conn, chapter_id="healthy", content_hash="abc", number=3)
        conn.commit(); conn.close()

        finding = check_null_content_hash(db_path)

        assert finding.count == 2
        assert finding.fixable_count == 1  # only the one whose file exists
        by_id = {d["chapter_id"]: d for d in finding.details}
        assert by_id["fixable"]["file_exists"] is True
        assert by_id["unfixable"]["file_exists"] is False

    def test_clean_db_passes(self, db_path):
        from agentic_pipeline.health.doctor import check_null_content_hash

        assert check_null_content_hash(db_path).count == 0


class TestCheckNullBookType:
    def _seed(self, db_path, profile):
        """Book with NULL book_type whose pipeline carries the given profile."""
        pid = _seed_complete_pipeline(db_path, source_path="/watch/typed.epub")
        conn = _connect(db_path)
        _seed_book(conn, book_id=pid, book_type=None)
        if profile is not None:
            conn.execute(
                "UPDATE processing_pipelines SET book_profile = ? WHERE id = ?",
                (json.dumps(profile), pid),
            )
        conn.commit(); conn.close()
        return pid

    def test_valid_enum_value_is_fixable(self, db_path):
        from agentic_pipeline.health.doctor import check_null_book_type

        self._seed(db_path, {"book_type": "travel_guide", "confidence": 0.9})

        finding = check_null_book_type(db_path)
        assert finding.count == 1
        assert finding.fixable_count == 1
        assert finding.details[0]["valid"] is True
        assert finding.details[0]["profile_confidence"] == 0.9

    def test_unknown_is_not_an_answer(self, db_path):
        """Spec Decision 11: writing 'unknown' as an answer is not an answer."""
        from agentic_pipeline.health.doctor import check_null_book_type

        self._seed(db_path, {"book_type": "unknown", "confidence": 0.0})

        finding = check_null_book_type(db_path)
        assert finding.count == 1
        assert finding.fixable_count == 0
        assert finding.details[0]["valid"] is False

    def test_off_enum_string_is_not_fixable(self, db_path):
        from agentic_pipeline.health.doctor import check_null_book_type

        self._seed(db_path, {"book_type": "guidebook", "confidence": 0.8})

        assert check_null_book_type(db_path).fixable_count == 0

    def test_missing_profile_is_reported_not_guessed(self, db_path):
        from agentic_pipeline.health.doctor import check_null_book_type

        self._seed(db_path, None)

        finding = check_null_book_type(db_path)
        assert finding.count == 1
        assert finding.fixable_count == 0
        assert finding.details[0]["profile_book_type"] is None

    def test_every_book_type_enum_value_judged_correctly(self, db_path):
        """Contract rule: enumeration completeness — all enum members handled."""
        from agentic_pipeline.agents.classifier_types import BookType
        from agentic_pipeline.health.doctor import _is_valid_book_type

        for member in BookType:
            expected = member is not BookType.UNKNOWN
            assert _is_valid_book_type(member.value) is expected, member
        assert _is_valid_book_type("guidebook") is False
        assert _is_valid_book_type(None) is False


class TestRunChecks:
    def test_returns_all_four_categories_in_order(self, db_path):
        from agentic_pipeline.health.doctor import CATEGORIES, run_checks

        findings = run_checks(db_path)

        assert [f.category for f in findings] == list(CATEGORIES)

    def test_has_violations(self, db_path):
        from agentic_pipeline.health.doctor import has_violations, run_checks

        assert has_violations(run_checks(db_path)) is False
        conn = _connect(db_path)
        _seed_chunk(conn, chunk_id="o", chapter_id="GONE", book_id="GONE")
        conn.commit(); conn.close()
        assert has_violations(run_checks(db_path)) is True
```

- [ ] **Step 2: Run — verify RED**

Run: `python -m pytest tests/test_doctor.py -q -k "NullContentHash or NullBookType or RunChecks"`
Expected: ImportErrors for the new names.

- [ ] **Step 3: Implement**

Append to `agentic_pipeline/health/doctor.py` (extend top imports with `import json` and `from agentic_pipeline.agents.classifier_types import BookType` in the same edit as their uses):

```python
def check_null_content_hash(db_path) -> Finding:
    """Chapters whose content_hash is NULL — the duplicate check skips them."""
    with get_pipeline_db(str(db_path)) as conn:
        rows = conn.execute(
            """SELECT id AS chapter_id, file_path FROM chapters
               WHERE content_hash IS NULL OR content_hash = ''"""
        ).fetchall()
    details = []
    for r in rows:
        file_exists = bool(r["file_path"]) and Path(r["file_path"]).is_file()
        details.append(
            {"chapter_id": r["chapter_id"], "file_path": r["file_path"],
             "file_exists": file_exists}
        )
    return Finding(
        category=CATEGORY_NULL_CONTENT_HASH,
        count=len(details),
        fixable_count=sum(1 for d in details if d["file_exists"]),
        details=details,
    )


def _is_valid_book_type(value) -> bool:
    """A real BookType member, and not 'unknown' — Decision 11."""
    if not isinstance(value, str):
        return False
    return value in {m.value for m in BookType} and value != BookType.UNKNOWN.value


def check_null_book_type(db_path) -> Finding:
    """Books with NULL book_type; fixable when the pipeline profile has a valid type."""
    with get_pipeline_db(str(db_path)) as conn:
        rows = conn.execute(
            """SELECT b.id AS book_id, b.title, p.book_profile
               FROM books b
               LEFT JOIN processing_pipelines p ON p.id = b.id
               WHERE b.book_type IS NULL"""
        ).fetchall()
    details = []
    for r in rows:
        profile = {}
        if r["book_profile"]:
            try:
                profile = json.loads(r["book_profile"])
            except (json.JSONDecodeError, TypeError):
                profile = {}
        ptype = profile.get("book_type")
        details.append(
            {
                "book_id": r["book_id"],
                "title": r["title"],
                "profile_book_type": ptype,
                "profile_confidence": profile.get("confidence"),
                "valid": _is_valid_book_type(ptype),
            }
        )
    return Finding(
        category=CATEGORY_NULL_BOOK_TYPE,
        count=len(details),
        fixable_count=sum(1 for d in details if d["valid"]),
        details=details,
    )


def run_checks(db_path) -> list[Finding]:
    """All four checks, in CATEGORIES order. Pure reads."""
    return [
        check_orphaned_chunks(db_path),
        check_lost_books(db_path),
        check_null_content_hash(db_path),
        check_null_book_type(db_path),
    ]


def has_violations(findings: list[Finding]) -> bool:
    return any(f.count > 0 for f in findings)
```

Update `agentic_pipeline/health/__init__.py` exports to add: `check_lost_books`, `check_null_content_hash`, `check_null_book_type`, `run_checks`, `has_violations` (import and `__all__` in one edit).

- [ ] **Step 4: Run — verify GREEN**

Run: `python -m pytest tests/test_doctor.py -q`
Expected: `20 passed`

- [ ] **Step 5: Commit**

```bash
git add agentic_pipeline/health/doctor.py agentic_pipeline/health/__init__.py tests/test_doctor.py
git commit -m "feat: doctor hash/type checks, run_checks, has_violations

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Backup with keep-2 rotation

**Files:**
- Modify: `agentic_pipeline/health/doctor.py`
- Modify: `tests/test_doctor.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `create_backup(db_path) -> Path` (WAL-safe copy named `<db-name>.backup-doctor-<YYYYMMDD-HHMMSS>` beside the DB, prunes doctor backups beyond newest 2, never touches other files). Task 8 calls it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_doctor.py`:

```python
import time


class TestBackup:
    def test_creates_walsafe_copy_with_doctor_pattern(self, db_path):
        from agentic_pipeline.health.doctor import create_backup

        backup = create_backup(db_path)

        assert backup.exists()
        assert backup.name.startswith(db_path.name + ".backup-doctor-")
        # The copy is a valid database with the same tables
        conn = sqlite3.connect(backup)
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
        conn.close()

    def test_keeps_only_newest_two(self, db_path):
        from agentic_pipeline.health.doctor import create_backup

        made = []
        for _ in range(3):
            made.append(create_backup(db_path))
            time.sleep(1.1)  # timestamp resolution is seconds

        survivors = sorted(db_path.parent.glob(f"{db_path.name}.backup-doctor-*"))
        assert len(survivors) == 2
        assert made[0] not in survivors          # oldest pruned
        assert made[2] in survivors              # newest kept

    def test_never_touches_non_doctor_files(self, db_path):
        from agentic_pipeline.health.doctor import create_backup

        bystander = db_path.parent / f"{db_path.name}.backup-manual"
        bystander.write_bytes(b"precious user backup")
        for _ in range(3):
            create_backup(db_path)
            time.sleep(1.1)

        assert bystander.exists()
        assert bystander.read_bytes() == b"precious user backup"
```

- [ ] **Step 2: Run — verify RED**

Run: `python -m pytest tests/test_doctor.py -q -k TestBackup`
Expected: `ImportError: cannot import name 'create_backup'`

- [ ] **Step 3: Implement**

Append to `agentic_pipeline/health/doctor.py` (add `import sqlite3` and `from datetime import datetime, timezone` at top in the same edit):

```python
_BACKUP_KEEP = 2


def create_backup(db_path) -> Path:
    """WAL-safe backup beside the DB; prune doctor backups beyond newest 2.

    The filename carries `-doctor-` deliberately: pruning pattern-matches on
    it, so a user's manual `<db>.backup-*` files are never candidates.
    """
    db_path = Path(db_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    dest = db_path.parent / f"{db_path.name}.backup-doctor-{stamp}"

    src = sqlite3.connect(str(db_path))
    dst = sqlite3.connect(str(dest))
    try:
        with dst:
            src.backup(dst)
    finally:
        src.close()
        dst.close()

    backups = sorted(db_path.parent.glob(f"{db_path.name}.backup-doctor-*"))
    for old in backups[:-_BACKUP_KEEP]:
        old.unlink()
        logger.info("Pruned old doctor backup: %s", old.name)

    return dest
```

- [ ] **Step 4: Run — verify GREEN**

Run: `python -m pytest tests/test_doctor.py -q -k TestBackup`
Expected: `3 passed` (suite total 23)

- [ ] **Step 5: Commit**

```bash
git add agentic_pipeline/health/doctor.py tests/test_doctor.py
git commit -m "feat: doctor WAL-safe backup with keep-2 rotation

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Manifest writer

**Files:**
- Modify: `agentic_pipeline/health/doctor.py`
- Modify: `tests/test_doctor.py`

**Interfaces:**
- Consumes: `Finding` for `lost_books` (Task 2 detail keys).
- Produces: `write_manifest(db_path, lost_books: Finding, manifest_path=None) -> Path | None` — default `<db-dir>/doctor/manifest-<YYYYMMDD-HHMMSS>.md`; returns None when there are no lost books. Task 8 calls it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_doctor.py`:

```python
class TestManifest:
    def _lost_finding(self):
        from agentic_pipeline.health.doctor import CATEGORY_LOST_BOOKS, Finding

        return Finding(
            category=CATEGORY_LOST_BOOKS, count=2, fixable_count=2,
            details=[
                {"pipeline_id": "p-1", "source_path": "/w/a.epub", "basename": "a.epub",
                 "chunk_count": 10, "source_available": True,
                 "resolved_path": "/proc/a.epub", "live_copy": False,
                 "sample": "sample text from book a"},
                {"pipeline_id": "p-2", "source_path": "/w/b.epub", "basename": "b.epub",
                 "chunk_count": 5, "source_available": False,
                 "resolved_path": None, "live_copy": False,
                 "sample": "sample text from book b"},
            ],
        )

    def test_writes_default_location_and_splits_sections(self, db_path):
        from agentic_pipeline.health.doctor import write_manifest

        path = write_manifest(db_path, self._lost_finding())

        assert path.parent == db_path.parent / "doctor"
        assert path.name.startswith("manifest-")
        text = path.read_text()
        # re-ingestable and source-gone are separated; samples present
        assert "a.epub" in text and "b.epub" in text
        assert "Re-ingestable" in text and "Source gone" in text
        assert "sample text from book a" in text
        assert text.index("a.epub") < text.index("b.epub") or "Source gone" in text

    def test_explicit_path_override(self, db_path, tmp_path):
        from agentic_pipeline.health.doctor import write_manifest

        target = tmp_path / "custom-manifest.md"
        path = write_manifest(db_path, self._lost_finding(), manifest_path=target)

        assert path == target
        assert target.exists()

    def test_no_lost_books_writes_nothing(self, db_path):
        from agentic_pipeline.health.doctor import CATEGORY_LOST_BOOKS, Finding, write_manifest

        empty = Finding(category=CATEGORY_LOST_BOOKS, count=0, fixable_count=0)
        assert write_manifest(db_path, empty) is None
        assert not (db_path.parent / "doctor").exists()
```

- [ ] **Step 2: Run — verify RED**

Run: `python -m pytest tests/test_doctor.py -q -k TestManifest`
Expected: `ImportError: cannot import name 'write_manifest'`

- [ ] **Step 3: Implement**

Append to `agentic_pipeline/health/doctor.py`:

```python
def write_manifest(db_path, lost_books: Finding, manifest_path=None) -> Path | None:
    """Record every lost book so silent loss becomes a visible TODO.

    Returns None when there is nothing to record.
    """
    if lost_books.count == 0:
        return None

    db_path = Path(db_path)
    if manifest_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        manifest_path = db_path.parent / "doctor" / f"manifest-{stamp}.md"
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    recoverable = [d for d in lost_books.details if d["source_available"]]
    gone = [d for d in lost_books.details if not d["source_available"]]

    lines = [
        "# Lost books manifest",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Total lost books: {lost_books.count} "
        f"(re-ingestable: {len(recoverable)}, source gone: {len(gone)})",
        "",
        "## Re-ingestable (source file found)",
        "",
    ]
    for d in recoverable:
        lines += [
            f"- **{d['basename']}** — {d['chunk_count']} orphaned chunks, "
            f"pipeline `{d['pipeline_id']}`",
            f"  - file: `{d['resolved_path']}`",
            f"  - sample: {d['sample'][:200]!r}",
        ]
    lines += ["", "## Source gone (re-acquire, then drop into the watch dir)", ""]
    for d in gone:
        lines += [
            f"- **{d['basename']}** — {d['chunk_count']} orphaned chunks, "
            f"pipeline `{d['pipeline_id']}`",
            f"  - last known path: `{d['source_path']}`",
            f"  - sample: {d['sample'][:200]!r}",
        ]
    manifest_path.write_text("\n".join(lines) + "\n")
    return manifest_path
```

- [ ] **Step 4: Run — verify GREEN**

Run: `python -m pytest tests/test_doctor.py -q`
Expected: `26 passed`

- [ ] **Step 5: Commit**

```bash
git add agentic_pipeline/health/doctor.py tests/test_doctor.py
git commit -m "feat: doctor lost-books manifest writer

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Destructive fixes — delete orphans, archive dead pipelines, repoint sources

**Files:**
- Modify: `agentic_pipeline/health/doctor.py`
- Modify: `tests/test_doctor.py`

**Interfaces:**
- Consumes: `_ORPHAN_WHERE` (Task 1), lost-books detail keys (Task 2), `PipelineRepository.update_state(..., expected_state=...)` and `.update_source_path(...)` (existing, from PR #2).
- Produces: `fix_delete_orphans(db_path) -> int`; `fix_archive_and_repoint(db_path, lost_books: Finding) -> tuple[int, int, list[dict]]` returning `(archived, repointed, skipped)`; `build_reingest_commands(lost_books: Finding) -> list[str]` — one command per distinct basename (newest pipeline id), only for source-available books.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_doctor.py`:

```python
class TestDestructiveFixes:
    def test_delete_orphans_removes_only_orphans(self, db_path):
        from agentic_pipeline.health.doctor import check_orphaned_chunks, fix_delete_orphans

        conn = _connect(db_path)
        _seed_book(conn); _seed_chapter(conn)
        _seed_chunk(conn, chunk_id="healthy")
        _seed_chunk(conn, chunk_id="o1", chapter_id="GONE")
        _seed_chunk(conn, chunk_id="o2", book_id="GONE", embedded=False)
        conn.commit(); conn.close()

        deleted = fix_delete_orphans(db_path)

        assert deleted == 2
        conn = _connect(db_path)
        remaining = [r["id"] for r in conn.execute("SELECT id FROM chunks")]
        conn.close()
        assert remaining == ["healthy"]
        assert check_orphaned_chunks(db_path).count == 0  # check and fix agree

    def test_archive_moves_complete_to_archived_with_reason(self, db_path, tmp_path):
        from agentic_pipeline.health.doctor import check_lost_books, fix_archive_and_repoint

        src = tmp_path / "lost.epub"
        src.write_bytes(b"x")
        pid = _seed_complete_pipeline(db_path, source_path=str(src))

        lost = check_lost_books(db_path)
        archived, repointed, skipped = fix_archive_and_repoint(db_path, lost)

        assert (archived, repointed, skipped) == (1, 1, [])
        conn = _connect(db_path)
        row = conn.execute(
            "SELECT state FROM processing_pipelines WHERE id = ?", (pid,)
        ).fetchone()
        hist = conn.execute(
            """SELECT agent_output FROM pipeline_state_history
               WHERE pipeline_id = ? AND to_state = 'archived'""", (pid,)
        ).fetchone()
        conn.close()
        assert row["state"] == "archived"
        assert "doctor" in (hist["agent_output"] or "")
        # after archiving, the check goes quiet — alarm-fatigue decision
        assert check_lost_books(db_path).count == 0

    def test_concurrently_moved_pipeline_is_skipped_not_forced(self, db_path):
        """Seed a state change between check and fix — CAS must protect it."""
        from agentic_pipeline.db.pipelines import PipelineRepository
        from agentic_pipeline.health.doctor import check_lost_books, fix_archive_and_repoint
        from agentic_pipeline.pipeline.states import PipelineState

        pid = _seed_complete_pipeline(db_path, source_path="/nowhere/x.epub")
        lost = check_lost_books(db_path)
        # a rival archives it first
        PipelineRepository(db_path).update_state(pid, PipelineState.ARCHIVED)

        archived, repointed, skipped = fix_archive_and_repoint(db_path, lost)

        assert archived == 0
        assert len(skipped) == 1 and skipped[0]["pipeline_id"] == pid

    def test_repoint_makes_reingest_resolvable_despite_sentinel_hash(self, db_path, tmp_path, monkeypatch):
        """The spec's sharpest decision: backfill:-sentinel books must become
        reingestable. After repoint, resolve_source_file succeeds WITHOUT the
        hash fallback (the file is at source_path)."""
        from agentic_pipeline.approval.actions import resolve_source_file
        from agentic_pipeline.db.pipelines import PipelineRepository
        from agentic_pipeline.health.doctor import check_lost_books, fix_archive_and_repoint

        processed = tmp_path / "processed"
        processed.mkdir()
        (processed / "sentinel.epub").write_bytes(b"book bytes")
        monkeypatch.setenv("PROCESSED_DIR", str(processed))
        pid = _seed_complete_pipeline(db_path, source_path="/watch/sentinel.epub")
        conn = _connect(db_path)
        conn.execute(
            "UPDATE processing_pipelines SET content_hash = 'backfill:deadbeef' WHERE id = ?",
            (pid,),
        )
        conn.commit(); conn.close()

        fix_archive_and_repoint(db_path, check_lost_books(db_path))

        record = PipelineRepository(db_path).get(pid)
        assert record["source_path"] == str(processed / "sentinel.epub")
        resolved = resolve_source_file(record["source_path"],
                                       expected_hash=record["content_hash"])
        assert resolved == processed / "sentinel.epub"

    def test_reingest_commands_deduped_per_basename_newest_id(self, db_path):
        from agentic_pipeline.health.doctor import CATEGORY_LOST_BOOKS, Finding, build_reingest_commands

        lost = Finding(
            category=CATEGORY_LOST_BOOKS, count=3, fixable_count=3,
            details=[
                {"pipeline_id": "old-id", "basename": "dupe.epub",
                 "source_available": True, "resolved_path": "/p/dupe.epub",
                 "source_path": "", "chunk_count": 1, "live_copy": False, "sample": ""},
                {"pipeline_id": "new-id", "basename": "dupe.epub",
                 "source_available": True, "resolved_path": "/p/dupe.epub",
                 "source_path": "", "chunk_count": 1, "live_copy": False, "sample": ""},
                {"pipeline_id": "gone-id", "basename": "gone.epub",
                 "source_available": False, "resolved_path": None,
                 "source_path": "", "chunk_count": 1, "live_copy": False, "sample": ""},
            ],
        )

        commands = build_reingest_commands(lost)

        assert commands == ["agentic-pipeline reingest new-id"]  # last wins, gone excluded
```

- [ ] **Step 2: Run — verify RED**

Run: `python -m pytest tests/test_doctor.py -q -k TestDestructiveFixes`
Expected: ImportErrors for `fix_delete_orphans` / `fix_archive_and_repoint` / `build_reingest_commands`.

- [ ] **Step 3: Implement**

Append to `agentic_pipeline/health/doctor.py` (extend top imports in the same edit: `from agentic_pipeline.db.pipelines import ConcurrentModificationError, PipelineRepository` and `from agentic_pipeline.pipeline.states import PipelineState`):

```python
def fix_delete_orphans(db_path) -> int:
    """Delete every orphaned chunk. Single transaction; same WHERE as the check."""
    with get_pipeline_db(str(db_path)) as conn:
        cursor = conn.execute(f"DELETE FROM chunks WHERE {_ORPHAN_WHERE}")
        conn.commit()
        return cursor.rowcount


def fix_archive_and_repoint(db_path, lost_books: Finding):
    """Archive dead pipelines (CAS-guarded) and repoint recoverable sources.

    Returns (archived_count, repointed_count, skipped) where each skipped
    entry is {"pipeline_id": ..., "reason": ...}. A pipeline that moved since
    the check is skipped, never forced (Decision 5); repointing uses the
    already-resolved path so printed reingest commands run without the
    hash fallback (Decision 13).
    """
    repo = PipelineRepository(db_path)
    archived = repointed = 0
    skipped: list[dict] = []
    for d in lost_books.details:
        try:
            repo.update_state(
                d["pipeline_id"],
                PipelineState.ARCHIVED,
                agent_output={"reason": "doctor: book absent from library; see manifest"},
                expected_state=PipelineState.COMPLETE,
            )
            archived += 1
        except ConcurrentModificationError as e:
            skipped.append({"pipeline_id": d["pipeline_id"], "reason": str(e)})
            continue
        if d["source_available"] and d["resolved_path"]:
            repo.update_source_path(d["pipeline_id"], d["resolved_path"])
            repointed += 1
    return archived, repointed, skipped


def build_reingest_commands(lost_books: Finding) -> list[str]:
    """One command per distinct recoverable book — newest pipeline id wins."""
    by_basename: dict[str, str] = {}
    for d in lost_books.details:
        if d["source_available"]:
            by_basename[d["basename"]] = d["pipeline_id"]  # later entries win
    return [f"agentic-pipeline reingest {pid}" for pid in by_basename.values()]
```

- [ ] **Step 4: Run — verify GREEN**

Run: `python -m pytest tests/test_doctor.py -q`
Expected: `31 passed`

- [ ] **Step 5: Commit**

```bash
git add agentic_pipeline/health/doctor.py tests/test_doctor.py
git commit -m "feat: doctor destructive fixes — delete, archive, repoint

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Backfills — content_hash and book_type

**Files:**
- Modify: `agentic_pipeline/health/doctor.py`
- Modify: `tests/test_doctor.py`

**Interfaces:**
- Consumes: `check_null_content_hash` / `check_null_book_type` findings; `compute_content_hash(content: str) -> str` and `read_chapter_content(file_path) -> str` from `src/utils/embedding_sync.py` / `src/utils/file_utils.py` (byte-compat requirement, Decision in spec step 5).
- Produces: `fix_backfill_hashes(db_path, finding) -> tuple[int, list[dict]]` and `fix_backfill_book_types(db_path, finding) -> tuple[int, list[dict]]`, each returning `(fixed_count, skipped)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_doctor.py`:

```python
class TestBackfills:
    def test_hash_backfill_matches_embedding_sync_exactly(self, db_path, tmp_path):
        """Byte-compat: backfilled hash == what embedding_sync would compute."""
        from src.utils.embedding_sync import compute_content_hash
        from src.utils.file_utils import read_chapter_content
        from agentic_pipeline.health.doctor import check_null_content_hash, fix_backfill_hashes

        real = tmp_path / "ch.md"
        real.write_text("# Chapter\n\nsome chapter prose")
        conn = _connect(db_path)
        _seed_book(conn)
        _seed_chapter(conn, chapter_id="fx", content_hash=None, file_path=str(real))
        _seed_chapter(conn, chapter_id="gone", content_hash=None,
                      file_path="/nope.md", number=2)
        conn.commit(); conn.close()

        fixed, skipped = fix_backfill_hashes(db_path, check_null_content_hash(db_path))

        assert fixed == 1
        assert len(skipped) == 1 and skipped[0]["chapter_id"] == "gone"
        conn = _connect(db_path)
        stored = conn.execute(
            "SELECT content_hash FROM chapters WHERE id = 'fx'"
        ).fetchone()["content_hash"]
        conn.close()
        assert stored == compute_content_hash(read_chapter_content(str(real)))
        # idempotent: nothing left to fix
        from agentic_pipeline.health.doctor import check_null_content_hash as chk
        assert chk(db_path).fixable_count == 0

    def test_book_type_backfill_stamps_and_validates(self, db_path):
        from agentic_pipeline.health.doctor import check_null_book_type, fix_backfill_book_types

        pid_ok = _seed_complete_pipeline(db_path, source_path="/w/ok.epub")
        pid_bad = _seed_complete_pipeline(db_path, source_path="/w/bad.epub")
        conn = _connect(db_path)
        _seed_book(conn, book_id=pid_ok, book_type=None)
        _seed_book(conn, book_id=pid_bad, book_type=None)
        conn.execute("UPDATE processing_pipelines SET book_profile=? WHERE id=?",
                     (json.dumps({"book_type": "travel_guide", "confidence": 0.95}), pid_ok))
        conn.execute("UPDATE processing_pipelines SET book_profile=? WHERE id=?",
                     (json.dumps({"book_type": "unknown", "confidence": 0.0}), pid_bad))
        conn.commit(); conn.close()

        fixed, skipped = fix_backfill_book_types(db_path, check_null_book_type(db_path))

        assert fixed == 1
        assert len(skipped) == 1 and skipped[0]["book_id"] == pid_bad
        conn = _connect(db_path)
        row = conn.execute(
            """SELECT book_type, classification_confidence, classified_by
               FROM books WHERE id = ?""", (pid_ok,)
        ).fetchone()
        conn.close()
        assert row["book_type"] == "travel_guide"
        assert row["classification_confidence"] == 0.95
        assert row["classified_by"] == "backfill:doctor"
```

- [ ] **Step 2: Run — verify RED**

Run: `python -m pytest tests/test_doctor.py -q -k TestBackfills`
Expected: ImportErrors for the two fix functions.

- [ ] **Step 3: Implement**

Append to `agentic_pipeline/health/doctor.py`:

```python
def fix_backfill_hashes(db_path, finding: Finding):
    """Backfill chapters.content_hash from chapter files.

    Uses the same read + hash routines embedding_sync uses when it writes
    content_hash, so backfilled values are byte-compatible with existing
    ones. Missing files land in skipped — reported, not silent.
    """
    # Local import: src.utils pulls openai at module import time; keep doctor
    # importable without it except when this fix actually runs.
    from src.utils.embedding_sync import compute_content_hash
    from src.utils.file_utils import read_chapter_content

    fixed = 0
    skipped: list[dict] = []
    with get_pipeline_db(str(db_path)) as conn:
        for d in finding.details:
            if not d["file_exists"]:
                skipped.append({"chapter_id": d["chapter_id"],
                                "reason": f"file missing: {d['file_path']}"})
                continue
            try:
                content = read_chapter_content(d["file_path"])
            except OSError as e:
                skipped.append({"chapter_id": d["chapter_id"], "reason": str(e)})
                continue
            conn.execute(
                "UPDATE chapters SET content_hash = ? WHERE id = ?",
                (compute_content_hash(content), d["chapter_id"]),
            )
            fixed += 1
        conn.commit()
    return fixed, skipped


def fix_backfill_book_types(db_path, finding: Finding):
    """Copy validated book_type/confidence from pipeline profiles into books."""
    fixed = 0
    skipped: list[dict] = []
    with get_pipeline_db(str(db_path)) as conn:
        for d in finding.details:
            if not d["valid"]:
                skipped.append({"book_id": d["book_id"],
                                "reason": f"no valid type in profile: "
                                          f"{d['profile_book_type']!r}"})
                continue
            conn.execute(
                """UPDATE books SET book_type = ?, classification_confidence = ?,
                                    classified_by = 'backfill:doctor'
                   WHERE id = ?""",
                (d["profile_book_type"], d["profile_confidence"], d["book_id"]),
            )
            fixed += 1
        conn.commit()
    return fixed, skipped
```

- [ ] **Step 4: Run — verify GREEN**

Run: `python -m pytest tests/test_doctor.py -q`
Expected: `33 passed`

- [ ] **Step 5: Commit**

```bash
git add agentic_pipeline/health/doctor.py tests/test_doctor.py
git commit -m "feat: doctor backfills — content_hash (embedding_sync-compat) and book_type

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: `apply_fixes` orchestrator + idempotency

**Files:**
- Modify: `agentic_pipeline/health/doctor.py`
- Modify: `agentic_pipeline/health/__init__.py`
- Modify: `tests/test_doctor.py`

**Interfaces:**
- Consumes: every Task 4–7 function.
- Produces: `FixReport(backup_path: str|None, manifest_path: str|None, fixed: dict[str,int], skipped: dict[str,list[dict]], reingest_commands: list[str])`; `apply_fixes(db_path, *, backup: bool = True, manifest_path=None) -> FixReport`. `FixReport.has_failures` property (any skipped list non-empty **excluding** report-only entries — used for `--fix` exit code). Spec order enforced: backup → manifest → delete → archive+repoint → hashes → types → commands. Backup only when something is fixable.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_doctor.py`:

```python
class TestApplyFixes:
    def _seed_everything(self, db_path, tmp_path):
        """One violation of each category, all fixable."""
        src = tmp_path / "lost.epub"
        src.write_bytes(b"book")
        chfile = tmp_path / "ch.md"
        chfile.write_text("prose")
        lost_pid = _seed_complete_pipeline(db_path, source_path=str(src))
        typed_pid = _seed_complete_pipeline(db_path, source_path="/w/typed.epub")
        conn = _connect(db_path)
        _seed_chunk(conn, chunk_id="orph", chapter_id="GONE", book_id=lost_pid)
        _seed_book(conn, book_id=typed_pid, book_type=None)
        _seed_chapter(conn, chapter_id="nohash", book_id=typed_pid,
                      content_hash=None, file_path=str(chfile))
        conn.execute("UPDATE processing_pipelines SET book_profile=? WHERE id=?",
                     (json.dumps({"book_type": "textbook", "confidence": 0.8}), typed_pid))
        conn.commit(); conn.close()
        return lost_pid

    def test_full_fix_then_clean_then_idempotent(self, db_path, tmp_path):
        from agentic_pipeline.health.doctor import apply_fixes, has_violations, run_checks

        self._seed_everything(db_path, tmp_path)

        report = apply_fixes(db_path)

        assert report.backup_path is not None and Path(report.backup_path).exists()
        assert report.manifest_path is not None
        assert report.fixed["orphaned_chunks"] == 1
        assert report.fixed["lost_books"] == 1          # archived
        assert report.fixed["null_content_hash"] == 1
        assert report.fixed["null_book_type"] == 1
        assert len(report.reingest_commands) == 1
        assert has_violations(run_checks(db_path)) is False
        assert report.has_failures is False

        # Second run: nothing to do, NO new backup (Decision 8)
        before = sorted(db_path.parent.glob(f"{db_path.name}.backup-doctor-*"))
        report2 = apply_fixes(db_path)
        after = sorted(db_path.parent.glob(f"{db_path.name}.backup-doctor-*"))
        assert sum(report2.fixed.values()) == 0
        assert report2.backup_path is None
        assert before == after

    def test_no_backup_flag(self, db_path, tmp_path):
        from agentic_pipeline.health.doctor import apply_fixes

        self._seed_everything(db_path, tmp_path)
        report = apply_fixes(db_path, backup=False)

        assert report.backup_path is None
        assert report.fixed["orphaned_chunks"] == 1

    def test_unfixable_items_are_reported_not_failures(self, db_path):
        """A source-gone chapter is report-only skip, not a fix failure."""
        from agentic_pipeline.health.doctor import apply_fixes

        conn = _connect(db_path)
        _seed_book(conn)
        _seed_chapter(conn, chapter_id="gone", content_hash=None, file_path="/nope.md")
        conn.commit(); conn.close()

        report = apply_fixes(db_path)

        assert report.fixed["null_content_hash"] == 0
        assert len(report.skipped["null_content_hash"]) == 1
        assert report.has_failures is False  # unfixable ≠ failed

    def test_every_category_has_a_fix_handler(self):
        """Contract rule: enumeration completeness over CATEGORIES."""
        from agentic_pipeline.health.doctor import CATEGORIES, FIX_HANDLED_CATEGORIES

        assert set(FIX_HANDLED_CATEGORIES) == set(CATEGORIES)

    def test_fixreport_field_types(self, db_path):
        from agentic_pipeline.health.doctor import apply_fixes

        report = apply_fixes(db_path)
        assert report.backup_path is None or isinstance(report.backup_path, str)
        assert isinstance(report.fixed, dict)
        assert isinstance(report.skipped, dict)
        assert isinstance(report.reingest_commands, list)
        assert isinstance(report.has_failures, bool)
```

- [ ] **Step 2: Run — verify RED**

Run: `python -m pytest tests/test_doctor.py -q -k TestApplyFixes`
Expected: ImportErrors (`apply_fixes`, `FIX_HANDLED_CATEGORIES`).

- [ ] **Step 3: Implement**

Append to `agentic_pipeline/health/doctor.py`:

```python
# Every category doctor can detect, it can also fix (or explicitly reports).
# The enum-completeness test pins this to CATEGORIES so a fifth check can
# never be added without deciding its fix story.
FIX_HANDLED_CATEGORIES = (
    CATEGORY_ORPHANED_CHUNKS,   # fix: delete
    CATEGORY_LOST_BOOKS,        # fix: archive + repoint + manifest
    CATEGORY_NULL_CONTENT_HASH, # fix: backfill from files
    CATEGORY_NULL_BOOK_TYPE,    # fix: backfill from profiles
)


@dataclass
class FixReport:
    """What apply_fixes did, per category."""

    backup_path: str | None = None
    manifest_path: str | None = None
    fixed: dict = field(default_factory=dict)
    skipped: dict = field(default_factory=dict)
    reingest_commands: list = field(default_factory=list)
    # skips that are fix FAILURES (vs. report-only "cannot ever fix" items)
    failure_categories: set = field(default_factory=set)

    @property
    def has_failures(self) -> bool:
        return bool(self.failure_categories)


def apply_fixes(db_path, *, backup: bool = True, manifest_path=None) -> FixReport:
    """Run checks fresh, then repair in spec order. Idempotent."""
    findings = {f.category: f for f in run_checks(db_path)}
    report = FixReport()

    will_mutate = (
        findings[CATEGORY_ORPHANED_CHUNKS].count > 0
        or findings[CATEGORY_LOST_BOOKS].count > 0
        or findings[CATEGORY_NULL_CONTENT_HASH].fixable_count > 0
        or findings[CATEGORY_NULL_BOOK_TYPE].fixable_count > 0
    )

    # 1. Backup — only when something will change (Decision 8).
    if backup and will_mutate:
        report.backup_path = str(create_backup(db_path))

    # 2. Manifest — before anything is deleted.
    manifest = write_manifest(db_path, findings[CATEGORY_LOST_BOOKS], manifest_path)
    report.manifest_path = str(manifest) if manifest else None

    # 3. Delete orphans.
    report.fixed[CATEGORY_ORPHANED_CHUNKS] = fix_delete_orphans(db_path)
    report.skipped[CATEGORY_ORPHANED_CHUNKS] = []

    # 4 + 4.5. Archive dead pipelines, repoint recoverable sources.
    archived, _repointed, arch_skipped = fix_archive_and_repoint(
        db_path, findings[CATEGORY_LOST_BOOKS]
    )
    report.fixed[CATEGORY_LOST_BOOKS] = archived
    report.skipped[CATEGORY_LOST_BOOKS] = arch_skipped
    if arch_skipped:
        # a CAS/ownership skip is a genuine failure to complete the fix
        report.failure_categories.add(CATEGORY_LOST_BOOKS)

    # 5. Backfill hashes (missing files are report-only, not failures).
    fixed, skipped = fix_backfill_hashes(db_path, findings[CATEGORY_NULL_CONTENT_HASH])
    report.fixed[CATEGORY_NULL_CONTENT_HASH] = fixed
    report.skipped[CATEGORY_NULL_CONTENT_HASH] = skipped

    # 6. Backfill book types (invalid profiles are report-only).
    fixed, skipped = fix_backfill_book_types(db_path, findings[CATEGORY_NULL_BOOK_TYPE])
    report.fixed[CATEGORY_NULL_BOOK_TYPE] = fixed
    report.skipped[CATEGORY_NULL_BOOK_TYPE] = skipped

    # 7. Re-ingest commands — printed advice, never affects exit code.
    report.reingest_commands = build_reingest_commands(findings[CATEGORY_LOST_BOOKS])
    return report
```

Update `agentic_pipeline/health/__init__.py` exports to add `FixReport`, `apply_fixes`, `FIX_HANDLED_CATEGORIES` (import + `__all__` in one edit).

- [ ] **Step 4: Run — verify GREEN**

Run: `python -m pytest tests/test_doctor.py -q`
Expected: `39 passed`

- [ ] **Step 5: Commit**

```bash
git add agentic_pipeline/health/doctor.py agentic_pipeline/health/__init__.py tests/test_doctor.py
git commit -m "feat: doctor apply_fixes orchestrator, idempotent with backup skip

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: CLI `doctor` command with exit codes

**Files:**
- Modify: `agentic_pipeline/cli.py` (new command, placed with the other commands before `if __name__ == "__main__":`)
- Modify: `tests/test_doctor.py`

**Interfaces:**
- Consumes: `run_checks`, `has_violations`, `apply_fixes` (Tasks 3, 8).
- Produces: `agentic-pipeline doctor [--fix] [--no-backup] [--manifest PATH]`. Exit codes (Decision 15): report mode 1 if violations else 0; `--fix` 1 if `report.has_failures` else 0.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_doctor.py`:

```python
from click.testing import CliRunner


class TestDoctorCli:
    def _run(self, db_path, *args):
        import os
        from agentic_pipeline.cli import main

        env = dict(os.environ, AGENTIC_PIPELINE_DB=str(db_path))
        return CliRunner().invoke(main, ["doctor", *args], env=env)

    def test_report_clean_exits_zero(self, db_path):
        result = self._run(db_path)
        assert result.exit_code == 0
        assert "OK" in result.output or "0" in result.output

    def test_report_violations_exit_one(self, db_path):
        conn = _connect(db_path)
        _seed_chunk(conn, chunk_id="o", chapter_id="GONE", book_id="GONE")
        conn.commit(); conn.close()

        result = self._run(db_path)

        assert result.exit_code == 1
        assert "orphaned_chunks" in result.output

    def test_fix_repairs_and_exits_zero(self, db_path):
        conn = _connect(db_path)
        _seed_chunk(conn, chunk_id="o", chapter_id="GONE", book_id="GONE")
        conn.commit(); conn.close()

        result = self._run(db_path, "--fix", "--no-backup")

        assert result.exit_code == 0
        assert self._run(db_path).exit_code == 0  # now clean

    def test_fix_prints_reingest_commands_and_manifest_path(self, db_path, tmp_path):
        src = tmp_path / "lost.epub"
        src.write_bytes(b"x")
        _seed_complete_pipeline(db_path, source_path=str(src))

        result = self._run(db_path, "--fix", "--no-backup",
                           "--manifest", str(tmp_path / "m.md"))

        assert result.exit_code == 0
        assert "agentic-pipeline reingest" in result.output
        assert str(tmp_path / "m.md") in result.output
```

- [ ] **Step 2: Run — verify RED**

Run: `python -m pytest tests/test_doctor.py -q -k TestDoctorCli`
Expected: `Error: No such command 'doctor'` in output → exit_code 2 → assertion failures.

- [ ] **Step 3: Implement**

Add to `agentic_pipeline/cli.py`, following the house pattern (lazy imports, `get_db_path()`), immediately before the `if __name__ == "__main__":` block:

```python
@main.command()
@click.option("--fix", "do_fix", is_flag=True, help="Repair what the checks find.")
@click.option("--no-backup", is_flag=True, help="Skip the automatic pre-fix backup.")
@click.option("--manifest", "manifest_path", type=click.Path(), default=None,
              help="Where to write the lost-books manifest.")
def doctor(do_fix: bool, no_backup: bool, manifest_path):
    """Detect (and with --fix, repair) library integrity drift.

    Bare `doctor` is the always-safe report and exits 1 when violations
    exist, so `agentic-pipeline doctor || alert` works in cron. `--fix` is
    the consent — no prompt (house --execute convention).
    """
    import sys

    from .db.config import get_db_path
    from .health.doctor import apply_fixes, has_violations, run_checks

    db_path = get_db_path()
    findings = run_checks(db_path)

    console.print("\n[bold]Integrity Doctor[/bold]")
    console.print("-" * 44)
    for f in findings:
        marker = "[red]✗[/red]" if f.count else "[green]✓[/green]"
        console.print(f"  {marker} {f.category:20} {f.count:>6}  (fixable: {f.fixable_count})")

    if not do_fix:
        if has_violations(findings):
            console.print("\n[yellow]Run 'agentic-pipeline doctor --fix' to repair.[/yellow]")
            sys.exit(1)
        console.print("\n[green]integrity: OK[/green]")
        return

    report = apply_fixes(db_path, backup=not no_backup, manifest_path=manifest_path)

    console.print("\n[bold]Fixes applied[/bold]")
    if report.backup_path:
        console.print(f"  backup   : {report.backup_path}")
    if report.manifest_path:
        console.print(f"  manifest : {report.manifest_path}")
    for category, n in report.fixed.items():
        skipped = report.skipped.get(category, [])
        line = f"  {category:20} fixed {n}"
        if skipped:
            line += f", skipped {len(skipped)}"
        console.print(line)
        for s in skipped[:5]:
            console.print(f"      [dim]{s.get('reason', s)}[/dim]")
    if report.reingest_commands:
        console.print("\n[bold]Re-ingest these recovered books:[/bold]")
        for cmd in report.reingest_commands:
            console.print(f"  {cmd}")
    if report.has_failures:
        console.print("\n[red]Some fixes could not be applied — see skipped above.[/red]")
        sys.exit(1)
```

- [ ] **Step 4: Run — verify GREEN**

Run: `python -m pytest tests/test_doctor.py -q`
Expected: `43 passed`

- [ ] **Step 5: Commit**

```bash
git add agentic_pipeline/cli.py tests/test_doctor.py
git commit -m "feat: agentic-pipeline doctor command with scriptable exit codes

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: `health` integrity line

**Files:**
- Modify: `agentic_pipeline/cli.py:395` (the existing `health` command)
- Modify: `tests/test_doctor.py`

**Interfaces:**
- Consumes: `run_checks`, `has_violations`.
- Produces: `report["integrity"] = {"issues": int, "categories": {category: count}}` in `--json` mode; a one-line verdict in console mode. `health`'s exit code is unchanged (dashboard, not gate — Decision 15).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_doctor.py`:

```python
class TestHealthIntegration:
    def _run_health(self, db_path, *args):
        import os
        from agentic_pipeline.cli import main

        env = dict(os.environ, AGENTIC_PIPELINE_DB=str(db_path))
        return CliRunner().invoke(main, ["health", *args], env=env)

    def test_health_reports_integrity_ok(self, db_path):
        result = self._run_health(db_path)
        assert "integrity: OK" in result.output
        assert result.exit_code == 0

    def test_health_reports_issue_count_and_stays_exit_zero(self, db_path):
        conn = _connect(db_path)
        _seed_chunk(conn, chunk_id="o", chapter_id="GONE", book_id="GONE")
        conn.commit(); conn.close()

        result = self._run_health(db_path)

        assert "integrity: 1 issue" in result.output
        assert "doctor" in result.output           # points at the tool
        assert result.exit_code == 0               # dashboard, not gate

    def test_health_json_carries_integrity_block(self, db_path):
        conn = _connect(db_path)
        _seed_chunk(conn, chunk_id="o", chapter_id="GONE", book_id="GONE")
        conn.commit(); conn.close()

        result = self._run_health(db_path, "--json")
        payload = json.loads(result.output)

        assert payload["integrity"]["issues"] == 1
        assert payload["integrity"]["categories"]["orphaned_chunks"] == 1
```

- [ ] **Step 2: Run — verify RED**

Run: `python -m pytest tests/test_doctor.py -q -k TestHealthIntegration`
Expected: assertion failures — no `integrity` in output/payload.

Note: if `health` errors on an empty DB for unrelated reasons, read its traceback before touching anything — the fix must be in the new integrity code only.

- [ ] **Step 3: Implement**

In `agentic_pipeline/cli.py`, inside `def health(...)` (line ~395): after `report["stuck"] = detector.detect()` add:

```python
    from .health.doctor import run_checks

    findings = run_checks(db_path)
    issues = sum(f.count for f in findings)
    report["integrity"] = {
        "issues": issues,
        "categories": {f.category: f.count for f in findings},
    }
```

Then, in the console (non-JSON) branch, after the existing stuck output, add:

```python
    if issues == 0:
        console.print("  [green]integrity: OK[/green]")
    else:
        plural = "issue" if issues == 1 else "issues"
        console.print(
            f"  [yellow]integrity: {issues} {plural} — run 'agentic-pipeline doctor'[/yellow]"
        )
```

(The `--json` early-return path already emits `report`, which now carries the `integrity` block — no separate change needed there, but verify the `run_checks` call sits BEFORE the `if as_json:` return.)

- [ ] **Step 4: Run — verify GREEN**

Run: `python -m pytest tests/test_doctor.py -q && python -m pytest tests/ -q`
Expected: doctor suite `46 passed`; full suite green (health tests elsewhere must not regress).

- [ ] **Step 5: Commit**

```bash
git add agentic_pipeline/cli.py tests/test_doctor.py
git commit -m "feat: health surfaces integrity verdict from doctor checks

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 11: `doctor_report` MCP tool (capped payload) + wrapper registration

**Files:**
- Modify: `agentic_pipeline/mcp_server.py`
- Modify: `agentic_mcp_server.py`
- Modify: `tests/test_doctor.py`
- Modify: `tests/test_mcp_server_tools.py`

**Interfaces:**
- Consumes: `run_checks`.
- Produces: `doctor_report(db_path: Optional[str] = None) -> dict` in `agentic_pipeline/mcp_server.py`; registered as tool `doctor_report` in the wrapper. Payload shape (Decision 14): per category `{count, fixable_count, samples: [≤10 detail dicts], truncated: bool}` — except `lost_books`, whose `samples` holds ALL details (it is the actionable list) with `truncated: False`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_doctor.py`:

```python
class TestDoctorReportMcp:
    def test_caps_samples_and_flags_truncation(self, db_path):
        from agentic_pipeline.mcp_server import doctor_report

        conn = _connect(db_path)
        for i in range(15):
            _seed_chunk(conn, chunk_id=f"o{i}", chapter_id="GONE", book_id="GONE")
        conn.commit(); conn.close()

        payload = doctor_report(db_path=str(db_path))

        oc = payload["orphaned_chunks"]
        assert oc["count"] == 15
        assert len(oc["samples"]) == 10
        assert oc["truncated"] is True

    def test_lost_books_returns_all_details(self, db_path):
        from agentic_pipeline.mcp_server import doctor_report

        for i in range(12):
            _seed_complete_pipeline(db_path, pipeline_id=str(i),
                                    source_path=f"/nowhere/book{i}.epub")

        lb = doctor_report(db_path=str(db_path))["lost_books"]

        assert lb["count"] == 12
        assert len(lb["samples"]) == 12         # the actionable list, uncapped
        assert lb["truncated"] is False

    def test_clean_db_shape(self, db_path):
        from agentic_pipeline.health.doctor import CATEGORIES
        from agentic_pipeline.mcp_server import doctor_report

        payload = doctor_report(db_path=str(db_path))

        assert set(payload.keys()) == set(CATEGORIES)
        for category in CATEGORIES:
            block = payload[category]
            assert block["count"] == 0
            assert block["samples"] == []
            assert block["truncated"] is False
```

Append to `tests/test_mcp_server_tools.py` (same style as the existing wrapper test at line 105):

```python
def test_doctor_report_registered_in_wrapper():
    """doctor_report must be exposed by the agentic_mcp_server wrapper."""
    import agentic_mcp_server

    tool_names = list(agentic_mcp_server.mcp._tool_manager._tools.keys())
    assert "doctor_report" in tool_names, f"doctor_report not registered. Tools: {tool_names}"
```

- [ ] **Step 2: Run — verify RED**

Run: `python -m pytest tests/test_doctor.py -q -k DoctorReportMcp && python -m pytest tests/test_mcp_server_tools.py -q -k doctor_report`
Expected: `ImportError: cannot import name 'doctor_report'`; wrapper assertion failure.

- [ ] **Step 3: Implement**

In `agentic_pipeline/mcp_server.py`, following the `review_pending_books` pattern (line 11):

```python
_DOCTOR_SAMPLE_CAP = 10


def doctor_report(db_path: Optional[str] = None) -> dict:
    """Read-only integrity report, payload-capped for chat contexts.

    lost_books returns every detail (it is the actionable list); other
    categories return at most _DOCTOR_SAMPLE_CAP samples plus a truncated
    flag. Repair is deliberately NOT exposed over MCP — use the CLI.
    """
    from agentic_pipeline.health.doctor import CATEGORY_LOST_BOOKS, run_checks

    path = Path(db_path) if db_path else get_db_path()
    payload = {}
    for finding in run_checks(path):
        if finding.category == CATEGORY_LOST_BOOKS:
            samples = finding.details
            truncated = False
        else:
            samples = finding.details[:_DOCTOR_SAMPLE_CAP]
            truncated = len(finding.details) > _DOCTOR_SAMPLE_CAP
        payload[finding.category] = {
            "count": finding.count,
            "fixable_count": finding.fixable_count,
            "samples": samples,
            "truncated": truncated,
        }
    return payload
```

In `agentic_mcp_server.py`: add `doctor_report as doctor_report_fn` to the existing `from agentic_pipeline.mcp_server import (...)` block, and alongside the other wrappers (near `def backfill`, line ~166):

```python
@mcp.tool()
def doctor_report() -> dict:
    """Library integrity report: orphaned chunks, lost books, NULL hashes/types."""
    return doctor_report_fn()
```

- [ ] **Step 4: Run — verify GREEN, then the full suite**

Run: `python -m pytest tests/test_doctor.py tests/test_mcp_server_tools.py -q && python -m pytest tests/ -q && ruff check agentic_pipeline/ tests/ agentic_mcp_server.py`
Expected: doctor suite `49 passed`; full suite green; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add agentic_pipeline/mcp_server.py agentic_mcp_server.py tests/test_doctor.py tests/test_mcp_server_tools.py
git commit -m "feat: doctor_report MCP tool with capped payload

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 12: Live-run verification and docs

**Files:**
- Modify: `CLAUDE.md` (CLI table mention), `ref/cli-commands.md`, `ref/mcp-tools.md`, `ref/module-map.md`

This task runs against the REAL library DB — it is the payoff and the end-to-end proof. The worker may stay running (spec records why no lock is needed), but do it deliberately:

- [ ] **Step 1: Report mode against the live DB**

Run: `source .venv/bin/activate && agentic-pipeline doctor; echo "exit=$?"`
Expected: the four categories with counts ≈ (orphans 2,449 · lost 25 · hash 840 · type 47 — numbers may have drifted slightly), `exit=1`.

- [ ] **Step 2: Fix (with backup)**

Run: `agentic-pipeline doctor --fix; echo "exit=$?"`
Expected: backup + manifest paths printed; ~2,449 orphans deleted; 25 archived; hashes/types backfilled with a skipped list for missing files and invalid profiles; 20 reingest commands printed. `exit=0` unless CAS skips occurred.

- [ ] **Step 3: Verify clean + idempotent + health**

Run: `agentic-pipeline doctor; echo "exit=$?"; agentic-pipeline doctor --fix; agentic-pipeline health | grep -i integrity`
Expected: report exits 0; second `--fix` fixes nothing and creates no new backup; health prints `integrity: OK`.

- [ ] **Step 4: Spot-verify searchability gain**

Run the searchable-chunks join count before/after (from the session's verification snippets); confirm searchable == embedded (no orphans remain), library integrity `ok`.

- [ ] **Step 5: Update docs**

Add `doctor` to `ref/cli-commands.md` (with exit-code semantics) and `CLAUDE.md`'s command list; add `doctor_report` to `ref/mcp-tools.md`; add `health/doctor.py` to `ref/module-map.md`.

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md ref/
git commit -m "docs: doctor command, doctor_report tool, module map

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

Deliberately NOT in this task: running the 20 `reingest` commands (operational, human-triggered per spec Decision 3) and re-acquiring the 5 source-gone books (manifest lists them).

---

## Self-Review (performed while writing)

- **Spec coverage:** Decisions 1–4 → Tasks 6/12 + plan scope; 5 → Task 6; 6 → Task 10; 7 → Task 5; 8 → Tasks 4/8; 9 → Task 11; 10 → Task 9 (no prompt); 11 → Tasks 3/7; 12 → Task 6 (single delete); 13 → Task 6 (repoint + sentinel test); 14 → Task 11; 15 → Tasks 9/10. All four checks have seeded-violation AND clean tests (Tasks 1–3); idempotency + backup-skip (Task 8); hash byte-compat (Task 7); manifest content (Task 5); rotation + bystander safety (Task 4).
- **Placeholders:** none — every step carries runnable code or an exact command.
- **Type consistency:** `Finding`/`FixReport` field names, lost-books detail keys (`pipeline_id`, `basename`, `source_available`, `resolved_path`, `chunk_count`, `live_copy`, `sample`, `source_path`), and function signatures are identical across producing and consuming tasks.

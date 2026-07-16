"""Tests for the integrity doctor."""

import json
import sqlite3
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from agentic_pipeline.db.migrations import run_migrations


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "doctor.db"
    run_migrations(path)
    # NOTE (adjustment): run_migrations() only creates agentic_pipeline's own
    # tables (processing_pipelines, chunks, etc.) — `books`/`chapters` are
    # created by the external book-ingestion package against the real,
    # shared library DB and don't exist in a fresh migrated test DB. Several
    # existing test files (test_backfill.py, test_library_status.py,
    # test_mcp_backfill_tools.py, test_reprocess_cli.py) already work around
    # this the same way: create the minimal books/chapters schema ad hoc.
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS books (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            author TEXT,
            book_type TEXT,
            source_file TEXT,
            word_count INTEGER DEFAULT 0,
            classification_confidence REAL,
            classified_by TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS chapters (
            id TEXT PRIMARY KEY,
            book_id TEXT NOT NULL,
            chapter_number INTEGER,
            title TEXT,
            file_path TEXT,
            word_count INTEGER DEFAULT 0,
            content_hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (book_id) REFERENCES books(id)
        )"""
    )
    conn.commit()
    conn.close()
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


def _seed_chapter(conn, chapter_id="c1", book_id="b1", file_path="/nope.md", content_hash="hash", number=1):
    conn.execute(
        """INSERT INTO chapters (id, book_id, chapter_number, title, file_path,
                                 word_count, content_hash)
           VALUES (?, ?, ?, ?, ?, 100, ?)""",
        (chapter_id, book_id, number, f"Chapter {number}", file_path, content_hash),
    )


def _seed_chunk(
    conn, chunk_id="k1", chapter_id="c1", book_id="b1", content="chunk text body", embedded=True, chunk_index=0
):
    # chunk_index defaults to 0 for every existing call site; only pass a
    # non-default value when seeding >1 chunk under the same chapter_id —
    # chunks has a UNIQUE INDEX on (chapter_id, chunk_index).
    conn.execute(
        """INSERT INTO chunks (id, chapter_id, book_id, chunk_index, content,
                               word_count, embedding, embedding_model, content_hash)
           VALUES (?, ?, ?, ?, ?, 3, ?, 'test-model', 'kh')""",
        (chunk_id, chapter_id, book_id, chunk_index, content, b"\x00\x01" if embedded else None),
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
        _seed_chunk(conn, chunk_id="ok", chapter_id="c1")  # healthy
        _seed_chunk(conn, chunk_id="orphan", chapter_id="GONE")  # dead chapter
        conn.commit()
        conn.close()

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
        conn.commit()
        conn.close()

        assert check_orphaned_chunks(db_path).count == 1

    def test_counts_unembedded_orphans_identically(self, db_path):
        """Spec: doctor treats embedded and never-embedded orphans identically."""
        from agentic_pipeline.health.doctor import check_orphaned_chunks

        conn = _connect(db_path)
        _seed_chunk(conn, chunk_id="o1", chapter_id="GONE", book_id="GONE", embedded=True, chunk_index=0)
        _seed_chunk(conn, chunk_id="o2", chapter_id="GONE", book_id="GONE", embedded=False, chunk_index=1)
        conn.commit()
        conn.close()

        assert check_orphaned_chunks(db_path).count == 2

    def test_clean_db_passes(self, db_path):
        from agentic_pipeline.health.doctor import check_orphaned_chunks

        conn = _connect(db_path)
        _seed_book(conn)
        _seed_chapter(conn)
        _seed_chunk(conn)
        conn.commit()
        conn.close()

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


def _seed_complete_pipeline(db_path, pipeline_id=None, source_path="/watch/lost.epub"):
    """Create a pipeline and walk it to COMPLETE through valid transitions."""
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    repo = PipelineRepository(db_path)
    pid = repo.create(source_path, f"hash-{pipeline_id or 'x'}")
    for state in (
        PipelineState.HASHING,
        PipelineState.CLASSIFYING,
        PipelineState.SELECTING_STRATEGY,
        PipelineState.PROCESSING,
        PipelineState.VALIDATING,
        PipelineState.PENDING_APPROVAL,
        PipelineState.APPROVED,
        PipelineState.EMBEDDING,
        PipelineState.COMPLETE,
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
        _seed_chunk(
            conn, chunk_id="lost-k", chapter_id="GONE", book_id=pid, content="a sample of the lost book's text " * 20
        )
        conn.commit()
        conn.close()

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
        conn.commit()
        conn.close()

        assert check_lost_books(db_path).count == 0

    def test_fixable_count_equals_count(self, db_path):
        """Archiving fixes every lost book, source or no source."""
        from agentic_pipeline.health.doctor import check_lost_books

        _seed_complete_pipeline(db_path, source_path="/nowhere/gone.epub")

        finding = check_lost_books(db_path)
        assert finding.fixable_count == finding.count == 1

    def test_live_copy_true_when_another_book_shares_basename(self, db_path):
        """A lost pipeline's basename also belongs to a pipeline WITH a books row."""
        from agentic_pipeline.health.doctor import check_lost_books

        lost_pid = _seed_complete_pipeline(db_path, source_path="/watch/dup.epub")
        live_pid = _seed_complete_pipeline(db_path, pipeline_id="live", source_path="/other/dup.epub")
        conn = _connect(db_path)
        _seed_book(conn, book_id=live_pid)
        conn.commit()
        conn.close()

        d = check_lost_books(db_path).details[0]
        assert d["pipeline_id"] == lost_pid
        assert d["live_copy"] is True

    def test_live_copy_false_for_wildcard_lookalike_basename(self, db_path):
        """LIKE metacharacters in the basename must not cause false-positive matches."""
        from agentic_pipeline.health.doctor import check_lost_books

        lost_pid = _seed_complete_pipeline(db_path, source_path="/watch/chapter_1.epub")
        other_pid = _seed_complete_pipeline(db_path, pipeline_id="other", source_path="/other/chapterX1.epub")
        conn = _connect(db_path)
        _seed_book(conn, book_id=other_pid)
        conn.commit()
        conn.close()

        d = check_lost_books(db_path).details[0]
        assert d["pipeline_id"] == lost_pid
        assert d["basename"] == "chapter_1.epub"
        assert d["live_copy"] is False


class TestCheckNullContentHash:
    def test_flags_null_hash_and_reports_file_availability(self, db_path, tmp_path):
        from agentic_pipeline.health.doctor import check_null_content_hash

        real = tmp_path / "ch.md"
        real.write_text("chapter body")
        conn = _connect(db_path)
        _seed_book(conn)
        _seed_chapter(conn, chapter_id="fixable", content_hash=None, file_path=str(real))
        _seed_chapter(conn, chapter_id="unfixable", content_hash=None, file_path="/nope.md", number=2)
        _seed_chapter(conn, chapter_id="healthy", content_hash="abc", number=3)
        conn.commit()
        conn.close()

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
        conn.commit()
        conn.close()
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

    def test_non_dict_profile_json_does_not_crash(self, db_path):
        """book_profile can be valid JSON with the wrong shape (string/list/number) —
        the check must report it, not raise AttributeError on .get()."""
        from agentic_pipeline.health.doctor import check_null_book_type

        pid = _seed_complete_pipeline(db_path, source_path="/watch/typed.epub")
        conn = _connect(db_path)
        _seed_book(conn, book_id=pid, book_type=None)
        conn.execute(
            "UPDATE processing_pipelines SET book_profile = ? WHERE id = ?",
            ('"just a string"', pid),
        )
        conn.commit()
        conn.close()

        finding = check_null_book_type(db_path)
        assert finding.count == 1
        assert finding.fixable_count == 0
        assert finding.details[0]["profile_book_type"] is None
        assert finding.details[0]["valid"] is False

    def test_clean_db_passes(self, db_path):
        from agentic_pipeline.health.doctor import check_null_book_type

        assert check_null_book_type(db_path).count == 0

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
        conn.commit()
        conn.close()
        assert has_violations(run_checks(db_path)) is True


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
        assert made[0] not in survivors  # oldest pruned
        assert made[2] in survivors  # newest kept

    def test_never_touches_non_doctor_files(self, db_path):
        from agentic_pipeline.health.doctor import create_backup

        bystander = db_path.parent / f"{db_path.name}.backup-manual"
        bystander.write_bytes(b"precious user backup")
        for _ in range(3):
            create_backup(db_path)
            time.sleep(1.1)

        assert bystander.exists()
        assert bystander.read_bytes() == b"precious user backup"


class TestManifest:
    def _lost_finding(self):
        from agentic_pipeline.health.doctor import CATEGORY_LOST_BOOKS, Finding

        return Finding(
            category=CATEGORY_LOST_BOOKS,
            count=2,
            fixable_count=2,
            details=[
                {
                    "pipeline_id": "p-1",
                    "source_path": "/w/a.epub",
                    "basename": "a.epub",
                    "chunk_count": 10,
                    "source_available": True,
                    "resolved_path": "/proc/a.epub",
                    "live_copy": False,
                    "sample": "sample text from book a",
                },
                {
                    "pipeline_id": "p-2",
                    "source_path": "/w/b.epub",
                    "basename": "b.epub",
                    "chunk_count": 5,
                    "source_available": False,
                    "resolved_path": None,
                    "live_copy": False,
                    "sample": "sample text from book b",
                },
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


class TestDestructiveFixes:
    def test_delete_orphans_removes_only_orphans(self, db_path):
        from agentic_pipeline.health.doctor import check_orphaned_chunks, fix_delete_orphans

        conn = _connect(db_path)
        _seed_book(conn)
        _seed_chapter(conn)
        _seed_chunk(conn, chunk_id="healthy")
        _seed_chunk(conn, chunk_id="o1", chapter_id="GONE")
        # NOTE (trap fix): chunks has UNIQUE(chapter_id, chunk_index) and
        # _seed_chunk defaults chunk_index=0. "healthy" and "o2" both default
        # to chapter_id="c1" at index 0, which collides. Pass a distinct
        # chunk_index for o2 to avoid the IntegrityError.
        _seed_chunk(conn, chunk_id="o2", book_id="GONE", embedded=False, chunk_index=1)
        conn.commit()
        conn.close()

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
        row = conn.execute("SELECT state FROM processing_pipelines WHERE id = ?", (pid,)).fetchone()
        hist = conn.execute(
            """SELECT agent_output FROM pipeline_state_history
               WHERE pipeline_id = ? AND to_state = 'archived'""",
            (pid,),
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
        conn.commit()
        conn.close()

        fix_archive_and_repoint(db_path, check_lost_books(db_path))

        record = PipelineRepository(db_path).get(pid)
        assert record["source_path"] == str(processed / "sentinel.epub")
        resolved = resolve_source_file(record["source_path"], expected_hash=record["content_hash"])
        assert resolved == processed / "sentinel.epub"

    def test_reingest_commands_deduped_per_basename_newest_id(self, db_path):
        from agentic_pipeline.health.doctor import CATEGORY_LOST_BOOKS, Finding, build_reingest_commands

        lost = Finding(
            category=CATEGORY_LOST_BOOKS,
            count=3,
            fixable_count=3,
            details=[
                {
                    "pipeline_id": "old-id",
                    "basename": "dupe.epub",
                    "source_available": True,
                    "resolved_path": "/p/dupe.epub",
                    "source_path": "",
                    "chunk_count": 1,
                    "live_copy": False,
                    "sample": "",
                },
                {
                    "pipeline_id": "new-id",
                    "basename": "dupe.epub",
                    "source_available": True,
                    "resolved_path": "/p/dupe.epub",
                    "source_path": "",
                    "chunk_count": 1,
                    "live_copy": False,
                    "sample": "",
                },
                {
                    "pipeline_id": "gone-id",
                    "basename": "gone.epub",
                    "source_available": False,
                    "resolved_path": None,
                    "source_path": "",
                    "chunk_count": 1,
                    "live_copy": False,
                    "sample": "",
                },
            ],
        )

        commands = build_reingest_commands(lost)

        assert commands == ["agentic-pipeline reingest new-id"]  # last wins, gone excluded


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
        _seed_chapter(conn, chapter_id="gone", content_hash=None, file_path="/nope.md", number=2)
        conn.commit()
        conn.close()

        fixed, skipped = fix_backfill_hashes(db_path, check_null_content_hash(db_path))

        assert fixed == 1
        assert len(skipped) == 1 and skipped[0]["chapter_id"] == "gone"
        conn = _connect(db_path)
        stored = conn.execute("SELECT content_hash FROM chapters WHERE id = 'fx'").fetchone()["content_hash"]
        conn.close()
        assert stored == compute_content_hash(read_chapter_content(str(real)))
        # idempotent: nothing left to fix
        from agentic_pipeline.health.doctor import check_null_content_hash as chk

        assert chk(db_path).fixable_count == 0

    def test_book_type_backfill_stamps_and_validates(self, db_path):
        from agentic_pipeline.health.doctor import check_null_book_type, fix_backfill_book_types

        # NOTE (adjustment): _seed_complete_pipeline's content_hash defaults to
        # f"hash-{pipeline_id or 'x'}"; two calls with the default None both
        # collide on "hash-x" against processing_pipelines' UNIQUE(content_hash).
        # Pass distinct pipeline_id values, as other multi-seed tests in this
        # file already do (e.g. TestCheckLostBooks.test_live_copy_true_...).
        pid_ok = _seed_complete_pipeline(db_path, pipeline_id="ok", source_path="/w/ok.epub")
        pid_bad = _seed_complete_pipeline(db_path, pipeline_id="bad", source_path="/w/bad.epub")
        conn = _connect(db_path)
        _seed_book(conn, book_id=pid_ok, book_type=None)
        _seed_book(conn, book_id=pid_bad, book_type=None)
        conn.execute(
            "UPDATE processing_pipelines SET book_profile=? WHERE id=?",
            (json.dumps({"book_type": "travel_guide", "confidence": 0.95}), pid_ok),
        )
        conn.execute(
            "UPDATE processing_pipelines SET book_profile=? WHERE id=?",
            (json.dumps({"book_type": "unknown", "confidence": 0.0}), pid_bad),
        )
        conn.commit()
        conn.close()

        fixed, skipped = fix_backfill_book_types(db_path, check_null_book_type(db_path))

        assert fixed == 1
        assert len(skipped) == 1 and skipped[0]["book_id"] == pid_bad
        conn = _connect(db_path)
        row = conn.execute(
            """SELECT book_type, classification_confidence, classified_by
               FROM books WHERE id = ?""",
            (pid_ok,),
        ).fetchone()
        conn.close()
        assert row["book_type"] == "travel_guide"
        assert row["classification_confidence"] == 0.95
        assert row["classified_by"] == "backfill:doctor"


class TestApplyFixes:
    def _seed_everything(self, db_path, tmp_path):
        """One violation of each category, all fixable."""
        src = tmp_path / "lost.epub"
        src.write_bytes(b"book")
        chfile = tmp_path / "ch.md"
        chfile.write_text("prose")
        # NOTE (adjustment): _seed_complete_pipeline's content_hash defaults to
        # f"hash-{pipeline_id or 'x'}"; two calls with the default None both
        # collide on "hash-x" against processing_pipelines' UNIQUE(content_hash).
        # Pass distinct pipeline_id values, as other multi-seed tests in this
        # file already do.
        lost_pid = _seed_complete_pipeline(db_path, pipeline_id="lost", source_path=str(src))
        typed_pid = _seed_complete_pipeline(db_path, pipeline_id="typed", source_path="/w/typed.epub")
        conn = _connect(db_path)
        _seed_chunk(conn, chunk_id="orph", chapter_id="GONE", book_id=lost_pid)
        _seed_book(conn, book_id=typed_pid, book_type=None)
        _seed_chapter(conn, chapter_id="nohash", book_id=typed_pid, content_hash=None, file_path=str(chfile))
        conn.execute(
            "UPDATE processing_pipelines SET book_profile=? WHERE id=?",
            (json.dumps({"book_type": "textbook", "confidence": 0.8}), typed_pid),
        )
        conn.commit()
        conn.close()
        return lost_pid

    def test_full_fix_then_clean_then_idempotent(self, db_path, tmp_path):
        from agentic_pipeline.health.doctor import apply_fixes, has_violations, run_checks

        self._seed_everything(db_path, tmp_path)

        report = apply_fixes(db_path)

        assert report.backup_path is not None and Path(report.backup_path).exists()
        assert report.manifest_path is not None
        assert report.fixed["orphaned_chunks"] == 1
        assert report.fixed["lost_books"] == 1  # archived
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
        conn.commit()
        conn.close()

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


class TestDoctorCli:
    def _run(self, db_path, monkeypatch, *args):
        from agentic_pipeline.cli import main

        monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(db_path))
        return CliRunner().invoke(main, ["doctor", *args])

    def test_report_clean_exits_zero(self, db_path, monkeypatch):
        result = self._run(db_path, monkeypatch)
        assert result.exit_code == 0
        assert "OK" in result.output or "0" in result.output

    def test_report_violations_exit_one(self, db_path, monkeypatch):
        conn = _connect(db_path)
        _seed_chunk(conn, chunk_id="o", chapter_id="GONE", book_id="GONE")
        conn.commit()
        conn.close()

        result = self._run(db_path, monkeypatch)

        assert result.exit_code == 1
        assert "orphaned_chunks" in result.output

    def test_fix_repairs_and_exits_zero(self, db_path, monkeypatch):
        conn = _connect(db_path)
        _seed_chunk(conn, chunk_id="o", chapter_id="GONE", book_id="GONE")
        conn.commit()
        conn.close()

        result = self._run(db_path, monkeypatch, "--fix", "--no-backup")

        assert result.exit_code == 0
        assert self._run(db_path, monkeypatch).exit_code == 0  # now clean

    def test_fix_prints_reingest_commands_and_manifest_path(self, db_path, tmp_path, monkeypatch):
        src = tmp_path / "lost.epub"
        src.write_bytes(b"x")
        _seed_complete_pipeline(db_path, source_path=str(src))

        result = self._run(db_path, monkeypatch, "--fix", "--no-backup", "--manifest", str(tmp_path / "m.md"))

        assert result.exit_code == 0
        assert "agentic-pipeline reingest" in result.output
        assert str(tmp_path / "m.md") in result.output


class TestHealthIntegration:
    def _run_health(self, db_path, monkeypatch, *args):
        from agentic_pipeline.cli import main

        monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(db_path))
        return CliRunner().invoke(main, ["health", *args])

    def test_health_reports_integrity_ok(self, db_path, monkeypatch):
        result = self._run_health(db_path, monkeypatch)
        assert "integrity: OK" in result.output
        assert result.exit_code == 0

    def test_health_reports_issue_count_and_stays_exit_zero(self, db_path, monkeypatch):
        conn = _connect(db_path)
        _seed_chunk(conn, chunk_id="o", chapter_id="GONE", book_id="GONE")
        conn.commit()
        conn.close()

        result = self._run_health(db_path, monkeypatch)

        assert "integrity: 1 issue" in result.output
        assert "doctor" in result.output  # points at the tool
        assert result.exit_code == 0  # dashboard, not gate

    def test_health_json_carries_integrity_block(self, db_path, monkeypatch):
        conn = _connect(db_path)
        _seed_chunk(conn, chunk_id="o", chapter_id="GONE", book_id="GONE")
        conn.commit()
        conn.close()

        result = self._run_health(db_path, monkeypatch, "--json")
        payload = json.loads(result.output)

        assert payload["integrity"]["issues"] == 1
        assert payload["integrity"]["categories"]["orphaned_chunks"] == 1


class TestDoctorReportMcp:
    def test_caps_samples_and_flags_truncation(self, db_path):
        from agentic_pipeline.mcp_server import doctor_report

        conn = _connect(db_path)
        for i in range(15):
            _seed_chunk(conn, chunk_id=f"o{i}", chapter_id="GONE", book_id="GONE", chunk_index=i)
        conn.commit()
        conn.close()

        payload = doctor_report(db_path=str(db_path))

        oc = payload["orphaned_chunks"]
        assert oc["count"] == 15
        assert len(oc["samples"]) == 10
        assert oc["truncated"] is True

    def test_lost_books_returns_all_details(self, db_path):
        from agentic_pipeline.mcp_server import doctor_report

        for i in range(12):
            _seed_complete_pipeline(db_path, pipeline_id=str(i), source_path=f"/nowhere/book{i}.epub")

        lb = doctor_report(db_path=str(db_path))["lost_books"]

        assert lb["count"] == 12
        assert len(lb["samples"]) == 12  # the actionable list, uncapped
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

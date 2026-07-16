"""library_meta.data_version migration and reader."""

import sqlite3

import pytest

from agentic_pipeline.db.migrations import run_migrations


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    db = tmp_path / "library.db"
    monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(db))
    monkeypatch.setenv("BOOK_DB_PATH", str(db))
    # Config.DB_PATH resolved BOOK_DB_PATH at import time — patch the class attr
    monkeypatch.setattr("src.config.Config.DB_PATH", db)
    return db


class TestMigration:
    def test_migration_creates_seeded_library_meta(self, tmp_db):
        run_migrations(tmp_db)
        conn = sqlite3.connect(tmp_db)
        row = conn.execute("SELECT data_version FROM library_meta WHERE id = 1").fetchone()
        conn.close()
        assert row == (1,)

    def test_migration_idempotent_does_not_reset_version(self, tmp_db):
        run_migrations(tmp_db)
        conn = sqlite3.connect(tmp_db)
        conn.execute("UPDATE library_meta SET data_version = 7")
        conn.commit()
        conn.close()
        run_migrations(tmp_db)
        conn = sqlite3.connect(tmp_db)
        row = conn.execute("SELECT data_version FROM library_meta WHERE id = 1").fetchone()
        conn.close()
        assert row == (7,)


class TestReader:
    def test_reader_returns_version(self, tmp_db):
        run_migrations(tmp_db)
        from src.utils.data_version import get_data_version

        assert get_data_version() == 1

    def test_reader_returns_none_when_table_missing(self, tmp_db):
        # create an empty DB with no migration
        sqlite3.connect(tmp_db).close()
        from src.utils.data_version import get_data_version

        assert get_data_version() is None

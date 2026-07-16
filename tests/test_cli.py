"""Tests for CLI commands."""

import pytest
from click.testing import CliRunner


def test_version_command():
    from agentic_pipeline.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["version"])

    assert result.exit_code == 0
    assert "agentic-pipeline v" in result.output


def test_strategies_command():
    from agentic_pipeline.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["strategies"])

    assert result.exit_code == 0
    assert "technical_tutorial_v1" in result.output


def test_status_shows_rejection_reason(tmp_path, monkeypatch):
    import sqlite3

    from click.testing import CliRunner

    from agentic_pipeline.cli import main
    from agentic_pipeline.db.migrations import run_migrations

    db = tmp_path / "pipeline.db"
    monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(db))
    run_migrations(db)

    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO processing_pipelines (id, source_path, content_hash, state) "
        "VALUES ('pipe-1', '/tmp/x.epub', 'hash-1', 'rejected')"
    )
    conn.execute(
        "INSERT INTO approval_audit (book_id, pipeline_id, action, actor, reason) "
        "VALUES ('', 'pipe-1', 'rejected', 'human:cli', 'duplicate of live copy')"
    )
    conn.commit()
    conn.close()

    result = CliRunner().invoke(main, ["status", "pipe-1"])
    assert result.exit_code == 0
    assert "human:cli" in result.output
    assert "duplicate of live copy" in result.output


def test_status_reason_with_bracket_does_not_crash(tmp_path, monkeypatch):
    """A reason containing Rich-markup-like brackets must not crash status."""
    import sqlite3

    from click.testing import CliRunner

    from agentic_pipeline.cli import main
    from agentic_pipeline.db.migrations import run_migrations

    db = tmp_path / "pipeline.db"
    monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(db))
    run_migrations(db)

    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO processing_pipelines (id, source_path, content_hash, state) "
        "VALUES ('pipe-b', '/tmp/b.epub', 'hash-b', 'rejected')"
    )
    conn.execute(
        "INSERT INTO approval_audit (book_id, pipeline_id, action, actor, reason) "
        "VALUES ('', 'pipe-b', 'rejected', 'human:cli', 'dup — see issue [/red] and [#42]')"
    )
    conn.commit()
    conn.close()

    result = CliRunner().invoke(main, ["status", "pipe-b"])
    assert result.exit_code == 0, result.output
    assert result.exception is None
    assert "[#42]" in result.output


def test_status_null_reason_renders_empty(tmp_path, monkeypatch):
    import sqlite3

    from click.testing import CliRunner

    from agentic_pipeline.cli import main
    from agentic_pipeline.db.migrations import run_migrations

    db = tmp_path / "pipeline.db"
    monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(db))
    run_migrations(db)

    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO processing_pipelines (id, source_path, content_hash, state) "
        "VALUES ('pipe-n', '/tmp/n.epub', 'hash-n', 'rejected')"
    )
    conn.execute(
        "INSERT INTO approval_audit (book_id, pipeline_id, action, actor, reason) "
        "VALUES ('', 'pipe-n', 'rejected', 'human:cli', NULL)"
    )
    conn.commit()
    conn.close()

    result = CliRunner().invoke(main, ["status", "pipe-n"])
    assert result.exit_code == 0, result.output
    assert "None" not in result.output


def test_status_no_reason_line_when_not_rejected(tmp_path, monkeypatch):
    import sqlite3

    from click.testing import CliRunner

    from agentic_pipeline.cli import main
    from agentic_pipeline.db.migrations import run_migrations

    db = tmp_path / "pipeline.db"
    monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(db))
    run_migrations(db)

    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO processing_pipelines (id, source_path, content_hash, state) "
        "VALUES ('pipe-2', '/tmp/y.epub', 'hash-2', 'complete')"
    )
    conn.commit()
    conn.close()

    result = CliRunner().invoke(main, ["status", "pipe-2"])
    assert result.exit_code == 0
    assert "Rejected by" not in result.output

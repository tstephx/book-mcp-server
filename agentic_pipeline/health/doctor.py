"""Integrity doctor — detect and repair library/pipeline drift.

Every check is a query that CAN return violations; the fixes reuse the
checks' SQL fragments so detector and repairer cannot drift apart.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from agentic_pipeline.agents.classifier_types import BookType
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
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks WHERE book_id = ?", (r["id"],)).fetchone()[0]
            sample_row = conn.execute(
                "SELECT content FROM chunks WHERE book_id = ? ORDER BY chunk_index LIMIT 1",
                (r["id"],),
            ).fetchone()
            sample = (sample_row["content"] or "")[:200] if sample_row else ""
            escaped_basename = basename.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            live_copy = bool(
                basename
                and conn.execute(
                    """SELECT 1 FROM processing_pipelines p2
                       JOIN books b ON b.id = p2.id
                       WHERE p2.source_path LIKE ? ESCAPE '\\' AND p2.id != ? LIMIT 1""",
                    (f"%{escaped_basename}", r["id"]),
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
        details.append({"chapter_id": r["chapter_id"], "file_path": r["file_path"], "file_exists": file_exists})
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
                parsed = json.loads(r["book_profile"])
                profile = parsed if isinstance(parsed, dict) else {}
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
        f"Total lost books: {lost_books.count} (re-ingestable: {len(recoverable)}, source gone: {len(gone)})",
        "",
        "## Re-ingestable (source file found)",
        "",
    ]
    for d in recoverable:
        lines += [
            f"- **{d['basename']}** — {d['chunk_count']} orphaned chunks, pipeline `{d['pipeline_id']}`",
            f"  - file: `{d['resolved_path']}`",
            f"  - sample: {d['sample'][:200]!r}",
        ]
    lines += ["", "## Source gone (re-acquire, then drop into the watch dir)", ""]
    for d in gone:
        lines += [
            f"- **{d['basename']}** — {d['chunk_count']} orphaned chunks, pipeline `{d['pipeline_id']}`",
            f"  - last known path: `{d['source_path']}`",
            f"  - sample: {d['sample'][:200]!r}",
        ]
    manifest_path.write_text("\n".join(lines) + "\n")
    return manifest_path

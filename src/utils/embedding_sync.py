"""
Incremental embedding synchronization

Detects content changes and regenerates only affected embeddings.
Uses file mtime as fast path and content hash for verification.
"""

import hashlib
import io
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..config import Config
from ..database import get_db_connection
from .cache import get_cache
from .file_utils import resolve_chapter_path, read_chapter_content

logger = logging.getLogger(__name__)


def compute_content_hash(content: str) -> str:
    """Compute SHA-256 hash of chapter content

    Args:
        content: Chapter text content

    Returns:
        Hex-encoded SHA-256 hash
    """
    return hashlib.sha256(content.encode()).hexdigest()


def get_file_mtime(file_path: str | Path) -> float:
    """Get file modification time, handling split chapters

    Args:
        file_path: Path to chapter file or directory

    Returns:
        Modification time as float (0 if not found)
    """
    try:
        path = resolve_chapter_path(file_path)

        if path.is_dir():
            # For split chapters, use latest mtime of parts
            mtimes = [p.stat().st_mtime for p in path.glob('*.md')]
            return max(mtimes) if mtimes else 0
        elif path.is_file():
            return path.stat().st_mtime
        else:
            return 0
    except Exception:
        return 0


def find_chapters_needing_update(force: bool = False) -> list[dict]:
    """Find chapters that need embedding updates

    Detection logic:
    1. No embedding exists → needs update (reason: 'new')
    2. No tracking data (hash/mtime NULL) → needs update (reason: 'no_tracking')
    3. File mtime changed → read content and compare hash
       - Hash different → needs update (reason: 'modified')
       - Hash same → update mtime only (no embedding regeneration)

    Args:
        force: If True, return all chapters regardless of tracking

    Returns:
        List of chapter dicts with: id, file_path, title, chapter_number, reason
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        if force:
            # Force mode: return all chapters
            cursor.execute("""
                SELECT id, file_path, title, chapter_number
                FROM chapters
                ORDER BY id
            """)
            rows = cursor.fetchall()
            return [
                {
                    'id': row['id'],
                    'file_path': row['file_path'],
                    'title': row['title'],
                    'chapter_number': row['chapter_number'],
                    'reason': 'forced'
                }
                for row in rows
            ]

        # Get all chapters with their tracking data
        cursor.execute("""
            SELECT id, file_path, title, chapter_number,
                   embedding, content_hash, file_mtime
            FROM chapters
            ORDER BY id
        """)
        rows = cursor.fetchall()

    chapters_needing_update = []
    mtime_updates = []  # Chapters where only mtime needs updating

    for row in rows:
        chapter_id = row['id']
        file_path = row['file_path']
        stored_hash = row['content_hash']
        stored_mtime = row['file_mtime']

        # Case 1: No embedding
        if row['embedding'] is None:
            chapters_needing_update.append({
                'id': chapter_id,
                'file_path': file_path,
                'title': row['title'],
                'chapter_number': row['chapter_number'],
                'reason': 'new'
            })
            continue

        # Case 2: No tracking data
        if stored_hash is None or stored_mtime is None:
            chapters_needing_update.append({
                'id': chapter_id,
                'file_path': file_path,
                'title': row['title'],
                'chapter_number': row['chapter_number'],
                'reason': 'no_tracking'
            })
            continue

        # Case 3: Check if file changed
        current_mtime = get_file_mtime(file_path)

        if current_mtime > stored_mtime:
            # File was modified, verify with hash
            try:
                content = read_chapter_content(file_path)
                current_hash = compute_content_hash(content)

                if current_hash != stored_hash:
                    # Content actually changed
                    chapters_needing_update.append({
                        'id': chapter_id,
                        'file_path': file_path,
                        'title': row['title'],
                        'chapter_number': row['chapter_number'],
                        'reason': 'modified'
                    })
                else:
                    # File touched but content unchanged, just update mtime
                    mtime_updates.append((current_mtime, chapter_id))
            except Exception as e:
                logger.warning(f"Error checking chapter {chapter_id}: {e}")
                # If we can't read it, mark for update to surface the error
                chapters_needing_update.append({
                    'id': chapter_id,
                    'file_path': file_path,
                    'title': row['title'],
                    'chapter_number': row['chapter_number'],
                    'reason': 'error'
                })

    # Update mtimes for unchanged content
    if mtime_updates:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                "UPDATE chapters SET file_mtime = ? WHERE id = ?",
                mtime_updates
            )
            conn.commit()
        logger.debug(f"Updated mtime for {len(mtime_updates)} unchanged chapters")

    return chapters_needing_update


def update_embeddings_incremental(
    batch_size: int = 50,
    force: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> dict:
    """Incrementally update embeddings for changed chapters

    Args:
        batch_size: Number of chapters to process per batch
        force: If True, regenerate all embeddings
        progress_callback: Optional callback(current, total) for progress

    Returns:
        Result dict with: status, updated, skipped, errors, duration_seconds
    """
    start_time = time.time()

    # Find chapters needing update
    chapters_to_update = find_chapters_needing_update(force=force)

    if not chapters_to_update:
        return {
            'status': 'no_updates_needed',
            'updated': 0,
            'skipped': 0,
            'errors': 0,
            'duration_seconds': round(time.time() - start_time, 2)
        }

    logger.info(f"Found {len(chapters_to_update)} chapters needing embedding update")

    # Import here to avoid circular imports and defer model loading
    from .openai_embeddings import OpenAIEmbeddingGenerator

    generator = OpenAIEmbeddingGenerator()

    updated = 0
    errors = 0
    now = datetime.now().isoformat()

    # Process in batches
    for i in range(0, len(chapters_to_update), batch_size):
        batch = chapters_to_update[i:i + batch_size]

        # Read content for batch
        batch_contents = []
        batch_valid = []

        for chapter in batch:
            try:
                content = read_chapter_content(chapter['file_path'])
                batch_contents.append(content)
                batch_valid.append(chapter)
            except Exception as e:
                logger.warning(f"Error reading chapter {chapter['id']}: {e}")
                errors += 1

        if not batch_contents:
            continue

        # Generate embeddings for batch
        try:
            embeddings = generator.generate_batch(batch_contents, batch_size=batch_size)
        except Exception as e:
            logger.error(f"Error generating embeddings for batch: {e}")
            errors += len(batch_valid)
            continue

        # Store embeddings in database
        with get_db_connection() as conn:
            cursor = conn.cursor()

            for chapter, content, embedding in zip(batch_valid, batch_contents, embeddings):
                try:
                    # Serialize embedding
                    embedding_blob = io.BytesIO()
                    np.save(embedding_blob, embedding)
                    embedding_bytes = embedding_blob.getvalue()

                    # Compute tracking data
                    content_hash = compute_content_hash(content)
                    file_mtime = get_file_mtime(chapter['file_path'])

                    # Update database
                    cursor.execute("""
                        UPDATE chapters
                        SET embedding = ?,
                            embedding_model = ?,
                            content_hash = ?,
                            file_mtime = ?,
                            embedding_updated_at = ?
                        WHERE id = ?
                    """, (
                        embedding_bytes,
                        'text-embedding-3-large',
                        content_hash,
                        file_mtime,
                        now,
                        chapter['id']
                    ))

                    updated += 1

                except Exception as e:
                    logger.warning(f"Error storing embedding for chapter {chapter['id']}: {e}")
                    errors += 1

            conn.commit()

        # Progress callback
        if progress_callback:
            progress_callback(min(i + batch_size, len(chapters_to_update)), len(chapters_to_update))

        logger.info(f"Progress: {updated}/{len(chapters_to_update)} chapters updated")

    # Invalidate embeddings cache
    cache = get_cache()
    cache.invalidate_embeddings()
    logger.info("Embeddings cache invalidated")

    duration = round(time.time() - start_time, 2)

    # Count skipped (total chapters - updated - errors)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM chapters")
        total_chapters = cursor.fetchone()['count']

    skipped = total_chapters - updated - errors

    result = {
        'status': 'updated',
        'updated': updated,
        'skipped': skipped,
        'errors': errors,
        'duration_seconds': duration
    }

    logger.info(f"Embedding sync complete: {result}")

    return result

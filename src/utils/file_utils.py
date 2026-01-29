"""
File utilities for reading chapter content

Handles both single-file chapters and split chapters (directories with parts).
Split chapters are used when content exceeds token limits for embedding generation.
Integrates with cache for performance.
"""

import logging
from pathlib import Path
from typing import Optional

from ..config import Config
from .cache import get_cache

logger = logging.getLogger(__name__)


def resolve_chapter_path(file_path: str | Path) -> Path:
    """Resolve a chapter file path from database format to absolute path

    Database stores relative paths like 'data/books/book-id/chapters/01-chapter.md'
    This resolves them to the actual file location using Config.BOOKS_DIR.

    Args:
        file_path: Path from database (relative or absolute)

    Returns:
        Resolved absolute Path object
    """
    path = Path(file_path)

    if path.is_absolute():
        return path

    # Strip 'data/books/' prefix if present and resolve relative to BOOKS_DIR
    try:
        relative_path = path.relative_to('data/books')
        return Config.BOOKS_DIR / relative_path
    except ValueError:
        # Path doesn't start with 'data/books/', try as-is from BOOKS_DIR
        return Config.BOOKS_DIR / path


def read_chapter_content(file_path: str | Path) -> str:
    """Read chapter content, handling both single files and split chapters

    Split chapters are directories containing numbered .md files that should
    be concatenated in order. This handles the following cases:

    1. Single file exists (e.g., 01-chapter.md) -> read directly
    2. Directory exists (e.g., 01-chapter/) -> concatenate parts in order
    3. Neither exists but removing .md gives a directory -> concatenate parts

    Uses cache when enabled for improved performance on repeated reads.

    Args:
        file_path: Path to chapter file (from database or resolved)

    Returns:
        Full chapter content as string

    Raises:
        FileNotFoundError: If chapter cannot be found
        IOError: If chapter cannot be read
    """
    cache = get_cache()
    cache_key = str(file_path)

    # Check cache first
    cached_content = cache.get_chapter(cache_key)
    if cached_content is not None:
        return cached_content

    path = resolve_chapter_path(file_path)

    # Determine the actual path and get mtime for cache
    actual_path = path
    if path.is_file():
        logger.debug(f"Reading single-file chapter: {path}")
        content = path.read_text(encoding='utf-8')
    elif path.is_dir():
        content = _read_split_chapter(path)
        actual_path = path
    elif path.suffix == '.md':
        dir_path = path.with_suffix('')
        if dir_path.is_dir():
            content = _read_split_chapter(dir_path)
            actual_path = dir_path
        else:
            raise FileNotFoundError(f"Chapter not found: {path}")
    else:
        raise FileNotFoundError(f"Chapter not found: {path}")

    # Get mtime for cache validation
    try:
        mtime = actual_path.stat().st_mtime
    except OSError:
        mtime = 0

    # Store in cache
    cache.set_chapter(cache_key, content, mtime)

    return content


def _read_split_chapter(dir_path: Path) -> str:
    """Read and concatenate parts of a split chapter directory

    Expects parts to be named with numeric prefixes (01-intro.md, 02-content.md)
    and concatenates them in sorted order. Skips _index.md files.

    Args:
        dir_path: Path to split chapter directory

    Returns:
        Concatenated content from all parts
    """
    # Find all numbered .md files (exclude _index.md)
    parts = sorted([
        p for p in dir_path.glob('[0-9]*.md')
        if not p.name.startswith('_')
    ])

    if not parts:
        # Fallback: try any .md files except _index.md
        parts = sorted([
            p for p in dir_path.glob('*.md')
            if not p.name.startswith('_')
        ])

    if not parts:
        raise FileNotFoundError(f"No chapter parts found in: {dir_path}")

    logger.debug(f"Reading split chapter with {len(parts)} parts: {dir_path}")

    content_parts = []
    for part in parts:
        try:
            content_parts.append(part.read_text(encoding='utf-8'))
        except Exception as e:
            logger.warning(f"Could not read chapter part {part}: {e}")

    return '\n\n'.join(content_parts)


def get_chapter_excerpt(file_path: str | Path, max_chars: int = 300) -> str:
    """Get a short excerpt from a chapter

    Convenience function for search results and previews.

    Args:
        file_path: Path to chapter file
        max_chars: Maximum characters to return (default: 300)

    Returns:
        Chapter excerpt with ellipsis if truncated, or error message
    """
    try:
        content = read_chapter_content(file_path)
        excerpt = content[:max_chars].strip()
        if len(content) > max_chars:
            excerpt += "..."
        return excerpt
    except Exception as e:
        logger.warning(f"Could not read chapter excerpt: {e}")
        return "[Content not available]"

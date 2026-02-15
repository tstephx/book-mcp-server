"""
Two-tier caching system for Book MCP Server

Tier 1: Embeddings Matrix - loaded once, invalidated manually
Tier 2: Chapter Content - TTL-based with file mtime validation
"""

import logging
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CachedChapter:
    """Cached chapter content with metadata for validation"""
    content: str
    mtime: float
    expires_at: float
    size_bytes: int


@dataclass
class CachedEmbeddings:
    """Cached embeddings matrix with metadata"""
    matrix: np.ndarray
    metadata: list[dict]
    loaded_at: float


class LibraryCache:
    """Two-tier in-memory cache for book library data

    Tier 1: Embeddings matrix (load once, manual invalidation)
    Tier 2: Chapter content (TTL-based with file mtime check)
    """

    def __init__(self, enabled: bool = True, chapter_ttl: int = 3600):
        """Initialize cache

        Args:
            enabled: Whether caching is enabled
            chapter_ttl: Time-to-live for chapter content in seconds (default: 1 hour)
        """
        self.enabled = enabled
        self.chapter_ttl = chapter_ttl

        # Thread-safe storage
        self._lock = threading.RLock()
        self._chapters: dict[str, CachedChapter] = {}
        self._embeddings: Optional[CachedEmbeddings] = None
        self._chunk_embeddings: Optional[CachedEmbeddings] = None
        self._summary_embeddings: Optional[CachedEmbeddings] = None

        # Stats
        self._hits = 0
        self._misses = 0

        logger.info(f"LibraryCache initialized (enabled={enabled}, chapter_ttl={chapter_ttl}s)")

    # ─────────────────────────────────────────────────────────────
    # Tier 1: Embeddings Cache
    # ─────────────────────────────────────────────────────────────

    def get_embeddings(self) -> Optional[tuple[np.ndarray, list[dict]]]:
        """Get cached embeddings matrix and metadata

        Returns:
            Tuple of (embeddings_matrix, metadata_list) or None if not cached
        """
        if not self.enabled:
            return None

        with self._lock:
            if self._embeddings is not None:
                self._hits += 1
                logger.debug("Embeddings cache hit")
                return self._embeddings.matrix, self._embeddings.metadata

            self._misses += 1
            return None

    def set_embeddings(self, matrix: np.ndarray, metadata: list[dict]) -> None:
        """Cache embeddings matrix and metadata

        Args:
            matrix: Numpy array of shape (n_chapters, embedding_dim)
            metadata: List of dicts with chapter info (book_title, chapter_title, etc.)
        """
        if not self.enabled:
            return

        with self._lock:
            self._embeddings = CachedEmbeddings(
                matrix=matrix,
                metadata=metadata,
                loaded_at=time()
            )
            logger.info(f"Cached embeddings matrix: {matrix.shape[0]} chapters, "
                       f"{matrix.nbytes / 1024 / 1024:.1f} MB")

    def invalidate_embeddings(self) -> None:
        """Invalidate embeddings cache (call when books are added/removed)"""
        with self._lock:
            if self._embeddings is not None:
                logger.info("Invalidating embeddings cache")
                self._embeddings = None

    # ─────────────────────────────────────────────────────────────
    # Tier 1a: Chunk Embeddings Cache (separate from chapter embeddings)
    # ─────────────────────────────────────────────────────────────

    def get_chunk_embeddings(self) -> Optional[tuple[np.ndarray, list[dict]]]:
        """Get cached chunk embeddings matrix and metadata"""
        if not self.enabled:
            return None

        with self._lock:
            if self._chunk_embeddings is not None:
                self._hits += 1
                logger.debug("Chunk embeddings cache hit")
                return self._chunk_embeddings.matrix, self._chunk_embeddings.metadata

            self._misses += 1
            return None

    def set_chunk_embeddings(self, matrix: np.ndarray, metadata: list[dict]) -> None:
        """Cache chunk embeddings matrix and metadata"""
        if not self.enabled:
            return

        with self._lock:
            self._chunk_embeddings = CachedEmbeddings(
                matrix=matrix,
                metadata=metadata,
                loaded_at=time()
            )
            logger.info(f"Cached chunk embeddings: {matrix.shape[0]} chunks, "
                       f"{matrix.nbytes / 1024 / 1024:.1f} MB")

    def invalidate_chunk_embeddings(self) -> None:
        """Invalidate chunk embeddings cache"""
        with self._lock:
            if self._chunk_embeddings is not None:
                logger.info("Invalidating chunk embeddings cache")
                self._chunk_embeddings = None

    # ─────────────────────────────────────────────────────────────
    # Tier 1b: Summary Embeddings Cache
    # ─────────────────────────────────────────────────────────────

    def get_summary_embeddings(self) -> Optional[tuple[np.ndarray, list[dict]]]:
        """Get cached summary embeddings matrix and metadata

        Returns:
            Tuple of (embeddings_matrix, metadata_list) or None if not cached
        """
        if not self.enabled:
            return None

        with self._lock:
            if self._summary_embeddings is not None:
                self._hits += 1
                logger.debug("Summary embeddings cache hit")
                return self._summary_embeddings.matrix, self._summary_embeddings.metadata

            self._misses += 1
            return None

    def set_summary_embeddings(self, matrix: np.ndarray, metadata: list[dict]) -> None:
        """Cache summary embeddings matrix and metadata

        Args:
            matrix: Numpy array of shape (n_summaries, embedding_dim)
            metadata: List of dicts with chapter info
        """
        if not self.enabled:
            return

        with self._lock:
            self._summary_embeddings = CachedEmbeddings(
                matrix=matrix,
                metadata=metadata,
                loaded_at=time()
            )
            logger.info(f"Cached summary embeddings: {matrix.shape[0]} summaries, "
                       f"{matrix.nbytes / 1024 / 1024:.1f} MB")

    def invalidate_summary_embeddings(self) -> None:
        """Invalidate summary embeddings cache"""
        with self._lock:
            if self._summary_embeddings is not None:
                logger.info("Invalidating summary embeddings cache")
                self._summary_embeddings = None

    # ─────────────────────────────────────────────────────────────
    # Tier 2: Chapter Content Cache
    # ─────────────────────────────────────────────────────────────

    def get_chapter(self, file_path: str | Path) -> Optional[str]:
        """Get cached chapter content if valid

        Validates both TTL expiration and file modification time.

        Args:
            file_path: Path to chapter file

        Returns:
            Chapter content string or None if cache miss/invalid
        """
        if not self.enabled:
            return None

        key = str(file_path)

        with self._lock:
            cached = self._chapters.get(key)

            if cached is None:
                self._misses += 1
                return None

            now = time()

            # Check TTL expiration
            if now > cached.expires_at:
                logger.debug(f"Chapter cache expired: {key}")
                del self._chapters[key]
                self._misses += 1
                return None

            # Check file modification time
            try:
                path = Path(file_path)
                # Handle split chapters (directory)
                if path.suffix == '.md' and not path.exists():
                    dir_path = path.with_suffix('')
                    if dir_path.is_dir():
                        # Check mtime of directory (changes when files added/modified)
                        current_mtime = dir_path.stat().st_mtime
                    else:
                        current_mtime = 0
                elif path.is_dir():
                    current_mtime = path.stat().st_mtime
                elif path.exists():
                    current_mtime = path.stat().st_mtime
                else:
                    current_mtime = 0

                if current_mtime > cached.mtime:
                    logger.debug(f"Chapter file modified, invalidating: {key}")
                    del self._chapters[key]
                    self._misses += 1
                    return None

            except OSError:
                # Can't check mtime, invalidate to be safe
                del self._chapters[key]
                self._misses += 1
                return None

            self._hits += 1
            logger.debug(f"Chapter cache hit: {key}")
            return cached.content

    def set_chapter(self, file_path: str | Path, content: str, mtime: float) -> None:
        """Cache chapter content

        Args:
            file_path: Path to chapter file (used as cache key)
            content: Chapter content string
            mtime: File modification time for validation
        """
        if not self.enabled:
            return

        key = str(file_path)

        with self._lock:
            self._chapters[key] = CachedChapter(
                content=content,
                mtime=mtime,
                expires_at=time() + self.chapter_ttl,
                size_bytes=len(content.encode('utf-8'))
            )
            logger.debug(f"Cached chapter: {key} ({len(content)} chars)")

    def invalidate_chapter(self, file_path: str | Path) -> None:
        """Invalidate a specific chapter from cache"""
        key = str(file_path)
        with self._lock:
            if key in self._chapters:
                del self._chapters[key]
                logger.debug(f"Invalidated chapter: {key}")

    def clear_chapters(self) -> None:
        """Clear all cached chapters"""
        with self._lock:
            count = len(self._chapters)
            self._chapters.clear()
            logger.info(f"Cleared {count} cached chapters")

    # ─────────────────────────────────────────────────────────────
    # Stats & Management
    # ─────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Get cache statistics

        Returns:
            Dictionary with cache stats for monitoring
        """
        with self._lock:
            chapter_memory = sum(c.size_bytes for c in self._chapters.values())
            embeddings_memory = (
                self._embeddings.matrix.nbytes if self._embeddings else 0
            )
            chunk_embeddings_memory = (
                self._chunk_embeddings.matrix.nbytes if self._chunk_embeddings else 0
            )
            summary_embeddings_memory = (
                self._summary_embeddings.matrix.nbytes if self._summary_embeddings else 0
            )

            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
            total_memory = chapter_memory + embeddings_memory + chunk_embeddings_memory + summary_embeddings_memory

            return {
                "enabled": self.enabled,
                "chapters_cached": len(self._chapters),
                "embeddings_loaded": self._embeddings is not None,
                "embeddings_chapters": (
                    self._embeddings.matrix.shape[0] if self._embeddings else 0
                ),
                "chunk_embeddings_loaded": self._chunk_embeddings is not None,
                "chunk_embeddings_count": (
                    self._chunk_embeddings.matrix.shape[0] if self._chunk_embeddings else 0
                ),
                "summary_embeddings_loaded": self._summary_embeddings is not None,
                "summary_embeddings_count": (
                    self._summary_embeddings.matrix.shape[0] if self._summary_embeddings else 0
                ),
                "memory_mb": round(total_memory / 1024 / 1024, 2),
                "chapter_memory_mb": round(chapter_memory / 1024 / 1024, 2),
                "embeddings_memory_mb": round(embeddings_memory / 1024 / 1024, 2),
                "summary_embeddings_memory_mb": round(summary_embeddings_memory / 1024 / 1024, 2),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percent": round(hit_rate, 1),
                "chapter_ttl_seconds": self.chapter_ttl
            }

    def clear_all(self) -> None:
        """Clear all caches"""
        with self._lock:
            self._chapters.clear()
            self._embeddings = None
            self._chunk_embeddings = None
            self._summary_embeddings = None
            self._hits = 0
            self._misses = 0
            logger.info("Cleared all caches")


# Global cache instance - initialized lazily to allow config to load first
_cache: Optional[LibraryCache] = None


def get_cache() -> LibraryCache:
    """Get the global cache instance

    Creates cache on first call using current config values.
    """
    global _cache
    if _cache is None:
        from ..config import Config
        _cache = LibraryCache(
            enabled=Config.ENABLE_CACHING,
            chapter_ttl=getattr(Config, 'CACHE_CHAPTER_TTL', 3600)
        )
    return _cache


# Convenience alias
cache = property(lambda self: get_cache())

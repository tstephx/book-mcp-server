"""LibraryCache data_version self-invalidation."""

import numpy as np

from src.utils.cache import LibraryCache


def _matrix():
    return np.ones((2, 4), dtype=np.float32), [{"chunk_id": "a"}, {"chunk_id": "b"}]


class TestChunkCacheVersioning:
    def test_hit_when_version_unchanged(self):
        cache = LibraryCache(enabled=True)
        m, meta = _matrix()
        cache.set_chunk_embeddings(m, meta, data_version=3)
        assert cache.get_chunk_embeddings(current_version=3) is not None

    def test_miss_and_invalidate_when_version_bumped(self):
        cache = LibraryCache(enabled=True)
        m, meta = _matrix()
        cache.set_chunk_embeddings(m, meta, data_version=3)
        assert cache.get_chunk_embeddings(current_version=4) is None
        # entry dropped: even the legacy no-version call now misses
        assert cache.get_chunk_embeddings() is None

    def test_legacy_calls_keep_working_without_versions(self):
        cache = LibraryCache(enabled=True)
        m, meta = _matrix()
        cache.set_chunk_embeddings(m, meta)
        assert cache.get_chunk_embeddings() is not None

    def test_none_current_version_skips_check(self):
        cache = LibraryCache(enabled=True)
        m, meta = _matrix()
        cache.set_chunk_embeddings(m, meta, data_version=3)
        assert cache.get_chunk_embeddings(current_version=None) is not None

    def test_versioned_entry_vs_unversioned_probe_and_vice_versa(self):
        cache = LibraryCache(enabled=True)
        m, meta = _matrix()
        # unversioned entry + versioned probe -> treated as stale
        cache.set_chunk_embeddings(m, meta)
        assert cache.get_chunk_embeddings(current_version=5) is None

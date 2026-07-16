# Re-Chunking for Retrieval Precision — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the chunker's chapter-passthrough failure (76% of chapters emit one ~2,500-word chunk), prove the fix with a before/after retrieval eval, and swap the re-chunked + re-embedded corpus into production behind an acceptance gate.

**Architecture:** Hierarchical chunker — existing paragraph pipeline plus a sentence-window fallback for segments over 2× target. A gold-query eval harness (semantic-gated, hybrid-reported hit@5/MRR at chapter level) scores live vs. staged. A `rechunk` CLI stages into `chunks_staging` with hash-keyed embedding reuse, and `rechunk --swap` atomically replaces `chunks`, bumping a new `data_version` that the MCP server's cache self-checks.

**Tech Stack:** Python 3.12, SQLite (WAL), numpy, tiktoken, OpenAI text-embedding-3-large, `claude -p` for gold generation, Click CLI, pytest.

**Spec:** `docs/superpowers/specs/2026-07-15-rechunking-design.md` (9 binding interview decisions in §0).

## Global Constraints

- Branch: `feat/rechunking` (create from `main` before Task 1).
- Tests NEVER touch the live library DB. Every test that opens a DB uses a tmp path via `monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(tmp_db))` and/or `monkeypatch.setenv("BOOK_DB_PATH", str(tmp_db))`. All OpenAI and `claude -p` calls are mocked in tests.
- GOTCHA (verified 2026-07-15): `src/config.py` resolves `BOOK_DB_PATH` at import time (`Config.DB_PATH: Path = Path(os.getenv(...))`, line 25). Tests that need `src.database.get_db_connection()` to hit a tmp DB must ALSO `monkeypatch.setattr("src.config.Config.DB_PATH", tmp_db)` — the env var alone is not enough once `src.config` has been imported.
- Chunking parameters, verbatim from spec: `target_words=500`, `overlap_sentences=2`, fallback threshold `2 * target_words` (1,000 words), `min_chunk_words=100` (final short window merges into predecessor).
- Acceptance gate, verbatim from spec (semantic mode only): auto-set hit@5 ≥ baseline AND auto-set MRR ≥ baseline AND manual-set hit@5 strictly > baseline. Hybrid numbers are reported, never gating.
- Verdict/marker persist to `<db-dir>/rechunk/last-verdict.json`. Gold query files live in the repo at `eval/gold-queries.json` and `eval/gold-queries-manual.json`.
- Embedding blobs are `np.save` format (matches `np.load` in `chunk_loader.py`); `embedding_model` column value: `text-embedding-3-large`.
- Staging table name: `chunks_staging`, with `UNIQUE(chapter_id, chunk_index)`. Production `chunks` is never written except inside `--swap`'s transaction.
- `doctor --fix` conventions reused: backups via `create_backup()` from `agentic_pipeline/health/doctor.py`.
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- PostToolUse hook runs ruff format on every .py write — do not fight formatting.
- Do NOT edit `pyproject.toml`, `requirements.txt`, `.db` files, or `.venv/` (PreToolUse hook blocks them; no new dependencies are needed).

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/utils/chunker.py` | modify | `sentence_windows()` + fallback integration in `chunk_chapter()` |
| `src/utils/data_version.py` | create | `get_data_version()` — read `library_meta.data_version`, `None` if absent |
| `src/utils/cache.py` | modify | version-aware chunk-embedding cache get/set |
| `src/utils/chunk_loader.py` | modify | wire `data_version` through load/set |
| `src/utils/retrieval_eval.py` | create | eval core: gold build, matrix load, semantic/hybrid ranking, hit@5/MRR |
| `scripts/retrieval_eval.py` | create | thin CLI wrapper (`build-gold`, `run`) |
| `eval/gold-queries-manual.json` | create | ~10 hand-picked known-miss queries |
| `agentic_pipeline/db/migrations.py` | modify | `library_meta` table + seed row |
| `agentic_pipeline/library/rechunk.py` | create | staging, hash reconcile, embed, gate eval, swap |
| `agentic_pipeline/cli.py` | modify | `rechunk` command |
| `src/tools/semantic_search_tools.py` | modify | per-chapter aggregation (audit finding) |
| `tests/test_chunker.py` | modify | new fallback tests |
| `tests/test_retrieval_eval.py` | create | eval harness tests |
| `tests/test_rechunk.py` | create | staging/reconcile/swap tests |
| `tests/test_cache.py` | modify (create if missing) | data_version self-invalidation tests |

---

### Task 1: `sentence_windows()` in the chunker

**Files:**
- Modify: `src/utils/chunker.py`
- Test: `tests/test_chunker.py`

**Interfaces:**
- Consumes: nothing new (module-local `re`).
- Produces: `sentence_windows(text: str, target_words: int = 500, overlap_sentences: int = 2, min_chunk_words: int = 100) -> list[str]` — Task 2 calls this from `chunk_chapter()`; Task 8's staging relies on its behavior transitively.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_chunker.py`:

```python
from src.utils.chunker import sentence_windows


def _numbered_sentences(n: int) -> str:
    """n sentences, 7 words each, individually identifiable."""
    return " ".join(f"This is numbered sentence {i:04d} padded body." for i in range(1, n + 1))


class TestSentenceWindows:
    def test_wall_of_text_produces_sized_windows(self):
        # ~2,520 words, zero double-newlines
        text = _numbered_sentences(360)
        windows = sentence_windows(text, target_words=500, overlap_sentences=2, min_chunk_words=100)
        assert len(windows) >= 4
        for w in windows:
            assert 300 <= len(w.split()) <= 700, f"window out of range: {len(w.split())} words"

    def test_consecutive_windows_overlap_by_two_sentences(self):
        import re as _re

        def _nums(s):
            return _re.findall(r"numbered sentence (\d{4})", s)

        text = _numbered_sentences(200)
        windows = sentence_windows(text, target_words=500, overlap_sentences=2, min_chunk_words=100)
        assert len(windows) >= 2
        for prev, nxt in zip(windows, windows[1:]):
            # window i+1 opens with the last 2 sentences of window i
            assert _nums(nxt)[:2] == _nums(prev)[-2:]

    def test_short_final_window_merges_into_predecessor(self):
        # 52 sentences x 7 words = 364 words... need >target to split: use target=300
        text = _numbered_sentences(52)
        windows = sentence_windows(text, target_words=300, overlap_sentences=2, min_chunk_words=100)
        # tail after the first window is < 100 new words -> merged, single window
        assert len(windows) == 1
        assert "0052" in windows[0]

    def test_single_giant_sentence_yields_one_window(self):
        text = " ".join(["word"] * 1500)  # no sentence boundaries at all
        windows = sentence_windows(text, target_words=500, overlap_sentences=2, min_chunk_words=100)
        assert len(windows) == 1

    def test_empty_text_returns_empty(self):
        assert sentence_windows("") == []
        assert sentence_windows("   \n  ") == []

    def test_coverage_no_content_lost(self):
        text = _numbered_sentences(300)
        windows = sentence_windows(text, target_words=500, overlap_sentences=2, min_chunk_words=100)
        joined = " ".join(windows)
        for i in range(1, 301):
            assert f"{i:04d}" in joined
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_chunker.py::TestSentenceWindows -v`
Expected: FAIL — `ImportError: cannot import name 'sentence_windows'`

- [ ] **Step 3: Implement `sentence_windows`**

Add to `src/utils/chunker.py` after `_split_at_sentences` (module-level, before `chunk_chapter`):

```python
_SENTENCE_SPLIT = r"(?<=[.!?])\s+"


def sentence_windows(
    text: str,
    target_words: int = 500,
    overlap_sentences: int = 2,
    min_chunk_words: int = 100,
) -> list[str]:
    """Pack sentences into ~target_words windows with sentence overlap.

    Fallback for wall-of-text input where paragraph splitting finds no
    boundaries. Each window after the first starts with the last
    `overlap_sentences` sentences of the previous window, so content at
    window edges is findable from both sides. A final window contributing
    fewer than `min_chunk_words` NEW words merges into its predecessor.
    """
    sentences = [s for s in re.split(_SENTENCE_SPLIT, text.strip()) if s.strip()]
    if not sentences:
        return []

    windows: list[list[str]] = []
    current: list[str] = []
    carried = 0  # sentences at the head of `current` copied from the previous window
    current_words = 0

    for sentence in sentences:
        s_words = len(sentence.split())
        # break only if this window already holds at least one NEW sentence
        if current_words + s_words > target_words and len(current) > carried:
            windows.append(current)
            overlap = current[-overlap_sentences:] if overlap_sentences > 0 else []
            carried = len(overlap)
            current = list(overlap)
            current_words = sum(len(s.split()) for s in current)
        current.append(sentence)
        current_words += s_words

    if len(current) > carried:
        new_sentences = current[carried:]
        new_words = sum(len(s.split()) for s in new_sentences)
        if windows and new_words < min_chunk_words:
            windows[-1].extend(new_sentences)
        else:
            windows.append(current)

    return [" ".join(w) for w in windows]
```

Also update `_split_at_sentences` line 38 to use the shared constant: `sentences = re.split(_SENTENCE_SPLIT, text)` (define `_SENTENCE_SPLIT` above `_split_at_sentences`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_chunker.py -v`
Expected: all PASS (new class + all pre-existing chunker tests).

- [ ] **Step 5: Commit**

```bash
git add src/utils/chunker.py tests/test_chunker.py
git commit -m "feat: add sentence_windows fallback packer to chunker

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Wire the fallback into `chunk_chapter()`

**Files:**
- Modify: `src/utils/chunker.py` (function `chunk_chapter`, currently lines 74–178)
- Test: `tests/test_chunker.py`

**Interfaces:**
- Consumes: `sentence_windows()` from Task 1, existing `_make_chunk(index, paragraphs=None, content=None)`.
- Produces: `chunk_chapter(text, target_words=500, min_chunk_words=100, max_tokens=8000, overlap_sentences=2) -> list[dict]` (new `overlap_sentences` kwarg, default 2). Dict keys unchanged: `chunk_index, content, word_count, token_count`. Tasks 8+ call it with defaults.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_chunker.py`:

```python
from src.utils.chunker import chunk_chapter


def _paragraph(words: int, tag: str) -> str:
    """One paragraph of exactly `words` words ending with a period."""
    body = " ".join(f"{tag}w{i}" for i in range(words - 1))
    return f"{body} end."


class TestChunkChapterFallback:
    def test_wall_of_text_chapter_no_longer_passes_through(self):
        # single giant paragraph, ~2,520 words, no double newlines
        text = _numbered_sentences(360)
        out = chunk_chapter(text)
        assert len(out) >= 4
        for c in out:
            assert c["word_count"] <= 2 * 500, "fallback failed to size a segment"

    def test_well_formatted_chapter_regression_pin(self):
        # 15 paragraphs x 100 words: old algorithm packs 5-para/500-word
        # chunks with one-paragraph overlap, tail 300 words. Must be unchanged.
        text = "\n\n".join(_paragraph(100, f"p{i}") for i in range(15))
        out = chunk_chapter(text)
        assert [c["word_count"] for c in out] == [500, 500, 500, 300]
        assert all("\n\n" in c["content"] for c in out)

    def test_boundary_exactly_double_target_does_not_fall_through(self):
        # one paragraph of exactly 1000 words -> single chunk, no windows
        text = " ".join(f"Sentence {i} has five words." for i in range(200))  # 200*5=1000
        out = chunk_chapter(text)
        assert len(out) == 1
        assert out[0]["word_count"] == 1000

    def test_boundary_one_sentence_over_falls_through(self):
        text = " ".join(f"Sentence {i} has five words." for i in range(201))  # 1005
        out = chunk_chapter(text)
        assert len(out) >= 2

    def test_chunk_indices_sequential_after_fallback(self):
        text = _numbered_sentences(360)
        out = chunk_chapter(text)
        assert [c["chunk_index"] for c in out] == list(range(len(out)))

    def test_max_tokens_guard_still_enforced(self):
        # pathological: no sentence boundaries, force token overflow path
        text = " ".join(["token"] * 20000)
        out = chunk_chapter(text, max_tokens=2000)
        assert all(c["token_count"] <= 2000 for c in out)

    def test_empty_text_still_returns_empty(self):
        assert chunk_chapter("") == []
        assert chunk_chapter("  \n\n  ") == []
```

Note on `test_boundary_exactly_double_target_does_not_fall_through`: 1000 words > `500 * 1.2`, so it takes the paragraph path; `_split_paragraphs` returns one paragraph; the loop emits one 1000-word chunk; `1000 > 2*500` is False → no fallback. The one-over test yields 1005 → fallback fires.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_chunker.py::TestChunkChapterFallback -v`
Expected: `test_wall_of_text_chapter_no_longer_passes_through`, `test_boundary_one_sentence_over_falls_through`, `test_chunk_indices_sequential_after_fallback` FAIL (passthrough produces 1 fat chunk). Regression pin and boundary-exact tests may already pass — that is correct.

- [ ] **Step 3: Integrate the fallback**

In `chunk_chapter`, make two changes.

(a) Signature (line 74):

```python
def chunk_chapter(
    text: str,
    target_words: int = 500,
    min_chunk_words: int = 100,
    max_tokens: int = 8000,
    overlap_sentences: int = 2,
) -> list[dict]:
```

(b) Insert a sizing pass over `chunks` immediately BEFORE the existing
"Post-process: split any chunks that exceed max_tokens" block (currently
line 158), and make the same fallback apply to the two early-return
passthrough paths. The cleanest implementation: replace both early
`return [{...single chunk...}]` bodies and the post-loop with a shared
tail. Restructure the end of the function so ALL paths flow through the
sizing pass:

```python
    # ... existing paragraph loop and tail-merge logic unchanged, ending with:
    #     chunks.append(_make_chunk(len(chunks), current_paragraphs))
    # (for the two early-return paths at lines 97-111 and 114-127, replace
    #  `return [{...}]` single-chunk returns with `chunks = [{...}]` followed
    #  by `return _finalize(chunks, target_words, min_chunk_words, max_tokens,
    #  overlap_sentences)`, and end the paragraph path with the same call.)

def _finalize(
    chunks: list[dict],
    target_words: int,
    min_chunk_words: int,
    max_tokens: int,
    overlap_sentences: int,
) -> list[dict]:
    """Shared tail: oversize fallback, then token guard, then re-index."""
    # Fallback: segments the paragraph pass couldn't size (wall-of-text input)
    sized: list[dict] = []
    for chunk in chunks:
        if chunk["word_count"] > 2 * target_words:
            for content in sentence_windows(
                chunk["content"],
                target_words=target_words,
                overlap_sentences=overlap_sentences,
                min_chunk_words=min_chunk_words,
            ):
                sized.append(_make_chunk(len(sized), None, content))
        else:
            chunk["chunk_index"] = len(sized)
            sized.append(chunk)

    # Hard token ceiling (pre-existing behavior, unchanged)
    final_chunks: list[dict] = []
    for chunk in sized:
        if chunk["token_count"] > max_tokens:
            sub_chunks = _split_at_sentences(chunk["content"], max_tokens)
            for sub_content in sub_chunks:
                final_chunks.append(
                    {
                        "chunk_index": len(final_chunks),
                        "content": sub_content,
                        "word_count": len(sub_content.split()),
                        "token_count": _count_tokens(sub_content),
                    }
                )
        else:
            chunk["chunk_index"] = len(final_chunks)
            final_chunks.append(chunk)

    return final_chunks
```

Concretely: the existing token post-process block moves INTO `_finalize`;
`chunk_chapter`'s three exit paths become `return _finalize(chunks, ...)`.
The early single-chunk paths already handle their own `max_tokens`
split — simplify them to build the one-chunk list and call `_finalize`
(which now does both jobs), deleting the duplicated inline token-split
code in those two branches.

(c) Update the module docstring (lines 1–6) and `chunk_chapter` docstring
to describe the two-level strategy: paragraph packing first; segments
over `2 * target_words` fall through to overlapped sentence windows.

- [ ] **Step 4: Run the full chunker suite**

Run: `python -m pytest tests/test_chunker.py -v`
Expected: all PASS, including all pre-existing tests (the regression pin proves well-formatted behavior is untouched).

- [ ] **Step 5: Commit**

```bash
git add src/utils/chunker.py tests/test_chunker.py
git commit -m "feat: chunk_chapter falls back to sentence windows for oversize segments

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: `library_meta.data_version` migration + reader

**Files:**
- Modify: `agentic_pipeline/db/migrations.py`
- Create: `src/utils/data_version.py`
- Test: `tests/test_data_version.py` (create)

**Interfaces:**
- Consumes: `run_migrations(db_path)` (existing, `agentic_pipeline/db/migrations.py:262`), `get_db_connection()` from `src/database.py`.
- Produces: `library_meta` table (one row, `id=1`, `data_version INTEGER NOT NULL DEFAULT 1`); `get_data_version() -> int | None` in `src/utils/data_version.py` (None ⇢ table absent, pre-migration DB). Task 4 consumes both; Task 10's swap increments the column with plain SQL.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_data_version.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_data_version.py -v`
Expected: FAIL — `no such table: library_meta` / `ModuleNotFoundError: src.utils.data_version`

- [ ] **Step 3: Add the migration**

In `agentic_pipeline/db/migrations.py`, append to the `MIGRATIONS` list:

```python
    # Library-wide data version for cache coherence (bumped by bulk mutations
    # like rechunk --swap; MCP server caches self-invalidate on mismatch)
    """
    CREATE TABLE IF NOT EXISTS library_meta (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        data_version INTEGER NOT NULL DEFAULT 1
    )
    """,
```

And in `run_migrations()`, next to the existing autonomy_config seed
(`INSERT OR IGNORE INTO autonomy_config (id) VALUES (1)`), add:

```python
        cursor.execute("INSERT OR IGNORE INTO library_meta (id, data_version) VALUES (1, 1)")
```

- [ ] **Step 4: Create the reader**

Create `src/utils/data_version.py`:

```python
"""Read the library-wide data version used for cache coherence.

`library_meta.data_version` is bumped inside bulk-mutation transactions
(e.g. `agentic-pipeline rechunk --swap`). In-memory caches store the
version they loaded under and self-invalidate when it changes.
"""

import sqlite3
from typing import Optional

from ..database import get_db_connection


def get_data_version() -> Optional[int]:
    """Current data_version, or None on a pre-migration DB (no self-check)."""
    try:
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT data_version FROM library_meta WHERE id = 1"
            ).fetchone()
            return row["data_version"] if row else None
    except sqlite3.OperationalError:
        return None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_data_version.py tests/test_*migrations*.py -v`
Expected: PASS (including all pre-existing migration tests).

- [ ] **Step 6: Commit**

```bash
git add agentic_pipeline/db/migrations.py src/utils/data_version.py tests/test_data_version.py
git commit -m "feat: library_meta.data_version migration and reader

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Version-aware chunk-embedding cache

**Files:**
- Modify: `src/utils/cache.py` (`CachedEmbeddings` dataclass; `get_chunk_embeddings`/`set_chunk_embeddings`, currently lines 116–144)
- Modify: `src/utils/chunk_loader.py` (`load_chunk_embeddings`, lines 19–96)
- Test: `tests/test_cache.py` (extend; create the file if it does not exist)

**Interfaces:**
- Consumes: `get_data_version()` (Task 3).
- Produces: `LibraryCache.get_chunk_embeddings(current_version: Optional[int] = None)` — returns None (miss) and drops the entry when `current_version` is not None and differs from the stored version; `LibraryCache.set_chunk_embeddings(matrix, metadata, data_version: Optional[int] = None)`. Existing no-arg callers keep legacy behavior (no self-check). `load_chunk_embeddings()` wires the version through automatically — Task 10's swap becomes visible to running servers with no other change.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cache.py` (create with these contents if absent; if the file exists, append the class and add the imports):

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cache.py::TestChunkCacheVersioning -v`
Expected: FAIL — `TypeError: set_chunk_embeddings() got an unexpected keyword argument 'data_version'`

- [ ] **Step 3: Implement version awareness in `cache.py`**

(a) Add a field to `CachedEmbeddings` (line 30):

```python
@dataclass
class CachedEmbeddings:
    """Cached embeddings matrix with metadata"""

    matrix: np.ndarray
    metadata: list[dict]
    loaded_at: float
    data_version: Optional[int] = None
```

(b) Replace `get_chunk_embeddings` / `set_chunk_embeddings` (lines 116–137):

```python
    def get_chunk_embeddings(
        self, current_version: Optional[int] = None
    ) -> Optional[tuple[np.ndarray, list[dict]]]:
        """Get cached chunk embeddings; self-invalidate on data_version mismatch.

        Passing current_version=None skips the check (legacy behavior).
        """
        if not self.enabled:
            return None

        with self._lock:
            if self._chunk_embeddings is not None:
                if (
                    current_version is not None
                    and self._chunk_embeddings.data_version != current_version
                ):
                    logger.info(
                        "Chunk embeddings cache stale "
                        f"(cached v{self._chunk_embeddings.data_version}, "
                        f"db v{current_version}) — invalidating"
                    )
                    self._chunk_embeddings = None
                    self._misses += 1
                    return None
                self._hits += 1
                logger.debug("Chunk embeddings cache hit")
                return self._chunk_embeddings.matrix, self._chunk_embeddings.metadata

            self._misses += 1
            return None

    def set_chunk_embeddings(
        self,
        matrix: np.ndarray,
        metadata: list[dict],
        data_version: Optional[int] = None,
    ) -> None:
        """Cache chunk embeddings matrix and metadata"""
        if not self.enabled:
            return

        with self._lock:
            self._chunk_embeddings = CachedEmbeddings(
                matrix=matrix,
                metadata=metadata,
                loaded_at=time(),
                data_version=data_version,
            )
            logger.info(
                f"Cached chunk embeddings: {matrix.shape[0]} chunks, "
                f"{matrix.nbytes / 1024 / 1024:.1f} MB (data_version={data_version})"
            )
```

- [ ] **Step 4: Wire the version through `chunk_loader.py`**

In `load_chunk_embeddings` (lines 32–38 and 92–93):

```python
    from .data_version import get_data_version

    if cache is None:
        cache = get_cache()

    current_version = get_data_version()

    if cache is not None:
        cached = cache.get_chunk_embeddings(current_version=current_version)
        if cached:
            return cached
```

(put the import at the top of the file with the other relative imports,
not inside the function) and at the bottom:

```python
    if cache is not None:
        cache.set_chunk_embeddings(matrix, metadata, data_version=current_version)
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_cache.py tests/test_chunker.py -v && python -m pytest tests/ -x -q -k "search or loader or cache"`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/utils/cache.py src/utils/chunk_loader.py tests/test_cache.py
git commit -m "feat: chunk cache self-invalidates on data_version bump

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Eval core — matrix loading, ranking, scoring

**Files:**
- Create: `src/utils/retrieval_eval.py`
- Test: `tests/test_retrieval_eval.py` (create)

**Interfaces:**
- Consumes: `find_top_k(query_vec, vectors, k, min_similarity)` from `src/utils/vector_store.py`; `full_text_search(query, limit)` from `src/utils/fts_search.py`; `reciprocal_rank_fusion(fts_results, semantic_results)` from `src/utils/hybrid_search.py`.
- Produces (Tasks 6, 9 consume):
  - `load_gold(paths: list) -> list[dict]`
  - `load_matrix(db_path, table: str) -> tuple[np.ndarray | None, list[dict]]` (metadata dicts: `chunk_id, chapter_id, book_id`; `table` must be `"chunks"` or `"chunks_staging"`)
  - `rank_chapters_semantic(query_vec, matrix, meta, k=10) -> list[dict]` (`chapter_id, book_id, similarity`, distinct chapters, best-first)
  - `rank_chapters_hybrid(query, query_vec, matrix, meta, k=10) -> list[dict]`
  - `evaluate(golds: list[dict], ranked_lists: list[list[dict]], k=5) -> dict` (`{"hit_at_5": float, "mrr": float, "n": int}`)
  - `run_eval(db_path, gold_paths, table, embedder) -> dict` — report keyed `{"auto": {"semantic": {...}, "hybrid": {...}}, "manual": {...}}`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_retrieval_eval.py`:

```python
"""Retrieval eval core: matrix loading, ranking, hit@5/MRR scoring."""

import io
import json
import sqlite3

import numpy as np
import pytest

from src.utils.retrieval_eval import (
    evaluate,
    load_gold,
    load_matrix,
    rank_chapters_semantic,
)


def _blob(vec):
    buf = io.BytesIO()
    np.save(buf, np.asarray(vec, dtype=np.float32))
    return buf.getvalue()


@pytest.fixture
def eval_db(tmp_path):
    db = tmp_path / "library.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE chunks (
            id TEXT PRIMARY KEY, chapter_id TEXT NOT NULL, book_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL, content TEXT NOT NULL,
            word_count INTEGER NOT NULL, embedding BLOB, embedding_model TEXT,
            content_hash TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
    )
    rows = [
        ("k1", "ch1", "b1", 0, "alpha", 1, _blob([1.0, 0.0, 0.0])),
        ("k2", "ch1", "b1", 1, "beta", 1, _blob([0.9, 0.1, 0.0])),  # same chapter as k1
        ("k3", "ch2", "b1", 0, "gamma", 1, _blob([0.0, 1.0, 0.0])),
        ("k4", "ch3", "b2", 0, "delta", 1, _blob([0.0, 0.0, 1.0])),
    ]
    conn.executemany(
        "INSERT INTO chunks (id, chapter_id, book_id, chunk_index, content, word_count, embedding) VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()
    return db


class TestLoadMatrix:
    def test_loads_only_embedded_rows(self, eval_db):
        matrix, meta = load_matrix(eval_db, table="chunks")
        assert matrix.shape == (4, 3)
        assert [m["chunk_id"] for m in meta] == ["k1", "k2", "k3", "k4"]

    def test_rejects_unknown_table(self, eval_db):
        with pytest.raises(ValueError):
            load_matrix(eval_db, table="chunks; DROP TABLE chunks")


class TestSemanticRanking:
    def test_distinct_chapters_and_ordering(self, eval_db):
        matrix, meta = load_matrix(eval_db, table="chunks")
        q = np.array([1.0, 0.05, 0.0], dtype=np.float32)
        ranked = rank_chapters_semantic(q, matrix, meta, k=10)
        ids = [r["chapter_id"] for r in ranked]
        assert ids[0] == "ch1"
        assert len(ids) == len(set(ids)), "chapters must be distinct"
        # k1 and k2 are both ch1 — aggregation collapses them
        assert ids.count("ch1") == 1


class TestEvaluate:
    def test_hit_and_mrr_chapter_gold(self):
        golds = [{"query": "q", "gold_chapter_id": "ch2", "gold_book_id": "b1"}]
        ranked = [[
            {"chapter_id": "ch9", "book_id": "b9"},
            {"chapter_id": "ch2", "book_id": "b1"},
        ]]
        r = evaluate(golds, ranked, k=5)
        assert r["hit_at_5"] == 1.0
        assert r["mrr"] == pytest.approx(0.5)

    def test_book_level_fallback_when_no_gold_chapter(self):
        golds = [{"query": "q", "gold_chapter_id": None, "gold_book_id": "b2"}]
        ranked = [[
            {"chapter_id": "chX", "book_id": "b1"},
            {"chapter_id": "chY", "book_id": "b2"},
        ]]
        r = evaluate(golds, ranked, k=5)
        assert r["hit_at_5"] == 1.0
        assert r["mrr"] == pytest.approx(0.5)

    def test_miss_beyond_k(self):
        golds = [{"query": "q", "gold_chapter_id": "ch1", "gold_book_id": "b1"}]
        ranked = [[{"chapter_id": f"c{i}", "book_id": "b"} for i in range(5)]
                  + [{"chapter_id": "ch1", "book_id": "b1"}]]
        r = evaluate(golds, ranked, k=5)
        assert r["hit_at_5"] == 0.0
        assert r["mrr"] == pytest.approx(1 / 6)


class TestLoadGold:
    def test_merges_files_and_tags_source(self, tmp_path):
        a = tmp_path / "auto.json"
        b = tmp_path / "manual.json"
        a.write_text(json.dumps([{"query": "x", "gold_chapter_id": "c", "gold_book_id": "b", "source": "auto"}]))
        b.write_text(json.dumps([{"query": "y", "gold_chapter_id": None, "gold_book_id": "b", "source": "manual"}]))
        golds = load_gold([a, b])
        assert {g["source"] for g in golds} == {"auto", "manual"}

    def test_missing_file_ok(self, tmp_path):
        a = tmp_path / "auto.json"
        a.write_text(json.dumps([]))
        assert load_gold([a, tmp_path / "nope.json"]) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_retrieval_eval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils.retrieval_eval'`

- [ ] **Step 3: Implement the eval core**

Create `src/utils/retrieval_eval.py`:

```python
"""Retrieval eval: gold-query scoring of chunk-embedding schemes.

Scores hit@5 and MRR at CHAPTER level so results are comparable across
chunking schemes. Semantic mode is the acceptance-gate number; hybrid
mode (production RRF path) is reported for user-facing truth but never
gates (spec decision 1).
"""

import io
import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

from .fts_search import full_text_search
from .hybrid_search import reciprocal_rank_fusion
from .vector_store import find_top_k

logger = logging.getLogger(__name__)

_ALLOWED_TABLES = {"chunks", "chunks_staging"}
_FETCH_K = 50  # chunk-level over-fetch before chapter aggregation


def load_gold(paths: list) -> list[dict]:
    """Load and merge gold-query files; missing files are skipped."""
    golds: list[dict] = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            logger.warning(f"Gold file missing, skipping: {p}")
            continue
        golds.extend(json.loads(p.read_text()))
    return golds


def load_matrix(db_path, table: str = "chunks"):
    """Load (matrix, metadata) for one chunk table from an explicit DB path."""
    if table not in _ALLOWED_TABLES:
        raise ValueError(f"table must be one of {_ALLOWED_TABLES}, got {table!r}")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"SELECT id, chapter_id, book_id, embedding FROM {table} "
            "WHERE embedding IS NOT NULL ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return None, []
    embeddings = [np.load(io.BytesIO(r["embedding"])) for r in rows]
    meta = [
        {"chunk_id": r["id"], "chapter_id": r["chapter_id"], "book_id": r["book_id"]}
        for r in rows
    ]
    return np.vstack(embeddings), meta


def rank_chapters_semantic(query_vec, matrix, meta, k: int = 10) -> list[dict]:
    """Top-k DISTINCT chapters by best-chunk cosine similarity."""
    top = find_top_k(query_vec, matrix, k=min(_FETCH_K, matrix.shape[0]), min_similarity=0.0)
    seen: set = set()
    chapters: list[dict] = []
    for idx, sim in top:
        cid = meta[idx]["chapter_id"]
        if cid in seen:
            continue
        seen.add(cid)
        chapters.append(
            {
                "chapter_id": cid,
                "book_id": meta[idx]["book_id"],
                "similarity": float(sim),
            }
        )
        if len(chapters) >= k:
            break
    return chapters


def rank_chapters_hybrid(query: str, query_vec, matrix, meta, k: int = 10) -> list[dict]:
    """Production-shaped ranking: chapter FTS + chunk semantic fused by RRF.

    Uses full_text_search() (reads BOOK_DB_PATH env) for the FTS arm — the
    live chapters_fts index is chapter-level and identical for both chunk
    schemes, so it cancels out in before/after comparison.
    """
    semantic = rank_chapters_semantic(query_vec, matrix, meta, k=_FETCH_K)
    fts = full_text_search(query, limit=10).get("results", [])
    fused = reciprocal_rank_fusion(fts, semantic)
    return fused[:k]


def _is_hit(gold: dict, item: dict) -> bool:
    if gold.get("gold_chapter_id"):
        return item.get("chapter_id") == gold["gold_chapter_id"]
    return item.get("book_id") == gold["gold_book_id"]


def evaluate(golds: list[dict], ranked_lists: list[list[dict]], k: int = 5) -> dict:
    """hit@k and MRR over parallel (gold, ranked) lists.

    Manual gold entries may carry gold_chapter_id=None — any chapter of
    the gold book counts (book-level hit).
    """
    hits = 0
    rr = 0.0
    for gold, ranked in zip(golds, ranked_lists):
        rank = next(
            (i + 1 for i, item in enumerate(ranked) if _is_hit(gold, item)), None
        )
        if rank is not None and rank <= k:
            hits += 1
        if rank is not None:
            rr += 1.0 / rank
    n = len(golds)
    return {
        "hit_at_5": hits / n if n else 0.0,
        "mrr": rr / n if n else 0.0,
        "n": n,
    }


def run_eval(db_path, gold_paths: list, table: str, embedder) -> dict:
    """Score one chunk table against the gold set, semantic + hybrid.

    embedder: object with .generate(text) -> np.ndarray (injectable for tests).
    Returns {"auto": {"semantic": {...}, "hybrid": {...}}, "manual": {...}}.
    """
    golds = load_gold(gold_paths)
    matrix, meta = load_matrix(db_path, table=table)
    if matrix is None:
        raise RuntimeError(f"No embedded rows in {table}")

    report: dict = {}
    for source in ("auto", "manual"):
        subset = [g for g in golds if g.get("source") == source]
        sem_lists, hyb_lists = [], []
        for g in subset:
            qv = embedder.generate(g["query"])
            sem_lists.append(rank_chapters_semantic(qv, matrix, meta, k=10))
            hyb_lists.append(rank_chapters_hybrid(g["query"], qv, matrix, meta, k=10))
        report[source] = {
            "semantic": evaluate(subset, sem_lists, k=5),
            "hybrid": evaluate(subset, hyb_lists, k=5),
        }
    return report
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_retrieval_eval.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/retrieval_eval.py tests/test_retrieval_eval.py
git commit -m "feat: retrieval eval core — chapter-level hit@5/MRR, semantic + hybrid

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Gold builder (`build-gold` via `claude -p`) + script wrapper

**Files:**
- Modify: `src/utils/retrieval_eval.py` (add `build_gold`, `_claude_generate`, `_parse_fenced_json`)
- Create: `scripts/retrieval_eval.py`
- Test: `tests/test_retrieval_eval.py`

**Interfaces:**
- Consumes: `read_chapter_content(file_path)` from `src/utils/file_utils.py:44` (raises FileNotFoundError); Task 5's `run_eval`.
- Produces: `build_gold(db_path, out_path, n=60, seed=42, min_words=300, passage_words=400, generator=None) -> int` (records written); `scripts/retrieval_eval.py` CLI with `build-gold` and `run` subcommands. Task 9's gate eval reads the produced `eval/gold-queries.json`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_retrieval_eval.py`:

```python
from src.utils.retrieval_eval import _parse_fenced_json, build_gold


@pytest.fixture
def gold_db(tmp_path):
    db = tmp_path / "library.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE books (id TEXT PRIMARY KEY, title TEXT)")
    conn.execute(
        """CREATE TABLE chapters (
            id TEXT PRIMARY KEY, book_id TEXT NOT NULL, title TEXT,
            chapter_number INTEGER, file_path TEXT, word_count INTEGER)"""
    )
    # 3 books x 3 chapters, each with a real content file
    for b in range(3):
        conn.execute("INSERT INTO books VALUES (?, ?)", (f"b{b}", f"Book {b}"))
        for c in range(3):
            path = tmp_path / f"b{b}-c{c}.md"
            path.write_text(" ".join(f"b{b}c{c}word{i}." for i in range(600)))
            conn.execute(
                "INSERT INTO chapters VALUES (?,?,?,?,?,?)",
                (f"b{b}ch{c}", f"b{b}", f"Ch {c}", c, str(path), 600),
            )
    conn.commit()
    conn.close()
    return db


class TestParseFencedJson:
    def test_plain_json(self):
        assert _parse_fenced_json('{"query": "q"}') == {"query": "q"}

    def test_fenced_json(self):
        out = _parse_fenced_json('Here:\n```json\n{"query": "q"}\n```\ndone')
        assert out == {"query": "q"}

    def test_garbage_returns_none(self):
        assert _parse_fenced_json("no json here") is None


class TestBuildGold:
    def test_deterministic_sampling_and_records(self, gold_db, tmp_path):
        calls = []

        def fake_generator(passage):
            calls.append(passage)
            return f"question {len(calls)}?"

        out = tmp_path / "gold.json"
        n1 = build_gold(gold_db, out, n=6, seed=42, min_words=100,
                        passage_words=50, generator=fake_generator)
        first_passages = list(calls)
        records = json.loads(out.read_text())
        assert n1 == 6 == len(records)
        for r in records:
            assert set(r) == {"query", "gold_chapter_id", "gold_book_id", "source"}
            assert r["source"] == "auto"

        # rerun with same seed -> identical passages sampled
        calls.clear()
        build_gold(gold_db, out, n=6, seed=42, min_words=100,
                   passage_words=50, generator=fake_generator)
        assert calls == first_passages

    def test_stratifies_across_books(self, gold_db, tmp_path):
        out = tmp_path / "gold.json"
        build_gold(gold_db, out, n=3, seed=1, min_words=100,
                   passage_words=50, generator=lambda p: "q?")
        records = json.loads(out.read_text())
        assert len({r["gold_book_id"] for r in records}) == 3

    def test_generator_failure_skips_and_logs(self, gold_db, tmp_path):
        def flaky(passage):
            return None  # simulates claude -p failure / unparseable output

        out = tmp_path / "gold.json"
        n = build_gold(gold_db, out, n=4, seed=7, min_words=100,
                       passage_words=50, generator=flaky)
        assert n == 0
        assert json.loads(out.read_text()) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_retrieval_eval.py::TestBuildGold tests/test_retrieval_eval.py::TestParseFencedJson -v`
Expected: FAIL — `ImportError: cannot import name 'build_gold'`

- [ ] **Step 3: Implement `build_gold` in `src/utils/retrieval_eval.py`**

Append:

```python
import random
import re
import subprocess
from collections import defaultdict

_GOLD_PROMPT = (
    "Below is a passage from a book. Write ONE specific question that this "
    "passage uniquely answers — concrete enough that only this passage (not "
    "general knowledge) answers it. Respond with ONLY a JSON object: "
    '{"query": "<the question>"}\n\nPASSAGE:\n'
)


def _parse_fenced_json(text: str) -> Optional[dict]:
    """Parse a JSON object from raw or ```json-fenced output."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else None
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _claude_generate(passage: str, timeout: int = 120) -> Optional[str]:
    """One gold question via `claude -p` (subscription billing, decision 7)."""
    try:
        proc = subprocess.run(
            ["claude", "-p", _GOLD_PROMPT + passage],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning(f"claude -p failed: {e}")
        return None
    if proc.returncode != 0:
        logger.warning(f"claude -p exit {proc.returncode}: {proc.stderr[:200]}")
        return None
    parsed = _parse_fenced_json(proc.stdout)
    if not parsed or not parsed.get("query"):
        logger.warning(f"Unparseable gold output: {proc.stdout[:200]}")
        return None
    return str(parsed["query"]).strip()


def build_gold(
    db_path,
    out_path,
    n: int = 60,
    seed: int = 42,
    min_words: int = 300,
    passage_words: int = 400,
    generator=None,
) -> int:
    """Sample chapters (NOT chunks — decision 6), excerpt a seeded passage,
    generate one question each, write gold records. Returns records written.
    """
    from .file_utils import read_chapter_content

    if generator is None:
        generator = _claude_generate

    rng = random.Random(seed)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        chapters = conn.execute(
            "SELECT c.id, c.book_id, c.file_path FROM chapters c "
            "JOIN books b ON b.id = c.book_id "
            "WHERE c.word_count >= ? AND c.file_path IS NOT NULL "
            "ORDER BY c.id",
            (min_words,),
        ).fetchall()
    finally:
        conn.close()

    # Stratify: round-robin across shuffled books so no book dominates
    by_book: dict = defaultdict(list)
    for ch in chapters:
        by_book[ch["book_id"]].append(ch)
    books = sorted(by_book)
    rng.shuffle(books)
    for b in books:
        rng.shuffle(by_book[b])

    picks = []
    round_idx = 0
    while len(picks) < n:
        advanced = False
        for b in books:
            if round_idx < len(by_book[b]):
                picks.append(by_book[b][round_idx])
                advanced = True
                if len(picks) >= n:
                    break
        if not advanced:
            break  # corpus smaller than n
        round_idx += 1

    records = []
    for ch in picks:
        try:
            text = read_chapter_content(ch["file_path"])
        except (FileNotFoundError, IOError) as e:
            logger.warning(f"Skipping {ch['id']}: {e}")
            continue
        words = text.split()
        if len(words) < min_words:
            continue
        start = rng.randrange(0, max(1, len(words) - passage_words))
        passage = " ".join(words[start : start + passage_words])
        query = generator(passage)
        if not query:
            logger.warning(f"Generator failed for chapter {ch['id']} — skipped")
            continue
        records.append(
            {
                "query": query,
                "gold_chapter_id": ch["id"],
                "gold_book_id": ch["book_id"],
                "source": "auto",
            }
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(records, indent=2))
    logger.info(f"Wrote {len(records)} gold records to {out_path}")
    return len(records)
```

NOTE on determinism: the passage offset is drawn BEFORE the generator is
called, so generator failures/skips never shift later samples. A missing
chapter FILE does skip before the draw and would shift later offsets —
acceptable; the determinism test uses a no-missing-file fixture.

- [ ] **Step 4: Create the script wrapper**

Create `scripts/retrieval_eval.py`:

```python
#!/usr/bin/env python3
"""Retrieval eval CLI — gold-set builder and before/after scorer.

Usage:
    python scripts/retrieval_eval.py build-gold [--n 60] [--seed 42]
    python scripts/retrieval_eval.py run [--table chunks|chunks_staging]

Requires: BOOK_DB_PATH (defaults to the standard library.db location),
OPENAI_API_KEY for `run` (query embeddings), `claude` CLI for build-gold.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DB = Path.home() / "Library/Application Support/book-library/library.db"
EVAL_DIR = REPO_ROOT / "eval"
GOLD_AUTO = EVAL_DIR / "gold-queries.json"
GOLD_MANUAL = EVAL_DIR / "gold-queries-manual.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    bg = sub.add_parser("build-gold", help="Generate the auto gold set via claude -p")
    bg.add_argument("--n", type=int, default=60)
    bg.add_argument("--seed", type=int, default=42)

    run = sub.add_parser("run", help="Score a chunk table against the gold sets")
    run.add_argument("--table", choices=["chunks", "chunks_staging"], default="chunks")

    args = parser.parse_args()
    db_path = Path(os.environ.get("BOOK_DB_PATH", DEFAULT_DB))
    if not db_path.exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        return 1

    from src.utils.retrieval_eval import build_gold, run_eval

    if args.cmd == "build-gold":
        written = build_gold(db_path, GOLD_AUTO, n=args.n, seed=args.seed)
        print(f"{written} gold records -> {GOLD_AUTO}")
        return 0 if written else 1

    from src.utils.openai_embeddings import OpenAIEmbeddingGenerator

    report = run_eval(
        db_path, [GOLD_AUTO, GOLD_MANUAL], table=args.table,
        embedder=OpenAIEmbeddingGenerator(),
    )
    print(json.dumps({"table": args.table, "report": report}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_retrieval_eval.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/utils/retrieval_eval.py scripts/retrieval_eval.py tests/test_retrieval_eval.py
git commit -m "feat: gold-set builder (claude -p) and retrieval_eval CLI wrapper

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Manual gold set (data)

**Files:**
- Create: `eval/gold-queries-manual.json`

**Interfaces:**
- Consumes: live library DB, READ-ONLY (title lookups). This task runs SQL SELECTs against the real DB to resolve book ids — allowed because it only reads.
- Produces: `eval/gold-queries-manual.json`, consumed by Task 9's gate eval. Schema per record: `{"query": str, "gold_chapter_id": null, "gold_book_id": str, "source": "manual"}` (book-level gold: any chapter of the book is a hit).

- [ ] **Step 1: Resolve book ids for the draft queries**

Run this (READ-ONLY) against the live DB and capture the id for each title fragment:

```bash
source .venv/bin/activate && python3 - <<'EOF'
import json, sqlite3
db = "/Users/taylorstephens/Library/Application Support/book-library/library.db"
conn = sqlite3.connect(db); conn.row_factory = sqlite3.Row
fragments = [
    "Rick Steves Sicily", "Designing Data", "Laws of UX",
    "Lonely Planet Naples", "The World", "Effective Shell",
    "Pro Angular", "Tailwind",
]
for f in fragments:
    rows = conn.execute(
        "SELECT id, title FROM books WHERE title LIKE ? LIMIT 3", (f"%{f}%",)
    ).fetchall()
    print(f, "->", [(r["id"], r["title"]) for r in rows])
EOF
```

Every fragment MUST resolve to exactly one book. If any is ambiguous or
missing, adjust the fragment until unique, or drop that query and note it
in the commit message. Do not guess ids.

- [ ] **Step 2: Write the manual gold file**

Create `eval/gold-queries-manual.json` with the resolved `gold_book_id`
values substituted for the `RESOLVE:` placeholders (the file MUST NOT be
committed with any `RESOLVE:` string remaining):

```json
[
  {"query": "best time of year to visit Mount Etna and how to get to the summit craters", "gold_chapter_id": null, "gold_book_id": "RESOLVE:Rick Steves Sicily", "source": "manual"},
  {"query": "day trip from Palermo to see the mosaics at Monreale cathedral", "gold_chapter_id": null, "gold_book_id": "RESOLVE:Rick Steves Sicily", "source": "manual"},
  {"query": "how LSM-tree compaction differs from B-tree page updates", "gold_chapter_id": null, "gold_book_id": "RESOLVE:Designing Data", "source": "manual"},
  {"query": "why exactly-once message delivery is hard in distributed stream processing", "gold_chapter_id": null, "gold_book_id": "RESOLVE:Designing Data", "source": "manual"},
  {"query": "examples of Jakob's law applied to product interface design", "gold_chapter_id": null, "gold_book_id": "RESOLVE:Laws of UX", "source": "manual"},
  {"query": "which Naples neighborhood is best for authentic pizza", "gold_chapter_id": null, "gold_book_id": "RESOLVE:Lonely Planet Naples", "source": "manual"},
  {"query": "planning an overland route across multiple continents", "gold_chapter_id": null, "gold_book_id": "RESOLVE:The World", "source": "manual"},
  {"query": "how to chain shell commands with pipes and redirect output to files", "gold_chapter_id": null, "gold_book_id": "RESOLVE:Effective Shell", "source": "manual"},
  {"query": "how Angular signals change detection differs from zone.js", "gold_chapter_id": null, "gold_book_id": "RESOLVE:Pro Angular", "source": "manual"},
  {"query": "building a responsive layout with Tailwind utility classes", "gold_chapter_id": null, "gold_book_id": "RESOLVE:Tailwind", "source": "manual"}
]
```

- [ ] **Step 3: Validate the file**

```bash
python3 - <<'EOF'
import json, sqlite3
records = json.load(open("eval/gold-queries-manual.json"))
assert 8 <= len(records) <= 12, f"expected ~10 records, got {len(records)}"
db = "/Users/taylorstephens/Library/Application Support/book-library/library.db"
conn = sqlite3.connect(db)
for r in records:
    assert r["source"] == "manual"
    assert "RESOLVE:" not in str(r["gold_book_id"])
    hit = conn.execute("SELECT 1 FROM books WHERE id = ?", (r["gold_book_id"],)).fetchone()
    assert hit, f"gold_book_id not in library: {r['gold_book_id']} ({r['query'][:40]})"
print(f"OK: {len(records)} manual gold records, all book ids verified")
EOF
```

Expected: `OK: ... all book ids verified`

- [ ] **Step 4: Commit**

```bash
git add eval/gold-queries-manual.json
git commit -m "feat: manual gold set — 10 known-miss queries with verified book ids

Operator (Taylor) reviews/edits this file before the baseline run
(spec decision 4).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: `rechunk` staging — hash-keyed reconcile + embedding

**Files:**
- Create: `agentic_pipeline/library/rechunk.py`
- Test: `tests/test_rechunk.py` (create)

**Interfaces:**
- Consumes: `chunk_chapter()` (Task 2), `compute_content_hash(content)` from `src/utils/embedding_sync.py:25`, `read_chapter_content(file_path)` from `src/utils/file_utils.py:44`, `OpenAIEmbeddingGenerator.generate_batch(texts) -> np.ndarray` from `src/utils/openai_embeddings.py`.
- Produces (Task 9/10 and the CLI consume):
  - `ensure_staging(conn) -> None`
  - `drop_staging(conn) -> None`
  - `stage_all(conn, chunk_fn=None) -> dict` — report: `{"chapters": int, "staged_chunks": int, "reused_embeddings": int, "pending_embeddings": int, "carried_chapters": [chapter_ids], "skipped": [chapter_ids]}`
  - `stage_chapter(conn, chapter_row, chunk_fn) -> tuple[int, int]` (staged, reused)
  - `embed_pending(conn, generator=None, batch_size=256) -> int` (rows embedded)
  - `estimate_embedding_cost(conn) -> dict` (`{"pending": int, "words": int, "est_tokens": int, "est_usd": float}`)
  - `snapshot_marker(conn) -> dict` (`{"chunk_count": int, "max_created_at": str}`)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rechunk.py`:

```python
"""rechunk staging: hash-keyed reconcile, embedding, snapshot marker."""

import io
import sqlite3

import numpy as np
import pytest

from agentic_pipeline.library import rechunk as rc


def _blob(vec):
    buf = io.BytesIO()
    np.save(buf, np.asarray(vec, dtype=np.float32))
    return buf.getvalue()


class FakeGenerator:
    """Deterministic stand-in for OpenAIEmbeddingGenerator."""

    def __init__(self):
        self.calls = []

    def generate_batch(self, texts):
        self.calls.append(list(texts))
        return np.stack([np.full(4, float(len(t)), dtype=np.float32) for t in texts])


@pytest.fixture
def staged_db(tmp_path, monkeypatch):
    db = tmp_path / "library.db"
    monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(db))
    monkeypatch.setenv("BOOK_DB_PATH", str(db))
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE books (id TEXT PRIMARY KEY, title TEXT)")
    conn.execute(
        """CREATE TABLE chapters (
            id TEXT PRIMARY KEY, book_id TEXT NOT NULL, title TEXT,
            chapter_number INTEGER, file_path TEXT, word_count INTEGER)"""
    )
    conn.execute(
        """CREATE TABLE chunks (
            id TEXT PRIMARY KEY, chapter_id TEXT NOT NULL, book_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL, content TEXT NOT NULL,
            word_count INTEGER NOT NULL, embedding BLOB, embedding_model TEXT,
            content_hash TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
    )
    conn.execute("CREATE UNIQUE INDEX idx_chunks_chapter_index ON chunks(chapter_id, chunk_index)")
    conn.execute("INSERT INTO books VALUES ('b1', 'Book One')")
    # chapter with a readable file (wall of text, will produce multiple windows)
    ch_file = tmp_path / "ch1.md"
    ch_file.write_text(" ".join(f"Sentence number {i} has five words." for i in range(400)))
    conn.execute(
        "INSERT INTO chapters VALUES ('ch1', 'b1', 'One', 1, ?, 2400)", (str(ch_file),)
    )
    # chapter whose source file is GONE, but has live chunks (carried case)
    conn.execute(
        "INSERT INTO chapters VALUES ('ch2', 'b1', 'Two', 2, ?, 100)",
        (str(tmp_path / "missing.md"),),
    )
    conn.execute(
        "INSERT INTO chunks VALUES ('old1', 'ch2', 'b1', 0, 'legacy content', 2, ?, 'text-embedding-3-large', 'oldhash', '2026-01-01')",
        (_blob([1, 2, 3, 4]),),
    )
    conn.commit()
    yield conn, db
    conn.close()


class TestStaging:
    def test_stage_all_populates_staging_and_never_touches_chunks(self, staged_db):
        conn, db = staged_db
        before = conn.execute("SELECT COUNT(*), COALESCE(SUM(LENGTH(content)),0) FROM chunks").fetchone()
        rc.ensure_staging(conn)
        report = rc.stage_all(conn)
        after = conn.execute("SELECT COUNT(*), COALESCE(SUM(LENGTH(content)),0) FROM chunks").fetchone()
        assert tuple(before) == tuple(after), "stage must not mutate chunks"
        n = conn.execute("SELECT COUNT(*) FROM chunks_staging").fetchone()[0]
        assert n >= 3  # ch1 windows + carried ch2 row
        assert report["carried_chapters"] == ["ch2"]

    def test_carried_chapter_keeps_live_rows_and_embeddings(self, staged_db):
        conn, db = staged_db
        rc.ensure_staging(conn)
        rc.stage_all(conn)
        row = conn.execute(
            "SELECT content, embedding FROM chunks_staging WHERE chapter_id = 'ch2'"
        ).fetchone()
        assert row["content"] == "legacy content"
        assert row["embedding"] is not None

    def test_hash_reconcile_preserves_embeddings_on_rerun(self, staged_db):
        conn, db = staged_db
        rc.ensure_staging(conn)
        rc.stage_all(conn)
        gen = FakeGenerator()
        embedded = rc.embed_pending(conn, generator=gen)
        assert embedded > 0
        # rerun stage: identical chunker output -> all embeddings reused
        report2 = rc.stage_all(conn)
        assert report2["pending_embeddings"] == 0
        assert report2["reused_embeddings"] == embedded
        gen2 = FakeGenerator()
        assert rc.embed_pending(conn, generator=gen2) == 0
        assert gen2.calls == []

    def test_param_change_reembeds_only_changed(self, staged_db):
        conn, db = staged_db
        rc.ensure_staging(conn)
        rc.stage_all(conn)
        rc.embed_pending(conn, generator=FakeGenerator())

        # different chunker output -> old rows replaced, need embedding again
        def tiny_chunker(text, **kw):
            words = text.split()
            half = len(words) // 2
            return [
                {"chunk_index": 0, "content": " ".join(words[:half]), "word_count": half, "token_count": half},
                {"chunk_index": 1, "content": " ".join(words[half:]), "word_count": len(words) - half, "token_count": len(words) - half},
            ]

        report = rc.stage_all(conn, chunk_fn=tiny_chunker)
        assert report["pending_embeddings"] == 2  # ch1 re-chunked; ch2 carried untouched
        assert report["reused_embeddings"] >= 1  # ch2's carried row

    def test_embed_pending_writes_numpy_blobs_and_model(self, staged_db):
        conn, db = staged_db
        rc.ensure_staging(conn)
        rc.stage_all(conn)
        rc.embed_pending(conn, generator=FakeGenerator())
        row = conn.execute(
            "SELECT embedding, embedding_model FROM chunks_staging WHERE chapter_id='ch1' LIMIT 1"
        ).fetchone()
        vec = np.load(io.BytesIO(row["embedding"]))
        assert vec.shape == (4,)
        assert row["embedding_model"] == "text-embedding-3-large"

    def test_snapshot_marker(self, staged_db):
        conn, db = staged_db
        marker = rc.snapshot_marker(conn)
        assert marker["chunk_count"] == 1
        assert marker["max_created_at"] == "2026-01-01"

    def test_estimate_cost_counts_pending_only(self, staged_db):
        conn, db = staged_db
        rc.ensure_staging(conn)
        rc.stage_all(conn)
        est = rc.estimate_embedding_cost(conn)
        assert est["pending"] > 0
        assert est["est_usd"] > 0
        rc.embed_pending(conn, generator=FakeGenerator())
        assert rc.estimate_embedding_cost(conn)["pending"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_rechunk.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agentic_pipeline.library.rechunk'`

- [ ] **Step 3: Implement staging**

Create `agentic_pipeline/library/rechunk.py`:

```python
"""Staged re-chunking of the chunks table (spec: 2026-07-15-rechunking-design).

`stage_all` re-chunks every chapter into chunks_staging, reconciling by
content_hash so embeddings survive reruns (decision 5: no stale-embedding
state is representable). Production `chunks` is only written by `swap()`
(Task 10), inside one transaction, behind the eval gate.
"""

import io
import json
import logging
import uuid
from pathlib import Path

import numpy as np

from src.utils.chunker import chunk_chapter
from src.utils.embedding_sync import compute_content_hash
from src.utils.file_utils import read_chapter_content

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-large"
# $0.13 / 1M tokens for text-embedding-3-large; ~1.35 tokens/word for book prose
_USD_PER_MTOKEN = 0.13
_TOKENS_PER_WORD = 1.35

_STAGING_DDL = """
    CREATE TABLE IF NOT EXISTS chunks_staging (
        id TEXT PRIMARY KEY,
        chapter_id TEXT NOT NULL,
        book_id TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        content TEXT NOT NULL,
        word_count INTEGER NOT NULL,
        embedding BLOB,
        embedding_model TEXT,
        content_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(chapter_id, chunk_index)
    )
"""


def ensure_staging(conn) -> None:
    conn.execute(_STAGING_DDL)
    conn.commit()


def drop_staging(conn) -> None:
    conn.execute("DROP TABLE IF EXISTS chunks_staging")
    conn.commit()


def stage_chapter(conn, chapter_row, chunk_fn) -> tuple[int, int]:
    """Re-chunk one chapter into staging, reusing embeddings by content_hash.

    Returns (staged_count, reused_embedding_count).
    """
    text = read_chapter_content(chapter_row["file_path"])
    desired = chunk_fn(text)

    # harvest existing embeddings for this chapter before replacing its rows
    existing = {}
    for row in conn.execute(
        "SELECT content_hash, embedding, embedding_model FROM chunks_staging "
        "WHERE chapter_id = ? AND embedding IS NOT NULL",
        (chapter_row["id"],),
    ):
        existing.setdefault(row["content_hash"], []).append(
            (row["embedding"], row["embedding_model"])
        )

    conn.execute("DELETE FROM chunks_staging WHERE chapter_id = ?", (chapter_row["id"],))

    reused = 0
    for chunk in desired:
        chash = compute_content_hash(chunk["content"])
        embedding, model = None, None
        if existing.get(chash):
            embedding, model = existing[chash].pop()
            reused += 1
        conn.execute(
            "INSERT INTO chunks_staging "
            "(id, chapter_id, book_id, chunk_index, content, word_count, "
            " embedding, embedding_model, content_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                chapter_row["id"],
                chapter_row["book_id"],
                chunk["chunk_index"],
                chunk["content"],
                chunk["word_count"],
                embedding,
                model,
                chash,
            ),
        )
    return len(desired), reused


def _carry_live_chunks(conn, chapter_id: str) -> int:
    """Source file gone: carry the chapter's live chunks into staging as-is."""
    conn.execute("DELETE FROM chunks_staging WHERE chapter_id = ?", (chapter_id,))
    cur = conn.execute(
        "INSERT INTO chunks_staging "
        "(id, chapter_id, book_id, chunk_index, content, word_count, "
        " embedding, embedding_model, content_hash) "
        "SELECT id, chapter_id, book_id, chunk_index, content, word_count, "
        "       embedding, embedding_model, COALESCE(content_hash, '') "
        "FROM chunks WHERE chapter_id = ?",
        (chapter_id,),
    )
    return cur.rowcount


def stage_all(conn, chunk_fn=None) -> dict:
    """Re-chunk every chapter into staging. Never writes `chunks`."""
    if chunk_fn is None:
        chunk_fn = chunk_chapter
    ensure_staging(conn)

    chapters = conn.execute(
        "SELECT id, book_id, file_path FROM chapters WHERE file_path IS NOT NULL ORDER BY id"
    ).fetchall()

    staged = reused = 0
    carried: list[str] = []
    skipped: list[str] = []
    for ch in chapters:
        try:
            s, r = stage_chapter(conn, ch, chunk_fn)
            staged += s
            reused += r
        except (FileNotFoundError, IOError):
            n = _carry_live_chunks(conn, ch["id"])
            if n:
                staged += n
                reused += n  # carried rows keep their embeddings
                carried.append(ch["id"])
            else:
                skipped.append(ch["id"])
                logger.warning(f"Chapter {ch['id']}: source gone and no live chunks — skipped")
    conn.commit()

    pending = conn.execute(
        "SELECT COUNT(*) FROM chunks_staging WHERE embedding IS NULL"
    ).fetchone()[0]

    report = {
        "chapters": len(chapters),
        "staged_chunks": staged,
        "reused_embeddings": reused,
        "pending_embeddings": pending,
        "carried_chapters": carried,
        "skipped": skipped,
    }
    logger.info(f"stage_all: {report}")
    return report


def estimate_embedding_cost(conn) -> dict:
    row = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(word_count), 0) FROM chunks_staging "
        "WHERE embedding IS NULL"
    ).fetchone()
    pending, words = row[0], row[1]
    est_tokens = int(words * _TOKENS_PER_WORD)
    return {
        "pending": pending,
        "words": words,
        "est_tokens": est_tokens,
        "est_usd": est_tokens / 1_000_000 * _USD_PER_MTOKEN,
    }


def embed_pending(conn, generator=None, batch_size: int = 256) -> int:
    """Embed staged rows lacking embeddings. Resumable by construction."""
    if generator is None:
        from src.utils.openai_embeddings import OpenAIEmbeddingGenerator

        generator = OpenAIEmbeddingGenerator()

    total = 0
    while True:
        rows = conn.execute(
            "SELECT id, content FROM chunks_staging WHERE embedding IS NULL "
            "ORDER BY id LIMIT ?",
            (batch_size,),
        ).fetchall()
        if not rows:
            break
        vectors = generator.generate_batch([r["content"] for r in rows])
        for row, vec in zip(rows, vectors):
            buf = io.BytesIO()
            np.save(buf, np.asarray(vec, dtype=np.float32))
            conn.execute(
                "UPDATE chunks_staging SET embedding = ?, embedding_model = ? WHERE id = ?",
                (buf.getvalue(), EMBEDDING_MODEL, row["id"]),
            )
        conn.commit()  # commit per batch: interruption loses at most one batch
        total += len(rows)
        logger.info(f"embed_pending: {total} embedded so far")
    return total


def snapshot_marker(conn) -> dict:
    """Live-chunks marker for the swap-time staleness check (decision 2)."""
    row = conn.execute(
        "SELECT COUNT(*), COALESCE(MAX(created_at), '') FROM chunks"
    ).fetchone()
    return {"chunk_count": row[0], "max_created_at": row[1]}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_rechunk.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agentic_pipeline/library/rechunk.py tests/test_rechunk.py
git commit -m "feat: rechunk staging with hash-keyed embedding reuse

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Gate eval, verdict persistence, and the `rechunk` CLI command

**Files:**
- Modify: `agentic_pipeline/library/rechunk.py`
- Modify: `agentic_pipeline/cli.py` (add `rechunk` command before `if __name__ == "__main__":`)
- Test: `tests/test_rechunk.py`

**Interfaces:**
- Consumes: `run_eval(db_path, gold_paths, table, embedder)` (Task 5), `snapshot_marker` / `stage_all` / `embed_pending` / `estimate_embedding_cost` (Task 8), CLI helpers `get_db_path()` and `run_migrations` already used by other commands in `cli.py`.
- Produces:
  - `gate_pass(baseline: dict, staged: dict) -> bool` (pure; spec gate verbatim)
  - `run_gate_eval(db_path, gold_paths, embedder=None) -> dict` — verdict: `{"baseline": ..., "staged": ..., "pass": bool, "snapshot": {...}, "gold_counts": {...}}`, persisted to `<db-dir>/rechunk/last-verdict.json`
  - `load_verdict(db_path) -> dict | None`
  - CLI: `agentic-pipeline rechunk [--fresh] [--yes] [--swap]` (Task 10 implements `--swap`'s body; this task wires the flag to a stub that reports "swap not yet implemented" and exits 1, so the CLI is testable now)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_rechunk.py`:

```python
class TestGate:
    BASE = {
        "auto": {"semantic": {"hit_at_5": 0.60, "mrr": 0.40, "n": 60},
                 "hybrid": {"hit_at_5": 0.70, "mrr": 0.50, "n": 60}},
        "manual": {"semantic": {"hit_at_5": 0.20, "mrr": 0.15, "n": 10},
                   "hybrid": {"hit_at_5": 0.30, "mrr": 0.20, "n": 10}},
    }

    def _staged(self, auto_hit, auto_mrr, manual_hit):
        return {
            "auto": {"semantic": {"hit_at_5": auto_hit, "mrr": auto_mrr, "n": 60},
                     "hybrid": {"hit_at_5": 0.0, "mrr": 0.0, "n": 60}},
            "manual": {"semantic": {"hit_at_5": manual_hit, "mrr": 0.5, "n": 10},
                       "hybrid": {"hit_at_5": 0.0, "mrr": 0.0, "n": 10}},
        }

    def test_pass_requires_all_three_arms(self):
        assert rc.gate_pass(self.BASE, self._staged(0.60, 0.40, 0.30)) is True

    def test_auto_hit_regression_fails(self):
        assert rc.gate_pass(self.BASE, self._staged(0.59, 0.40, 0.30)) is False

    def test_auto_mrr_regression_fails(self):
        assert rc.gate_pass(self.BASE, self._staged(0.60, 0.39, 0.30)) is False

    def test_manual_equal_is_not_strict_improvement(self):
        assert rc.gate_pass(self.BASE, self._staged(0.60, 0.40, 0.20)) is False

    def test_hybrid_numbers_never_gate(self):
        # staged hybrid is 0.0 everywhere; gate still passes on semantic arms
        assert rc.gate_pass(self.BASE, self._staged(0.65, 0.45, 0.40)) is True


class TestVerdictPersistence:
    def test_run_gate_eval_persists_verdict(self, staged_db, tmp_path, monkeypatch):
        conn, db = staged_db
        rc.ensure_staging(conn)
        rc.stage_all(conn)
        rc.embed_pending(conn, generator=FakeGenerator())

        gold = tmp_path / "gold.json"
        gold.write_text(
            '[{"query": "q", "gold_chapter_id": "ch1", "gold_book_id": "b1", "source": "auto"}]'
        )

        class FakeEmbedder:
            def generate(self, text):
                return np.full(4, 5.0, dtype=np.float32)

        # avoid the FTS arm in unit tests (no chapters_fts table in fixture)
        monkeypatch.setattr(
            "src.utils.retrieval_eval.full_text_search",
            lambda q, limit=10: {"results": []},
        )
        verdict = rc.run_gate_eval(db, [gold], embedder=FakeEmbedder())
        assert "baseline" in verdict and "staged" in verdict
        assert isinstance(verdict["pass"], bool)
        assert verdict["snapshot"]["chunk_count"] == 1

        loaded = rc.load_verdict(db)
        assert loaded == verdict

    def test_load_verdict_none_when_absent(self, tmp_path):
        assert rc.load_verdict(tmp_path / "library.db") is None


class TestRechunkCli:
    def test_stage_flow_reports_and_never_swaps(self, staged_db, tmp_path, monkeypatch):
        from click.testing import CliRunner
        from agentic_pipeline.cli import cli

        conn, db = staged_db
        gold = tmp_path / "gold.json"
        gold.write_text("[]")
        monkeypatch.setattr("agentic_pipeline.library.rechunk.GOLD_PATHS", [gold])
        monkeypatch.setattr(
            "agentic_pipeline.library.rechunk.embed_pending",
            lambda conn, generator=None, batch_size=256: 0,
        )
        monkeypatch.setattr(
            "agentic_pipeline.library.rechunk.run_gate_eval",
            lambda db_path, gold_paths=None, embedder=None: {"pass": False, "baseline": {}, "staged": {}, "snapshot": {}},
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["rechunk", "--yes"])
        assert result.exit_code in (0, 1)  # FAIL verdict exits 1; either way no crash
        before = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert before == 1, "CLI stage flow must not touch chunks"
```

NOTE: the CLI test monkeypatches `run_gate_eval` and `embed_pending` at
module level — the CLI command must call them via the `rc.` module
namespace (`rc.embed_pending(...)`), not via `from ... import` names, or
the monkeypatch will not take.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_rechunk.py::TestGate tests/test_rechunk.py::TestVerdictPersistence tests/test_rechunk.py::TestRechunkCli -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'gate_pass'`

- [ ] **Step 3: Implement gate + verdict in `rechunk.py`**

Append to `agentic_pipeline/library/rechunk.py`:

```python
# Gold files live in the repo (spec: only runtime state lives beside the DB)
_REPO_ROOT = Path(__file__).resolve().parents[2]
GOLD_PATHS = [
    _REPO_ROOT / "eval" / "gold-queries.json",
    _REPO_ROOT / "eval" / "gold-queries-manual.json",
]


def _verdict_path(db_path) -> Path:
    return Path(db_path).parent / "rechunk" / "last-verdict.json"


def gate_pass(baseline: dict, staged: dict) -> bool:
    """Spec acceptance gate, semantic mode only (decision 1), verbatim:

    auto hit@5 >= baseline AND auto MRR >= baseline AND manual hit@5 > baseline.
    """
    b_auto = baseline["auto"]["semantic"]
    s_auto = staged["auto"]["semantic"]
    b_manual = baseline["manual"]["semantic"]
    s_manual = staged["manual"]["semantic"]
    return (
        s_auto["hit_at_5"] >= b_auto["hit_at_5"]
        and s_auto["mrr"] >= b_auto["mrr"]
        and s_manual["hit_at_5"] > b_manual["hit_at_5"]
    )


def run_gate_eval(db_path, gold_paths=None, embedder=None) -> dict:
    """Score live vs staged, decide the gate, persist the verdict."""
    import sqlite3 as _sqlite3

    from src.utils.retrieval_eval import run_eval

    if gold_paths is None:
        gold_paths = GOLD_PATHS
    if embedder is None:
        from src.utils.openai_embeddings import OpenAIEmbeddingGenerator

        embedder = OpenAIEmbeddingGenerator()

    baseline = run_eval(db_path, gold_paths, table="chunks", embedder=embedder)
    staged = run_eval(db_path, gold_paths, table="chunks_staging", embedder=embedder)

    conn = _sqlite3.connect(str(db_path))
    try:
        snapshot = snapshot_marker(conn)
    finally:
        conn.close()

    verdict = {
        "baseline": baseline,
        "staged": staged,
        "pass": gate_pass(baseline, staged),
        "snapshot": snapshot,
        "gold_counts": {
            "auto": baseline["auto"]["semantic"]["n"],
            "manual": baseline["manual"]["semantic"]["n"],
        },
    }
    path = _verdict_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(verdict, indent=2))
    logger.info(f"Verdict ({'PASS' if verdict['pass'] else 'FAIL'}) -> {path}")
    return verdict


def load_verdict(db_path):
    path = _verdict_path(db_path)
    if not path.exists():
        return None
    return json.loads(path.read_text())
```

- [ ] **Step 4: Add the CLI command**

In `agentic_pipeline/cli.py`, before `if __name__ == "__main__":`, following the house pattern (lazy imports, `get_db_path()`, `sqlite3.connect(timeout=10)`, `try/finally`):

```python
@cli.command()
@click.option("--swap", "do_swap", is_flag=True, help="Apply staged chunks to production (requires PASS verdict).")
@click.option("--fresh", is_flag=True, help="Drop staging and start over.")
@click.option("--yes", is_flag=True, help="Skip the embedding-cost confirmation.")
def rechunk(do_swap: bool, fresh: bool, yes: bool) -> None:
    """Re-chunk the library into staging, eval it, and (with --swap) apply.

    Default mode stages + embeds + evals; production chunks are never
    written. --swap applies a PASSed staging atomically (backup first).
    Exit codes: 0 = PASS (or swap applied), 1 = FAIL verdict or refusal.
    """
    import sqlite3 as sqlite3_mod

    from agentic_pipeline.db.migrations import run_migrations
    from agentic_pipeline.library import rechunk as rc

    db_path = get_db_path()
    run_migrations(db_path)

    if do_swap:
        try:
            report = rc.swap(db_path)
        except rc.SwapRefused as e:
            click.echo(f"REFUSED: {e}", err=True)
            raise SystemExit(1)
        click.echo(f"Swapped: {report['chunks']} chunks live (backup: {report['backup']})")
        click.echo(f"data_version bumped to {report['data_version']}.")
        return

    conn = sqlite3_mod.connect(db_path, timeout=10)
    conn.row_factory = sqlite3_mod.Row
    try:
        if fresh:
            rc.drop_staging(conn)
        rc.ensure_staging(conn)
        report = rc.stage_all(conn)
        click.echo(
            f"Staged {report['staged_chunks']} chunks from {report['chapters']} chapters "
            f"({report['reused_embeddings']} embeddings reused, "
            f"{report['pending_embeddings']} pending)."
        )
        if report["carried_chapters"]:
            click.echo(f"Carried as-is (source gone): {len(report['carried_chapters'])} chapters")
        for ch in report["carried_chapters"]:
            click.echo(f"  - {ch}")

        est = rc.estimate_embedding_cost(conn)
        if est["pending"]:
            click.echo(
                f"Embedding {est['pending']} chunks (~{est['est_tokens']:,} tokens, "
                f"~${est['est_usd']:.2f})."
            )
            if not yes and not click.confirm("Proceed with embedding spend?"):
                click.echo("Aborted before embedding. Staging kept; rerun to resume.")
                raise SystemExit(1)
            rc.embed_pending(conn)
    finally:
        conn.close()

    verdict = rc.run_gate_eval(db_path)
    if verdict.get("baseline"):  # tolerate short-circuit verdicts in tests
        for source in ("auto", "manual"):
            for mode in ("semantic", "hybrid"):
                b = verdict["baseline"][source][mode]
                s = verdict["staged"][source][mode]
                gate_tag = " <- GATE" if mode == "semantic" else ""
                click.echo(
                    f"{source:>6}/{mode:<8} hit@5 {b['hit_at_5']:.2f} -> {s['hit_at_5']:.2f}  "
                    f"MRR {b['mrr']:.3f} -> {s['mrr']:.3f}{gate_tag}"
                )
    click.echo(f"VERDICT: {'PASS — run `agentic-pipeline rechunk --swap` to apply' if verdict['pass'] else 'FAIL — no swap'}")
    raise SystemExit(0 if verdict["pass"] else 1)
```

Also add to `rechunk.py` (Task 10 fills in the real body — this keeps the
CLI importable and testable now):

```python
class SwapRefused(RuntimeError):
    """Swap preconditions not met (no PASS verdict, unembedded rows, ...)."""


def swap(db_path) -> dict:
    raise SwapRefused("swap not yet implemented")
```

(The `TestRechunkCli` test monkeypatches `run_gate_eval`, so a `{"pass": False, ...}` verdict short-circuits before any real eval. The verdict-report loop must tolerate empty `baseline`/`staged` dicts — guard it: `if verdict.get("baseline"):` around the report loop.)

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_rechunk.py tests/test_cli*.py -v`
Expected: PASS (all new tests plus every pre-existing CLI test).

- [ ] **Step 6: Commit**

```bash
git add agentic_pipeline/library/rechunk.py agentic_pipeline/cli.py tests/test_rechunk.py
git commit -m "feat: rechunk CLI — stage, embed, gate eval, persisted verdict

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: `rechunk --swap` — staleness delta, backup, atomic swap

**Files:**
- Modify: `agentic_pipeline/library/rechunk.py` (replace the `swap` stub)
- Test: `tests/test_rechunk.py`

**Interfaces:**
- Consumes: `create_backup(db_path) -> Path` from `agentic_pipeline/health/doctor.py:217`, `load_verdict` / `stage_chapter` / `embed_pending` / `snapshot_marker` (Tasks 8–9), `library_meta` (Task 3).
- Produces: `swap(db_path, generator=None) -> dict` — `{"chunks": int, "backup": str, "data_version": int, "delta_chapters": int}`; raises `SwapRefused` on any precondition failure. Module test seam: `_pre_commit_hook()` (no-op; monkeypatched by the mid-transaction failure test).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_rechunk.py`:

```python
def _prep_passing_swap(conn, db, tmp_path):
    """Stage + embed + write a PASS verdict with a current snapshot."""
    import json as _json

    rc.ensure_staging(conn)
    rc.stage_all(conn)
    rc.embed_pending(conn, generator=FakeGenerator())
    # library_meta must exist for the version bump
    conn.execute(
        "CREATE TABLE IF NOT EXISTS library_meta "
        "(id INTEGER PRIMARY KEY CHECK (id = 1), data_version INTEGER NOT NULL DEFAULT 1)"
    )
    conn.execute("INSERT OR IGNORE INTO library_meta (id, data_version) VALUES (1, 1)")
    conn.commit()
    verdict_path = db.parent / "rechunk" / "last-verdict.json"
    verdict_path.parent.mkdir(parents=True, exist_ok=True)
    verdict_path.write_text(_json.dumps({
        "pass": True,
        "baseline": {}, "staged": {},
        "snapshot": rc.snapshot_marker(conn),
    }))


class TestSwap:
    def test_swap_refused_without_pass_verdict(self, staged_db):
        conn, db = staged_db
        rc.ensure_staging(conn)
        with pytest.raises(rc.SwapRefused):
            rc.swap(db)

    def test_swap_refused_with_fail_verdict(self, staged_db):
        import json as _json

        conn, db = staged_db
        rc.ensure_staging(conn)
        p = db.parent / "rechunk" / "last-verdict.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_json.dumps({"pass": False, "snapshot": {}}))
        with pytest.raises(rc.SwapRefused):
            rc.swap(db)

    def test_swap_refused_with_unembedded_staging(self, staged_db, tmp_path):
        conn, db = staged_db
        _prep_passing_swap(conn, db, tmp_path)
        conn.execute("UPDATE chunks_staging SET embedding = NULL WHERE rowid = 1")
        conn.commit()
        with pytest.raises(rc.SwapRefused):
            rc.swap(db)

    def test_swap_replaces_chunks_bumps_version_drops_staging(self, staged_db, tmp_path):
        conn, db = staged_db
        _prep_passing_swap(conn, db, tmp_path)
        staged_n = conn.execute("SELECT COUNT(*) FROM chunks_staging").fetchone()[0]

        report = rc.swap(db)

        assert report["chunks"] == staged_n
        live = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert live == staged_n
        nulls = conn.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL").fetchone()[0]
        assert nulls == 0
        version = conn.execute("SELECT data_version FROM library_meta").fetchone()[0]
        assert version == 2 == report["data_version"]
        remaining = conn.execute(
            "SELECT name FROM sqlite_master WHERE name = 'chunks_staging'"
        ).fetchone()
        assert remaining is None
        assert (db.parent / db.name).exists()
        backups = list(db.parent.glob(f"{db.name}.backup-doctor-*"))
        assert backups, "swap must take a backup first"

    def test_swap_stages_delta_for_books_added_after_marker(self, staged_db, tmp_path):
        conn, db = staged_db
        _prep_passing_swap(conn, db, tmp_path)
        # simulate the worker approving a new book AFTER the marker
        ch_file = tmp_path / "ch3.md"
        ch_file.write_text(" ".join(f"Delta sentence {i} five words." for i in range(200)))
        conn.execute(
            "INSERT INTO chapters VALUES ('ch3', 'b1', 'Three', 3, ?, 1000)", (str(ch_file),)
        )
        conn.execute(
            "INSERT INTO chunks (id, chapter_id, book_id, chunk_index, content, word_count, embedding, created_at) "
            "VALUES ('new1', 'ch3', 'b1', 0, 'delta live', 2, ?, '2026-12-31')",
            (_blob([9, 9, 9, 9]),),
        )
        conn.commit()

        report = rc.swap(db, generator=FakeGenerator())

        assert report["delta_chapters"] == 1
        survived = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE chapter_id = 'ch3'"
        ).fetchone()[0]
        assert survived >= 1, "book approved mid-run must survive the swap"

    def test_mid_transaction_failure_leaves_chunks_and_version_untouched(
        self, staged_db, tmp_path, monkeypatch
    ):
        conn, db = staged_db
        _prep_passing_swap(conn, db, tmp_path)
        before_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

        def boom():
            raise RuntimeError("injected failure")

        monkeypatch.setattr(rc, "_pre_commit_hook", boom)
        with pytest.raises(RuntimeError, match="injected failure"):
            rc.swap(db)

        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == before_chunks
        assert conn.execute("SELECT data_version FROM library_meta").fetchone()[0] == 1
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE name = 'chunks_staging'"
        ).fetchone() is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_rechunk.py::TestSwap -v`
Expected: FAIL — `SwapRefused: swap not yet implemented` on the happy-path tests.

- [ ] **Step 3: Implement `swap`**

Replace the stub in `agentic_pipeline/library/rechunk.py`:

```python
def _pre_commit_hook() -> None:
    """Test seam: raised-from here simulates mid-transaction failure."""


def _stage_delta(conn, marker: dict, generator=None) -> int:
    """Stage chapters whose live chunks postdate the marker (decision 2)."""
    rows = conn.execute(
        "SELECT DISTINCT c.id, c.book_id, c.file_path FROM chapters c "
        "JOIN chunks k ON k.chapter_id = c.id "
        "WHERE k.created_at > ? "
        "   OR c.id NOT IN (SELECT DISTINCT chapter_id FROM chunks_staging)",
        (marker.get("max_created_at", ""),),
    ).fetchall()
    for ch in rows:
        try:
            stage_chapter(conn, ch, chunk_chapter)
        except (FileNotFoundError, IOError):
            _carry_live_chunks(conn, ch["id"])
    conn.commit()
    if rows:
        embed_pending(conn, generator=generator)
    return len(rows)


def swap(db_path, generator=None) -> dict:
    """Apply staged chunks to production. Refuses without a PASS verdict."""
    import sqlite3 as _sqlite3

    from agentic_pipeline.health.doctor import create_backup

    verdict = load_verdict(db_path)
    if verdict is None:
        raise SwapRefused("no verdict found — run `agentic-pipeline rechunk` first")
    if not verdict.get("pass"):
        raise SwapRefused("last eval verdict is FAIL — refusing to swap")

    conn = _sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = _sqlite3.Row
    # autocommit mode: python's implicit-transaction management would fight
    # the explicit BEGIN IMMEDIATE below ("cannot start a transaction within
    # a transaction" when a DML statement auto-begins first)
    conn.isolation_level = None
    try:
        staging_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE name = 'chunks_staging'"
        ).fetchone()
        if not staging_exists:
            raise SwapRefused("chunks_staging does not exist — run `agentic-pipeline rechunk` first")

        delta = _stage_delta(conn, verdict.get("snapshot", {}), generator=generator)

        unembedded = conn.execute(
            "SELECT COUNT(*) FROM chunks_staging WHERE embedding IS NULL"
        ).fetchone()[0]
        if unembedded:
            raise SwapRefused(f"{unembedded} staged chunks lack embeddings")

        staged_n = conn.execute("SELECT COUNT(*) FROM chunks_staging").fetchone()[0]
        if staged_n == 0:
            raise SwapRefused("staging is empty")

        backup = create_backup(db_path)

        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM chunks")
            conn.execute(
                "INSERT INTO chunks (id, chapter_id, book_id, chunk_index, content, "
                " word_count, embedding, embedding_model, content_hash) "
                "SELECT id, chapter_id, book_id, chunk_index, content, "
                " word_count, embedding, embedding_model, content_hash "
                "FROM chunks_staging"
            )
            conn.execute("DROP TABLE chunks_staging")
            conn.execute("UPDATE library_meta SET data_version = data_version + 1")
            _pre_commit_hook()
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        live_n = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        nulls = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE embedding IS NULL"
        ).fetchone()[0]
        if live_n != staged_n or nulls:
            raise RuntimeError(
                f"post-swap verification failed (count {live_n} vs {staged_n}, "
                f"{nulls} NULL embeddings) — restore backup: {backup}"
            )
        version = conn.execute(
            "SELECT data_version FROM library_meta WHERE id = 1"
        ).fetchone()[0]
    finally:
        conn.close()

    logger.info(f"swap complete: {live_n} chunks live, data_version={version}")
    return {
        "chunks": live_n,
        "backup": str(backup),
        "data_version": version,
        "delta_chapters": delta,
    }
```

- [ ] **Step 4: Run the full new-module suite**

Run: `python -m pytest tests/test_rechunk.py tests/test_doctor.py -v`
Expected: PASS (doctor tests confirm `create_backup` reuse didn't regress).

- [ ] **Step 5: Commit**

```bash
git add agentic_pipeline/library/rechunk.py tests/test_rechunk.py
git commit -m "feat: rechunk --swap — staleness delta, backup, atomic apply, version bump

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 11: Per-chapter aggregation in `semantic_search` (audit finding)

**Files:**
- Modify: `src/tools/semantic_search_tools.py` (lines 85–111)
- Test: `tests/test_semantic_search_tools.py` (extend; create if missing)

**Interfaces:**
- Consumes: `best_chunk_per_chapter(chunk_results)` from `src/utils/chunk_loader.py:99` (requires keys `chapter_id`, `similarity`, `content`; strips `chunk_id`/`chunk_index`; adds `excerpt`).
- Produces: `semantic_search` returns at most ONE result per chapter. Audit context: `hybrid_search_tools.py` (own `_best_chunk_per_chapter`), `discovery_tools.py`, `learning_tools.py`, `project_learning_tools.py`, `project_planning_tools.py`, `batch_ops.py` already aggregate — verified 2026-07-15. `semantic_search_tools.py` is the ONLY unaggregated path.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_semantic_search_tools.py` (create the file if it does not exist; if it exists, add this test using the file's existing tool-invocation helper pattern — tools registered via closures are reached through the registered FastMCP instance or a `_func` helper):

```python
"""semantic_search must not return two chunks of the same chapter."""

import numpy as np
import pytest


class FakeMcp:
    def __init__(self):
        self.tools = {}
        self.resources = {}

    def tool(self):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco

    def resource(self, pattern):
        def deco(fn):
            self.resources[pattern] = fn
            return fn
        return deco


@pytest.fixture
def semantic_tool(monkeypatch):
    from src.tools import semantic_search_tools as sst

    matrix = np.array(
        [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=np.float32
    )
    meta = [
        {"chunk_id": "k1", "chapter_id": "ch1", "book_id": "b1", "book_title": "B",
         "chapter_title": "C1", "chapter_number": 1, "content": "chunk one text"},
        {"chunk_id": "k2", "chapter_id": "ch1", "book_id": "b1", "book_title": "B",
         "chapter_title": "C1", "chapter_number": 1, "content": "chunk two text"},
        {"chunk_id": "k3", "chapter_id": "ch2", "book_id": "b1", "book_title": "B",
         "chapter_title": "C2", "chapter_number": 2, "content": "chunk three text"},
    ]
    monkeypatch.setattr(sst, "load_chunk_embeddings", lambda: (matrix, meta))

    class FakeGen:
        def generate(self, q):
            return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(sst, "_get_generator", lambda: FakeGen())

    mcp = FakeMcp()
    sst.register_semantic_search_tools(mcp)
    return mcp.tools["semantic_search"]


def test_adjacent_overlapping_chunks_do_not_crowd_results(semantic_tool):
    out = semantic_tool("query", limit=5, min_similarity=0.0, rerank=False)
    titles = [(r["chapter_title"],) for r in out["results"]]
    assert len(titles) == len(set(titles)), f"duplicate chapters in results: {titles}"
    assert len(out["results"]) == 2  # ch1 (best chunk) + ch2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_semantic_search_tools.py -v`
Expected: FAIL — 3 results, ch1 twice (k1 and k2 both surface).

- [ ] **Step 3: Add aggregation**

In `src/tools/semantic_search_tools.py`:

(a) Import (line 8 area): `from ..utils.chunk_loader import load_chunk_embeddings, best_chunk_per_chapter`

(b) In the candidate-build loop (lines 86–100), rename the content key
from `"chunk_content"` to `"content"` (that is what
`best_chunk_per_chapter` expects), then insert between the loop and the
rerank block (line 102):

```python
            # One result per chapter: overlapped windows from the same
            # chapter must not crowd the top-k (spec decision 3).
            # best_chunk_per_chapter keeps the best-scoring chunk per
            # chapter and attaches its text as "excerpt".
            candidates = best_chunk_per_chapter(candidates)
```

(c) Update the two downstream references: the rerank call's
`content_key="chunk_content"` becomes `content_key="excerpt"`, and the
format loop's `r["chunk_content"]` becomes `r["excerpt"]`. The final
response schema (book_title, chapter_title, chapter_number, similarity,
rerank_score, excerpt) is unchanged.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_semantic_search_tools.py tests/ -q -k "semantic or search"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tools/semantic_search_tools.py tests/test_semantic_search_tools.py
git commit -m "fix: semantic_search aggregates per chapter — overlap windows can't crowd top-k

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 12: Docs sync + full-suite gate

**Files:**
- Modify: `ref/cli-commands.md` (add `rechunk` to the library-maintenance group)
- Modify: `ref/module-map.md` (add `agentic_pipeline/library/rechunk.py`, `src/utils/retrieval_eval.py`, `src/utils/data_version.py`; note the cache versioning in the `cache.py` row)
- Modify: `ref/db-schema.md` (add `library_meta`; note `chunks_staging` as transient)
- Modify: `ref/mcp-tools.md` — NO changes (no new MCP tools; verify nothing claims otherwise)

**Interfaces:**
- Consumes: everything prior.
- Produces: docs matching reality; a green full suite.

- [ ] **Step 1: Update `ref/cli-commands.md`**

Add under the library/maintenance command group (match the file's existing table/entry format exactly — read it first):

```markdown
### `rechunk`
Re-chunk the whole library into `chunks_staging`, embed, and score against
the gold query sets (`eval/gold-queries*.json`). Hash-keyed: reruns reuse
existing staged embeddings; interruption is resumable by rerunning.
Production `chunks` untouched. Exit 0 = eval PASS, 1 = FAIL.

- `--fresh` — drop staging first
- `--yes` — skip the embedding-cost confirmation
- `--swap` — apply a PASSed staging: stages any delta (books approved since
  the eval), backs up the DB, atomically replaces `chunks`, bumps
  `library_meta.data_version` (running MCP servers self-invalidate their
  chunk cache). Refuses without a PASS verdict.
```

- [ ] **Step 2: Update `ref/module-map.md` and `ref/db-schema.md`**

module-map additions (match existing row format):

```markdown
| `agentic_pipeline/library/rechunk.py` | Staged re-chunking: hash-keyed staging, embedding, gate eval, atomic swap |
| `src/utils/retrieval_eval.py` | Gold-query retrieval eval: build-gold (claude -p), chapter-level hit@5/MRR, semantic + hybrid |
| `src/utils/data_version.py` | Reads library_meta.data_version for cache coherence |
```

db-schema additions:

```markdown
### library_meta
One-row table for cache coherence. `data_version` is bumped inside bulk-mutation
transactions (`rechunk --swap`); the MCP server's chunk-embedding cache
self-invalidates on mismatch.

| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | CHECK (id = 1) |
| data_version | INTEGER NOT NULL | seeded 1 |

### chunks_staging (transient)
Created by `agentic-pipeline rechunk`, dropped by `--swap`. Same columns as
`chunks` plus NOT NULL `content_hash`; `UNIQUE(chapter_id, chunk_index)`.
```

- [ ] **Step 3: Run the full suite**

Run: `source .venv/bin/activate && make test`
Expected: ALL tests pass (562 pre-existing + ~45 new). If `make test-fast` exists and is quicker, run both anyway — the plain run is the gate.

- [ ] **Step 4: Commit**

```bash
git add ref/cli-commands.md ref/module-map.md ref/db-schema.md
git commit -m "docs: sync ref docs for rechunk, retrieval eval, data_version

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Post-merge operational runbook (NOT part of the implementation tasks)

For the operator, after the PR merges — none of this runs in CI or tests:

1. `python scripts/retrieval_eval.py build-gold` (~14 min of `claude -p`).
2. Taylor reviews/edits `eval/gold-queries-manual.json` (decision 4), commit both gold files.
3. `agentic-pipeline rechunk` — confirm the ~$4–6 spend; 1–2 h embedding; verdict prints.
4. On PASS: `agentic-pipeline rechunk --swap`. Running MCP servers pick up the new matrix via data_version on their next search. On FAIL: no swap; tune params, rerun (hash reuse).
5. Rollback if ever needed: restore the printed `.backup-doctor-*` file.
```

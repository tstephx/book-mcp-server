# EPUB Chapter Splitter Fix — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the chapter splitter so oversized/under-split EPUB books are automatically re-split, and add a pipeline retry mechanism for edge cases.

**Architecture:** Two-project fix. In `book-ingestion-python`, add a post-split quality gate inside `ChapterSplitter.split_with_stats()` that re-splits oversized chapters at heading/paragraph boundaries. In `book-mcp-server`, change the VALIDATING stage to retry once with `force_fallback=True` before rejecting, and add a `reprocess` CLI command for the 78 flagged books.

**Tech Stack:** Python 3.12, SQLite, Click CLI, pytest

---

## Context

### Projects

| Project | Path | Venv |
|---------|------|------|
| book-ingestion-python | `/Users/taylorstephens/_Projects/book-ingestion-python/` | `.venv/` |
| book-mcp-server | `/Users/taylorstephens/_Projects/book-mcp-server/` | `.venv/` |

### Key Files (book-ingestion-python)

- `book_ingestion/processors/chapter_splitter.py` — `ChapterSplitter` class, `split_with_stats()` method (line 34-105), `_validate_chapter_sizes()` (line 302-334), `_fixed_size_split()` (line 336-355)
- `book_ingestion/processors/enhanced_pipeline.py` — `EnhancedPipeline._detect_chapters()` (line 288-389), orchestrates detection strategies
- `book_ingestion/processors/chapter_detector.py` — `AnchorMerger`, `CandidateExtractor`, confidence thresholds

### Key Files (book-mcp-server)

- `agentic_pipeline/orchestrator/orchestrator.py` — `_process_book()` VALIDATING block (line 269-290), `_run_processing()` (line 149-177), `_retry_one()` (line 352-386)
- `agentic_pipeline/adapters/processing_adapter.py` — `ProcessingAdapter.process_book()` (line 100-175)
- `agentic_pipeline/pipeline/states.py` — `TRANSITIONS` map, `NEEDS_RETRY` already valid from `VALIDATING`
- `agentic_pipeline/validation/extraction_validator.py` — `check_extraction_quality()`, threshold constants
- `agentic_pipeline/cli.py` — Click commands, `audit-quality` at line 839
- `agentic_pipeline/db/pipelines.py` — `PipelineRepository`, `increment_retry_count()` at line 258

### Thresholds (from extraction validator)

```python
MIN_CHAPTERS = 7
MAX_CHAPTER_WORDS = 20_000
MAX_TO_MEDIAN_RATIO = 4.0
MIN_CHAPTER_WORDS = 100  # warning only
MIN_TOTAL_WORDS = 5_000
MAX_DUPLICATE_RATIO = 0.10
```

---

## Task 1: Post-Split Quality Gate in ChapterSplitter (book-ingestion-python)

Add a `_quality_resplit()` method to `ChapterSplitter` that checks the split result against quality thresholds and re-splits oversized chapters at heading/paragraph boundaries.

**Files:**
- Modify: `book_ingestion/processors/chapter_splitter.py:89-105`
- Create: `tests/processors/test_quality_resplit.py`

**Step 1: Write failing tests**

Create `tests/processors/test_quality_resplit.py`:

```python
"""Tests for post-split quality gate in ChapterSplitter."""

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def splitter():
    """Create a ChapterSplitter with default config."""
    from book_ingestion.processors.chapter_splitter import ChapterSplitter

    config = MagicMock()
    config.chapter_detection = {
        "patterns": [],
        "min_words_per_chapter": 100,
        "max_words_per_chapter": 20000,
    }
    return ChapterSplitter(config)


def _make_chapter(book_id, num, title, word_count):
    """Helper to create a chapter dict with content of given word count."""
    content = " ".join(["word"] * word_count)
    return {
        "id": f"{book_id}-ch{num}",
        "book_id": book_id,
        "chapter_number": num,
        "title": title,
        "content": content,
        "word_count": word_count,
        "file_path": "",
    }


class TestQualityResplit:
    """Test the _quality_resplit method."""

    def test_good_split_unchanged(self, splitter):
        """Chapters that pass quality checks are returned unchanged."""
        chapters = [_make_chapter("b1", i, f"Ch {i}", 5000) for i in range(1, 11)]
        result = splitter._quality_resplit(chapters, "b1")
        assert len(result) == 10
        for i, ch in enumerate(result):
            assert ch["word_count"] == 5000

    def test_oversized_chapter_resplit(self, splitter):
        """A chapter exceeding MAX_CHAPTER_WORDS gets re-split."""
        chapters = [
            _make_chapter("b1", 1, "Ch 1", 5000),
            _make_chapter("b1", 2, "Ch 2", 30000),  # 30k > 20k threshold
            _make_chapter("b1", 3, "Ch 3", 5000),
        ]
        result = splitter._quality_resplit(chapters, "b1")
        # The oversized chapter should be split into 2+ sub-chapters
        assert len(result) > 3
        for ch in result:
            assert ch["word_count"] <= 20000

    def test_too_few_chapters_triggers_resplit(self, splitter):
        """When total chapters < MIN_CHAPTERS (7), oversized ones are re-split."""
        # 3 chapters of ~15k each — none exceeds 20k but only 3 chapters
        chapters = [_make_chapter("b1", i, f"Ch {i}", 15000) for i in range(1, 4)]
        result = splitter._quality_resplit(chapters, "b1")
        assert len(result) >= 7

    def test_lopsided_ratio_triggers_resplit(self, splitter):
        """When max/median ratio > 4x, the largest chapter is re-split."""
        chapters = [
            _make_chapter("b1", 1, "Ch 1", 2000),
            _make_chapter("b1", 2, "Ch 2", 2000),
            _make_chapter("b1", 3, "Ch 3", 2000),
            _make_chapter("b1", 4, "Ch 4", 2000),
            _make_chapter("b1", 5, "Ch 5", 2000),
            _make_chapter("b1", 6, "Ch 6", 2000),
            _make_chapter("b1", 7, "Ch 7", 2000),
            _make_chapter("b1", 8, "Big One", 18000),  # 18k / 2k median = 9x > 4x
        ]
        result = splitter._quality_resplit(chapters, "b1")
        word_counts = [ch["word_count"] for ch in result]
        max_wc = max(word_counts)
        from statistics import median
        med_wc = median(word_counts)
        assert med_wc == 0 or max_wc / med_wc <= 4.0

    def test_chapters_renumbered_after_resplit(self, splitter):
        """After re-splitting, chapters are sequentially renumbered."""
        chapters = [
            _make_chapter("b1", 1, "Ch 1", 5000),
            _make_chapter("b1", 2, "Ch 2", 30000),
        ]
        result = splitter._quality_resplit(chapters, "b1")
        for i, ch in enumerate(result):
            assert ch["chapter_number"] == i + 1
            assert ch["id"] == f"b1-ch{i + 1}"

    def test_resplit_preserves_good_chapters(self, splitter):
        """Only oversized chapters are re-split; good ones stay intact."""
        chapters = [
            _make_chapter("b1", 1, "Ch 1", 5000),
            _make_chapter("b1", 2, "Ch 2", 30000),
            _make_chapter("b1", 3, "Ch 3", 8000),
        ]
        result = splitter._quality_resplit(chapters, "b1")
        # First chapter should be unchanged
        assert result[0]["word_count"] == 5000
        # Last chapter (now renumbered) should be 8000
        assert result[-1]["word_count"] == 8000


class TestResplitAtBoundaries:
    """Test the _resplit_chapter method that splits at headings/paragraphs."""

    def test_splits_at_blank_lines(self, splitter):
        """Re-splitting uses double newlines as boundaries."""
        # Create content with clear paragraph breaks
        paragraphs = [" ".join(["word"] * 3000) for _ in range(8)]
        content = "\n\n".join(paragraphs)
        chapter = {
            "content": content,
            "word_count": 24000,
            "title": "Big Chapter",
        }
        result = splitter._resplit_chapter(chapter, "b1", max_words=20000)
        assert len(result) >= 2
        for sub in result:
            assert sub["word_count"] <= 20000

    def test_fallback_to_fixed_size(self, splitter):
        """If no paragraph breaks, falls back to fixed-size splitting."""
        # One giant paragraph with no breaks
        content = " ".join(["word"] * 25000)
        chapter = {
            "content": content,
            "word_count": 25000,
            "title": "No Breaks",
        }
        result = splitter._resplit_chapter(chapter, "b1", max_words=20000)
        assert len(result) >= 2
        for sub in result:
            assert sub["word_count"] <= 20000
```

**Step 2: Run tests to verify they fail**

```bash
cd /Users/taylorstephens/_Projects/book-ingestion-python
source .venv/bin/activate
python -m pytest tests/processors/test_quality_resplit.py -v
```
Expected: FAIL — `_quality_resplit` and `_resplit_chapter` don't exist yet.

**Step 3: Implement `_quality_resplit()` and `_resplit_chapter()`**

In `book_ingestion/processors/chapter_splitter.py`, add these methods to the `ChapterSplitter` class and call `_quality_resplit()` from `split_with_stats()`.

Add at the top of the file:
```python
from statistics import median
```

Replace lines 89-105 of `split_with_stats()` (after anchor merge) with:

```python
        # Build chapters from anchors
        if anchors:
            chapters = self._build_chapters_from_anchors(text, book_id, anchors)
        else:
            chapters = self._fixed_size_split(text, book_id)
            stats.method = 'fallback'
            stats.confidence = 'low'

        # Validate chapter sizes (existing filter)
        chapters = self._validate_chapter_sizes(chapters, book_id)

        if len(chapters) == 0:
            chapters = self._fixed_size_split(text, book_id)
            stats.method = 'fallback'
            stats.confidence = 'low'

        # Post-split quality gate: re-split oversized chapters
        chapters = self._quality_resplit(chapters, book_id)

        return {'chapters': chapters, 'stats': stats}
```

Add these two methods to the class (after `_fixed_size_split`):

```python
    # Quality gate thresholds (match agentic pipeline extraction validator)
    _QG_MIN_CHAPTERS = 7
    _QG_MAX_CHAPTER_WORDS = 20_000
    _QG_MAX_TO_MEDIAN_RATIO = 4.0

    def _quality_resplit(self, chapters: List[Dict], book_id: str) -> List[Dict]:
        """Post-split quality gate: re-split chapters that fail quality checks.

        Checks:
        1. Any chapter exceeding _QG_MAX_CHAPTER_WORDS → re-split it
        2. If total chapters < _QG_MIN_CHAPTERS → re-split largest chapters
        3. If max/median ratio > _QG_MAX_TO_MEDIAN_RATIO → re-split the largest

        Only modifies chapters that violate thresholds. Good chapters stay intact.
        """
        if not chapters:
            return chapters

        max_words = self._QG_MAX_CHAPTER_WORDS
        changed = True

        # Iterate until stable (max 3 rounds to avoid infinite loops)
        for _ in range(3):
            if not changed:
                break
            changed = False

            # Check 1: Oversized chapters
            new_chapters = []
            for ch in chapters:
                if ch["word_count"] > max_words:
                    sub_chapters = self._resplit_chapter(ch, book_id, max_words)
                    new_chapters.extend(sub_chapters)
                    changed = True
                else:
                    new_chapters.append(ch)
            chapters = new_chapters

            # Check 2: Too few chapters — re-split the largest to increase count
            if len(chapters) < self._QG_MIN_CHAPTERS:
                chapters = sorted(chapters, key=lambda c: c["word_count"], reverse=True)
                new_chapters = []
                for ch in chapters:
                    if len(new_chapters) + (len(chapters) - len(new_chapters)) < self._QG_MIN_CHAPTERS and ch["word_count"] > max_words // 2:
                        target = max(max_words // 2, ch["word_count"] // 2)
                        sub_chapters = self._resplit_chapter(ch, book_id, target)
                        new_chapters.extend(sub_chapters)
                        changed = True
                    else:
                        new_chapters.append(ch)
                chapters = new_chapters

            # Check 3: Lopsided ratio
            word_counts = [ch["word_count"] for ch in chapters]
            if word_counts:
                med_wc = median(word_counts)
                if med_wc > 0:
                    max_wc = max(word_counts)
                    if max_wc / med_wc > self._QG_MAX_TO_MEDIAN_RATIO:
                        threshold = int(med_wc * self._QG_MAX_TO_MEDIAN_RATIO)
                        new_chapters = []
                        for ch in chapters:
                            if ch["word_count"] > threshold:
                                sub_chapters = self._resplit_chapter(ch, book_id, threshold)
                                new_chapters.extend(sub_chapters)
                                changed = True
                            else:
                                new_chapters.append(ch)
                        chapters = new_chapters

        # Renumber all chapters
        for i, ch in enumerate(chapters):
            ch["chapter_number"] = i + 1
            ch["id"] = f"{book_id}-ch{i + 1}"

        return chapters

    def _resplit_chapter(self, chapter: Dict, book_id: str, max_words: int) -> List[Dict]:
        """Re-split a single oversized chapter at paragraph or heading boundaries.

        Strategy:
        1. Split at double-newline (paragraph) boundaries
        2. Greedily merge paragraphs into sub-chapters up to max_words
        3. If no paragraph boundaries, fall back to word-count chunking
        """
        content = chapter.get("content", "")
        title = chapter.get("title", "Section")

        # Try splitting at paragraph boundaries
        paragraphs = re.split(r'\n\s*\n', content)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if len(paragraphs) <= 1:
            # No paragraph boundaries — word-count chunk
            words = content.split()
            sub_chapters = []
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i:i + max_words])
                sub_chapters.append({
                    "id": "",
                    "book_id": book_id,
                    "chapter_number": 0,
                    "title": f"{title} (part {len(sub_chapters) + 1})",
                    "content": chunk,
                    "word_count": len(chunk.split()),
                    "file_path": "",
                })
            return sub_chapters

        # Greedily merge paragraphs into sub-chapters
        sub_chapters = []
        current_parts = []
        current_words = 0

        for para in paragraphs:
            para_words = len(para.split())
            if current_words + para_words > max_words and current_parts:
                merged = "\n\n".join(current_parts)
                sub_chapters.append({
                    "id": "",
                    "book_id": book_id,
                    "chapter_number": 0,
                    "title": f"{title} (part {len(sub_chapters) + 1})",
                    "content": merged,
                    "word_count": len(merged.split()),
                    "file_path": "",
                })
                current_parts = []
                current_words = 0

            current_parts.append(para)
            current_words += para_words

        # Flush remaining
        if current_parts:
            merged = "\n\n".join(current_parts)
            sub_chapters.append({
                "id": "",
                "book_id": book_id,
                "chapter_number": 0,
                "title": f"{title} (part {len(sub_chapters) + 1})" if len(sub_chapters) > 0 else title,
                "content": merged,
                "word_count": len(merged.split()),
                "file_path": "",
            })

        return sub_chapters if sub_chapters else [{
            "id": "",
            "book_id": book_id,
            "chapter_number": 0,
            "title": title,
            "content": content,
            "word_count": len(content.split()),
            "file_path": "",
        }]
```

**Step 4: Run tests to verify they pass**

```bash
cd /Users/taylorstephens/_Projects/book-ingestion-python
python -m pytest tests/processors/test_quality_resplit.py -v
```
Expected: All tests PASS.

**Step 5: Run existing tests to check for regressions**

```bash
cd /Users/taylorstephens/_Projects/book-ingestion-python
python -m pytest tests/ -v --tb=short 2>&1 | tail -30
```
Expected: No new failures.

**Step 6: Commit**

```bash
cd /Users/taylorstephens/_Projects/book-ingestion-python
git add book_ingestion/processors/chapter_splitter.py tests/processors/test_quality_resplit.py
git commit -m "feat: add post-split quality gate to re-split oversized chapters"
```

---

## Task 2: Add `force_fallback` Parameter to ProcessingAdapter (book-mcp-server)

Thread a `force_fallback` flag through `ProcessingAdapter` → `BookIngestionApp` → `EnhancedPipeline` so the pipeline can request heading-based splitting on retry.

**Files:**
- Modify: `agentic_pipeline/adapters/processing_adapter.py:100-130`
- Modify: `book_ingestion/processors/chapter_splitter.py` (add `force_fallback` param)
- Modify: `book_ingestion/processors/enhanced_pipeline.py:164-286` (pass through)
- Modify: `book_ingestion/bootstrap.py` (pass through)
- Create: `tests/test_force_fallback.py` (in book-mcp-server)

**Step 1: Write failing test**

Create `tests/test_force_fallback.py` in book-mcp-server:

```python
"""Tests for force_fallback parameter in ProcessingAdapter."""

import pytest
from unittest.mock import patch, MagicMock


def test_process_book_passes_force_fallback():
    """ProcessingAdapter.process_book forwards force_fallback to BookIngestionApp."""
    from agentic_pipeline.adapters.processing_adapter import ProcessingAdapter

    mock_app = MagicMock()
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.book_id = "test-id"
    mock_result.llm_fallback_used = False
    mock_result.pipeline_result.quality_report.quality_score = 80
    mock_result.pipeline_result.detection_confidence = 0.9
    mock_result.pipeline_result.detection_method = "test"
    mock_result.pipeline_result.needs_review = False
    mock_result.pipeline_result.warnings = []
    mock_result.pipeline_result.chapters = []
    mock_app.process.return_value = mock_result

    with patch("agentic_pipeline.adapters.processing_adapter.BookIngestionApp") as MockApp:
        MockApp.create.return_value = mock_app
        adapter = ProcessingAdapter(db_path="/tmp/test.db")
        adapter.process_book("/tmp/book.epub", force_fallback=True)

        mock_app.process.assert_called_once()
        call_kwargs = mock_app.process.call_args
        assert call_kwargs.kwargs.get("force_fallback") is True or call_kwargs[1].get("force_fallback") is True


def test_process_book_default_no_force_fallback():
    """By default, force_fallback is not passed (or False)."""
    from agentic_pipeline.adapters.processing_adapter import ProcessingAdapter

    mock_app = MagicMock()
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.book_id = "test-id"
    mock_result.llm_fallback_used = False
    mock_result.pipeline_result.quality_report.quality_score = 80
    mock_result.pipeline_result.detection_confidence = 0.9
    mock_result.pipeline_result.detection_method = "test"
    mock_result.pipeline_result.needs_review = False
    mock_result.pipeline_result.warnings = []
    mock_result.pipeline_result.chapters = []
    mock_app.process.return_value = mock_result

    with patch("agentic_pipeline.adapters.processing_adapter.BookIngestionApp") as MockApp:
        MockApp.create.return_value = mock_app
        adapter = ProcessingAdapter(db_path="/tmp/test.db")
        adapter.process_book("/tmp/book.epub")

        call_kwargs = mock_app.process.call_args
        # force_fallback should be False or absent
        ff = call_kwargs.kwargs.get("force_fallback", call_kwargs[1].get("force_fallback", False))
        assert ff is False
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/taylorstephens/_Projects/book-mcp-server
source .venv/bin/activate
python -m pytest tests/test_force_fallback.py -v
```
Expected: FAIL — `process_book()` doesn't accept `force_fallback`.

**Step 3: Thread `force_fallback` through the chain**

3a. In `agentic_pipeline/adapters/processing_adapter.py`, add `force_fallback` param to `process_book()`:

```python
    def process_book(
        self,
        book_path: str,
        title: Optional[str] = None,
        author: Optional[str] = None,
        book_id: Optional[str] = None,
        save_to_storage: bool = True,
        force_fallback: bool = False,
    ) -> ProcessingResult:
```

And pass it to `self._app.process(...)`:

```python
            result = self._app.process(
                file_path=book_path,
                title=title,
                author=author,
                book_id=book_id,
                save_to_storage=save_to_storage,
                force_fallback=force_fallback,
            )
```

3b. In `book_ingestion/bootstrap.py`, add `force_fallback` param to `BookIngestionApp.process()` and pass it to `EnhancedPipeline.process_book()`. Find the `process()` method and add the parameter:

```python
    def process(self, file_path, ..., force_fallback=False):
```

Pass it through to `pipeline.process_book(..., force_fallback=force_fallback)`.

3c. In `book_ingestion/processors/enhanced_pipeline.py`, add `force_fallback` param to `process_book()` and `_detect_chapters()`. When `force_fallback=True`, skip anchor/TOC detection and go straight to fixed-size splitting:

In `_detect_chapters()`, after the method signature, add:

```python
        if force_fallback:
            logger.info(f"force_fallback=True: skipping anchor detection, using fixed-size split")
            from .chapter_splitter import ChapterSplitter
            from ..utils.config import Config
            config = Config()
            splitter = ChapterSplitter(config)
            chapters = splitter._fixed_size_split(text, book_id)
            return ChapterDetectionResult(
                chapters=chapters,
                method="force_fallback",
                confidence=0.5,
                toc_chapters_found=0,
                semantic_boundaries_found=0,
                merge_suggestions=[],
            )
```

3d. In `book_ingestion/processors/chapter_splitter.py`, the `_fixed_size_split()` + `_quality_resplit()` chain will still apply to force_fallback results, producing reasonable output.

**Step 4: Run tests to verify they pass**

```bash
cd /Users/taylorstephens/_Projects/book-mcp-server
python -m pytest tests/test_force_fallback.py -v
```
Expected: PASS.

**Step 5: Run full test suite in both projects**

```bash
cd /Users/taylorstephens/_Projects/book-ingestion-python && python -m pytest tests/ -v --tb=short 2>&1 | tail -10
cd /Users/taylorstephens/_Projects/book-mcp-server && python -m pytest tests/ -v --tb=short 2>&1 | tail -10
```
Expected: No regressions.

**Step 6: Commit in both projects**

```bash
cd /Users/taylorstephens/_Projects/book-ingestion-python
git add -A && git commit -m "feat: add force_fallback parameter to skip anchor detection"

cd /Users/taylorstephens/_Projects/book-mcp-server
git add tests/test_force_fallback.py agentic_pipeline/adapters/processing_adapter.py
git commit -m "feat: add force_fallback support to ProcessingAdapter"
```

---

## Task 3: Pipeline Retry on Validation Failure (book-mcp-server)

Change the VALIDATING block in the orchestrator to retry once with `force_fallback=True` instead of immediately rejecting.

**Files:**
- Modify: `agentic_pipeline/orchestrator/orchestrator.py:269-290`
- Modify: `tests/test_orchestrator_integration.py`
- Create: `tests/test_validation_retry.py`

**Step 1: Write failing tests**

Create `tests/test_validation_retry.py`:

```python
"""Tests for validation retry mechanism."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


@pytest.fixture
def db_path():
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def sample_book(tmp_path):
    book = tmp_path / "sample.txt"
    book.write_text("Chapter 1: Intro\n\nSome content here for testing.\n" * 100)
    return str(book)


@pytest.fixture
def config(db_path):
    from agentic_pipeline.config import OrchestratorConfig

    return OrchestratorConfig(
        db_path=db_path,
        processing_timeout=10,
        embedding_timeout=5,
        confidence_threshold=0.7,
    )


def _mock_profile():
    from agentic_pipeline.agents.classifier_types import BookProfile, BookType

    return BookProfile(
        book_type=BookType.TECHNICAL_TUTORIAL,
        confidence=0.9,
        suggested_tags=["python"],
        reasoning="Test",
    )


def _mock_processing_result():
    return {
        "book_id": "test-id",
        "quality_score": 85,
        "detection_confidence": 0.9,
        "detection_method": "mock",
        "needs_review": False,
        "warnings": [],
        "chapter_count": 10,
        "word_count": 50000,
        "llm_fallback_used": False,
    }


def test_first_validation_failure_retries(config, sample_book):
    """First validation failure triggers retry with force_fallback, not immediate rejection."""
    from agentic_pipeline.orchestrator import Orchestrator
    from agentic_pipeline.validation import ValidationResult

    orchestrator = Orchestrator(config)

    fail_validation = ValidationResult(
        passed=False,
        reasons=["Too few chapters: 3 (minimum 7 required)"],
        warnings=[],
        metrics={"chapter_count": 3},
    )
    pass_validation = ValidationResult(
        passed=True, reasons=[], warnings=[], metrics={"chapter_count": 10}
    )

    # First call fails, second call (retry) passes
    validation_calls = [fail_validation, pass_validation]

    mock_adapter_cls = MagicMock()
    mock_adapter_instance = mock_adapter_cls.return_value
    mock_embed_result = MagicMock(success=True, chapters_processed=10, error=None)
    mock_adapter_instance.generate_embeddings.return_value = mock_embed_result

    with patch.object(orchestrator.classifier, "classify", return_value=_mock_profile()):
        with patch.object(orchestrator, "_run_processing", return_value=_mock_processing_result()):
            with patch(
                "agentic_pipeline.orchestrator.orchestrator.ExtractionValidator.validate",
                side_effect=validation_calls,
            ):
                with patch(
                    "agentic_pipeline.adapters.processing_adapter.ProcessingAdapter",
                    mock_adapter_cls,
                ):
                    result = orchestrator.process_one(sample_book)

    assert result["state"] == "complete"


def test_both_attempts_fail_rejects(config, sample_book):
    """If both normal and fallback validation fail, book is rejected."""
    from agentic_pipeline.orchestrator import Orchestrator
    from agentic_pipeline.validation import ValidationResult

    orchestrator = Orchestrator(config)

    fail_validation = ValidationResult(
        passed=False,
        reasons=["Too few chapters: 3 (minimum 7 required)"],
        warnings=[],
        metrics={"chapter_count": 3},
    )

    with patch.object(orchestrator.classifier, "classify", return_value=_mock_profile()):
        with patch.object(orchestrator, "_run_processing", return_value=_mock_processing_result()):
            with patch(
                "agentic_pipeline.orchestrator.orchestrator.ExtractionValidator.validate",
                return_value=fail_validation,
            ):
                result = orchestrator.process_one(sample_book)

    assert result["state"] == "rejected"
    assert "Too few chapters" in result["reason"]


def test_retry_uses_force_fallback(config, sample_book):
    """On retry, _run_processing is called with force_fallback=True."""
    from agentic_pipeline.orchestrator import Orchestrator
    from agentic_pipeline.validation import ValidationResult

    orchestrator = Orchestrator(config)

    fail_validation = ValidationResult(
        passed=False,
        reasons=["Too few chapters"],
        warnings=[],
        metrics={"chapter_count": 3},
    )

    call_args_list = []
    original_run = orchestrator._run_processing

    def tracking_run(*args, **kwargs):
        call_args_list.append(kwargs)
        return _mock_processing_result()

    with patch.object(orchestrator.classifier, "classify", return_value=_mock_profile()):
        with patch.object(orchestrator, "_run_processing", side_effect=tracking_run):
            with patch(
                "agentic_pipeline.orchestrator.orchestrator.ExtractionValidator.validate",
                return_value=fail_validation,
            ):
                result = orchestrator.process_one(sample_book)

    # Should have been called twice: once normal, once with force_fallback
    assert len(call_args_list) == 2
    assert call_args_list[1].get("force_fallback") is True
```

**Step 2: Run tests to verify they fail**

```bash
cd /Users/taylorstephens/_Projects/book-mcp-server
python -m pytest tests/test_validation_retry.py -v
```
Expected: FAIL — retry logic doesn't exist yet.

**Step 3: Implement retry logic in orchestrator**

In `agentic_pipeline/orchestrator/orchestrator.py`, replace the VALIDATING block (lines 269-290) and update `_run_processing()` to accept `force_fallback`:

Update `_run_processing` signature:

```python
    def _run_processing(self, book_path: str, book_id: Optional[str] = None, force_fallback: bool = False) -> dict:
        result = self.processing_adapter.process_book(
            book_path=book_path,
            book_id=book_id,
            force_fallback=force_fallback,
        )
```

Replace the VALIDATING block with:

```python
        # VALIDATING
        self._transition(pipeline_id, PipelineState.VALIDATING)
        validator = ExtractionValidator()
        validation = validator.validate(
            book_id=pipeline_id, db_path=str(self.config.db_path)
        )

        if not validation.passed:
            # Check if this is the first attempt — retry with force_fallback
            pipeline_record = self.repo.get(pipeline_id)
            retry_count = pipeline_record.get("retry_count", 0) if pipeline_record else 0

            if retry_count == 0:
                self.logger.error(pipeline_id, "ValidationFailed", f"Retrying with force_fallback: {'; '.join(validation.reasons)}")
                self.repo.increment_retry_count(pipeline_id)

                # Re-process with force_fallback
                self._transition(pipeline_id, PipelineState.NEEDS_RETRY)
                self._transition(pipeline_id, PipelineState.PROCESSING)
                try:
                    processing_result = self._run_processing(book_path, book_id=pipeline_id, force_fallback=True)
                except (ProcessingError, PipelineTimeoutError) as e:
                    self.logger.error(pipeline_id, type(e).__name__, str(e))
                    self._transition(pipeline_id, PipelineState.NEEDS_RETRY)
                    return {"pipeline_id": pipeline_id, "state": PipelineState.NEEDS_RETRY.value, "error": str(e)}

                self.repo.update_processing_result(pipeline_id, {
                    "quality_score": processing_result.get("quality_score"),
                    "detection_confidence": processing_result.get("detection_confidence"),
                    "detection_method": processing_result.get("detection_method"),
                    "chapter_count": processing_result.get("chapter_count"),
                    "word_count": processing_result.get("word_count"),
                    "warnings": processing_result.get("warnings", []),
                    "llm_fallback_used": processing_result.get("llm_fallback_used", False),
                })

                # Re-validate
                self._transition(pipeline_id, PipelineState.VALIDATING)
                validation = validator.validate(
                    book_id=pipeline_id, db_path=str(self.config.db_path)
                )

            if not validation.passed:
                reason = "; ".join(validation.reasons)
                self.logger.error(pipeline_id, "ValidationFailed", reason)
                self.logger.state_transition(pipeline_id, PipelineState.VALIDATING.value, PipelineState.REJECTED.value)
                self.repo.update_state(
                    pipeline_id,
                    PipelineState.REJECTED,
                    error_details={"validation_reasons": validation.reasons, "metrics": validation.metrics},
                )
                return {
                    "pipeline_id": pipeline_id,
                    "state": PipelineState.REJECTED.value,
                    "reason": reason,
                    "metrics": validation.metrics,
                }
```

Note: The TRANSITIONS map already allows `VALIDATING → NEEDS_RETRY` and `NEEDS_RETRY → PROCESSING`.

**Step 4: Run tests to verify they pass**

```bash
cd /Users/taylorstephens/_Projects/book-mcp-server
python -m pytest tests/test_validation_retry.py -v
```
Expected: PASS.

**Step 5: Run full test suite**

```bash
cd /Users/taylorstephens/_Projects/book-mcp-server
python -m pytest tests/ -v --tb=short 2>&1 | tail -15
```
Expected: No regressions. Existing orchestrator tests may need the `ExtractionValidator.validate` mock updated to account for the retry path.

**Step 6: Commit**

```bash
cd /Users/taylorstephens/_Projects/book-mcp-server
git add agentic_pipeline/orchestrator/orchestrator.py tests/test_validation_retry.py
git commit -m "feat: retry with force_fallback on first validation failure"
```

---

## Task 4: Reprocess CLI Command (book-mcp-server)

Add `agentic-pipeline reprocess` command that re-queues flagged books.

**Files:**
- Modify: `agentic_pipeline/cli.py` (after line 905)
- Create: `tests/test_reprocess_cli.py`

**Step 1: Write failing tests**

Create `tests/test_reprocess_cli.py`:

```python
"""Tests for reprocess CLI command."""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch
from click.testing import CliRunner

from agentic_pipeline.cli import main


@pytest.fixture
def db_path():
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def db_with_books(db_path):
    """Create a DB with books and chapters for testing."""
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS books (
            id TEXT PRIMARY KEY, title TEXT, author TEXT, source_file TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chapters (
            id TEXT PRIMARY KEY, book_id TEXT, chapter_number INTEGER,
            title TEXT, word_count INTEGER, content_hash TEXT,
            file_path TEXT, embedding BLOB, embedding_model TEXT
        )
    """)
    # Good book: 10 chapters, ~5k words each
    conn.execute("INSERT INTO books VALUES ('good-1', 'Good Book', 'Author', '/tmp/good.epub')")
    for i in range(1, 11):
        conn.execute(
            "INSERT INTO chapters VALUES (?, 'good-1', ?, ?, 5000, ?, '', NULL, NULL)",
            (f"good-1-ch{i}", i, f"Chapter {i}", f"hash-good-{i}"),
        )

    # Bad book: 2 chapters, huge
    conn.execute("INSERT INTO books VALUES ('bad-1', 'Bad Book', 'Author', '/tmp/bad.epub')")
    for i in range(1, 3):
        conn.execute(
            "INSERT INTO chapters VALUES (?, 'bad-1', ?, ?, 25000, ?, '', NULL, NULL)",
            (f"bad-1-ch{i}", i, f"Chapter {i}", f"hash-bad-{i}"),
        )

    conn.commit()
    conn.close()
    return db_path


def test_reprocess_dry_run(db_with_books):
    """Dry run shows what would be reprocessed without modifying anything."""
    runner = CliRunner()
    with patch("agentic_pipeline.cli.get_db_path", return_value=str(db_with_books)):
        result = runner.invoke(main, ["reprocess", "--flagged"])

    assert result.exit_code == 0
    assert "bad-1" in result.output or "Bad Book" in result.output
    assert "dry run" in result.output.lower() or "DRY RUN" in result.output


def test_reprocess_dry_run_json(db_with_books):
    """Dry run with --json outputs valid JSON."""
    runner = CliRunner()
    with patch("agentic_pipeline.cli.get_db_path", return_value=str(db_with_books)):
        result = runner.invoke(main, ["reprocess", "--flagged", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "flagged" in data
    assert data["flagged"] >= 1
```

**Step 2: Run tests to verify they fail**

```bash
cd /Users/taylorstephens/_Projects/book-mcp-server
python -m pytest tests/test_reprocess_cli.py -v
```
Expected: FAIL — `reprocess` command doesn't exist.

**Step 3: Implement the `reprocess` command**

In `agentic_pipeline/cli.py`, add after the `audit_quality` command (line 905):

```python
@main.command("reprocess")
@click.option("--flagged", is_flag=True, help="Reprocess books that fail quality checks")
@click.option("--execute", is_flag=True, help="Actually reprocess (default is dry run)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def reprocess(flagged: bool, execute: bool, as_json: bool):
    """Reprocess books that fail extraction quality checks.

    By default runs as dry run showing what would be reprocessed.
    Pass --execute to actually delete chapters and re-queue.
    """
    import json as json_module
    from .db.config import get_db_path
    from .db.connection import get_pipeline_db
    from .validation import check_extraction_quality

    if not flagged:
        console.print("[red]Specify --flagged to reprocess books failing quality checks.[/red]")
        return

    db_path = get_db_path()

    with get_pipeline_db(db_path) as conn:
        books = conn.execute("SELECT id, title, source_file FROM books").fetchall()

        flagged_books = []
        for book in books:
            rows = conn.execute(
                "SELECT title, word_count, content_hash FROM chapters WHERE book_id = ? ORDER BY chapter_number",
                (book["id"],),
            ).fetchall()

            if not rows:
                continue

            chapter_count = len(rows)
            word_counts = [r["word_count"] or 0 for r in rows]
            titles = [r["title"] or "" for r in rows]
            content_hashes = [r["content_hash"] or "" for r in rows]

            validation = check_extraction_quality(chapter_count, word_counts, titles, content_hashes)

            if not validation.passed:
                flagged_books.append({
                    "book_id": book["id"],
                    "title": book["title"],
                    "source_file": book["source_file"],
                    "chapter_count": chapter_count,
                    "reasons": validation.reasons,
                })

    if as_json:
        output = {
            "total": len(books),
            "flagged": len(flagged_books),
            "mode": "execute" if execute else "dry_run",
            "books": flagged_books,
        }
        console.print(json_module.dumps(output, indent=2))
        return

    if not flagged_books:
        console.print("[green]No books fail quality checks.[/green]")
        return

    if not execute:
        console.print(f"\n[bold yellow]DRY RUN[/bold yellow] — {len(flagged_books)} books would be reprocessed:\n")
        table = Table(title="Books to Reprocess")
        table.add_column("Title", max_width=45)
        table.add_column("Ch", justify="right")
        table.add_column("Issues")

        for book in flagged_books:
            table.add_row(
                book["title"][:45],
                str(book["chapter_count"]),
                "; ".join(book["reasons"][:2]),
            )

        console.print(table)
        console.print("\n[yellow]Run with --execute to actually reprocess.[/yellow]")
        return

    # Execute mode: delete chapters and re-queue
    from .db.pipelines import PipelineRepository
    from .pipeline.states import PipelineState

    repo = PipelineRepository(db_path)
    reprocessed = 0

    for book in flagged_books:
        book_id = book["book_id"]
        source_file = book["source_file"]

        if not source_file or not Path(source_file).exists():
            console.print(f"[yellow]Skipping {book['title']}: source file not found[/yellow]")
            continue

        with get_pipeline_db(db_path) as conn:
            # Backup old chapter count in audit
            old_count = book["chapter_count"]
            # Delete old chapters
            conn.execute("DELETE FROM chapters WHERE book_id = ?", (book_id,))
            # Delete old book record (will be recreated by pipeline)
            conn.execute("DELETE FROM books WHERE id = ?", (book_id,))
            conn.commit()

        # Reset pipeline state to DETECTED for re-processing
        try:
            existing = repo.get(book_id)
            if existing:
                # Delete old pipeline record — create fresh
                with get_pipeline_db(db_path) as conn:
                    conn.execute("DELETE FROM processing_pipelines WHERE id = ?", (book_id,))
                    conn.execute("DELETE FROM pipeline_state_history WHERE pipeline_id = ?", (book_id,))
                    conn.commit()

            import hashlib
            hasher = hashlib.sha256()
            with open(source_file, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            content_hash = hasher.hexdigest()

            repo.create(source_file, content_hash)
            reprocessed += 1
            console.print(f"[green]Re-queued: {book['title']} (was {old_count} chapters)[/green]")
        except Exception as e:
            console.print(f"[red]Failed: {book['title']}: {e}[/red]")

    console.print(f"\n[bold green]Reprocessed {reprocessed}/{len(flagged_books)} books.[/bold green]")
```

**Step 4: Run tests to verify they pass**

```bash
cd /Users/taylorstephens/_Projects/book-mcp-server
python -m pytest tests/test_reprocess_cli.py -v
```
Expected: PASS.

**Step 5: Run full test suite**

```bash
cd /Users/taylorstephens/_Projects/book-mcp-server
python -m pytest tests/ -v --tb=short 2>&1 | tail -15
```
Expected: No regressions.

**Step 6: Commit**

```bash
cd /Users/taylorstephens/_Projects/book-mcp-server
git add agentic_pipeline/cli.py tests/test_reprocess_cli.py
git commit -m "feat: add reprocess CLI command for re-queuing flagged books"
```

---

## Task 5: Run Reprocessing (Manual)

After all code changes are committed and tests pass:

**Step 1: Dry-run the reprocess command**

```bash
cd /Users/taylorstephens/_Projects/book-mcp-server
source .venv/bin/activate
agentic-pipeline reprocess --flagged
```

Review the output. Verify the flagged count matches expectations (~78 books).

**Step 2: Execute the reprocess**

```bash
agentic-pipeline reprocess --flagged --execute
```

**Step 3: Start the worker to process re-queued books**

```bash
agentic-pipeline worker
```

Monitor output. Books should go through the improved pipeline (quality gate in splitter + retry on validation failure).

**Step 4: Re-run audit to verify improvement**

```bash
agentic-pipeline audit-quality
```

Compare flagged count against the original 78.

# Backlog Cleanup: DB Refactor, Test Coverage, Planning Tools LLM

> **Date:** 2026-02-11
> **Scope:** 3 backlog items across 6 sessions
> **Status:** Design approved, not started

## Overview

Three improvements to the book-mcp-server codebase, sequenced by dependency:

1. **DB context manager refactor** (1 session) — Eliminate duplicate connection code in agentic_pipeline/
2. **Full src/ test coverage** (4 sessions) — Systematic tests for all 36 src/ modules
3. **Planning tools LLM enhancement** (1 session) — Hybrid template + LLM output for planning tools

## 1. DB Context Manager Refactor

### Goal

Eliminate 6+ duplicate `_connect()` methods and 44 manual `try/finally` blocks across `agentic_pipeline/` by introducing a shared context manager.

### Design

**New file: `agentic_pipeline/db/connection.py`**

```python
@contextmanager
def get_pipeline_db(db_path: str = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for pipeline database connections."""
    if db_path is None:
        db_path = get_db_path()
    conn = None
    try:
        conn = sqlite3.connect(str(db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        if conn:
            conn.close()
```

### Migration per class

- Remove `_connect()` method and `self.db_path` field
- Replace `conn = self._connect()` / `try:` / `finally: conn.close()` with `with get_pipeline_db(self.db_path) as conn:`
- Classes that need a configurable path (e.g. tests) pass `db_path` to the constructor, which passes it through to `get_pipeline_db()`

### Files to change (15)

`pipelines.py`, `trail.py`, `monitor.py`, `stuck_detector.py`, `filters.py`, `operations.py`, `config.py`, `metrics.py`, `calibration.py`, `spot_check.py`, `actions.py`, `processing_adapter.py`, `status.py`, `manager.py`, `validator.py`

### Not changing

`src/database.py` stays as-is. No cross-package dependency. The two packages are intentionally separate (agentic_pipeline is a CLI tool, src/ is an MCP server).

### Verification

Existing 210 agentic_pipeline tests must still pass. No new tests needed — the context manager behavior is validated by the existing tests continuing to work.

---

## 2. Full src/ Test Coverage

### Goal

Systematic tests for all 36 src/ modules (~9,400 lines), organized in 4 priority batches. Target: ~150-200 new tests.

### Test strategy by module type

| Type | Approach | Mocking |
|------|----------|---------|
| **Utilities** (pure logic) | Unit tests, no mocks needed | None — test real functions |
| **Tools** (MCP endpoints) | Mock DB + embedding generator, test orchestration logic | DB, embedding model, file I/O |
| **Schemas** | Validation tests with valid/invalid inputs | None |
| **Core** (server, config, database) | Integration tests | Temp DB |

### Batch 1 — Core utilities (foundation everything else depends on)

- `cache.py` — TTL expiration, mtime validation, thread safety, stats
- `vector_store.py` — cosine similarity, find_top_k, edge cases
- `fts_search.py` — FTS5 query building, result parsing
- `file_utils.py` — chapter content reading, split chapters, excerpts
- `excerpt_utils.py` — relevant excerpt extraction
- `embedding_loader.py` — cache hit/miss, DB fallback

### Batch 2 — Tool modules (high user-facing risk)

- `chapter_tools.py` (718 lines) — get_chapter, get_section, list_sections
- `book_tools.py` (180 lines) — list_books, get_book_info
- `search_tools.py` (97 lines) — text_search, search_all_books
- `semantic_search_tool.py` (183 lines) — fill gaps in existing coverage
- `reading_tools.py` (531 lines) — mark_as_read, bookmarks, progress

### Batch 3 — Discovery and learning tools

- `discovery_tools.py` (467 lines) — topic_coverage, find_related, duplicates
- `learning_tools.py` (642 lines) — teach_concept, study guides
- `analytics_tools.py` (500 lines) — library stats, author insights
- `export_tools.py` (408 lines) — markdown export, code extraction

### Batch 4 — Planning, schemas, core

- `project_learning_tools.py` (897 lines) — learning path generation
- `project_planning_tools.py` (3,793 lines) — BRD, architecture, implementation plans
- `tool_schemas.py` + `response_schemas.py` — validation edge cases
- `config.py`, `database.py`, `server.py` — integration smoke tests

### File naming

`tests/test_src_<module>.py` (e.g., `tests/test_src_cache.py`) to distinguish from existing `agentic_pipeline` tests.

---

## 3. Planning Tools Hybrid LLM Enhancement

### Goal

Use LLM for high-value narrative sections, keep templates for structure. No change to the template system itself — add an LLM layer on top.

### Sections enhanced by LLM vs kept as templates

| Section | Current (template) | Enhanced (LLM) |
|---------|-------------------|----------------|
| Phase objectives | Static list | Keep template |
| Deliverables/checklists | Static list | Keep template |
| Timelines | Computed dates | Keep template |
| **Executive summary** | Generic boilerplate | LLM generates from goal + book context |
| **Risk analysis** | Static risk catalog | LLM tailors risks to specific goal |
| **Decision rationale** | Options listed, no guidance | LLM recommends based on book knowledge |
| **Chapter relevance notes** | Just title + similarity score | LLM explains why each chapter matters |

### Architecture

```
User goal
    |
_detect_project_type() -> template (unchanged)
    |
_search_for_best_practices() -> book chapters (unchanged)
    |
_enhance_sections(goal, template, search_results, sections_to_enhance)
    |                         |
    LLM call               Fallback: return template text as-is
    |
_build_*_markdown() -> merge enhanced + template sections
```

### New file: `src/utils/llm_enhancer.py`

- `enhance_planning_section(section_type, goal, context, provider="openai")` — single LLM call per section
- Uses existing `OPENAI_API_KEY` from env (already available for embeddings)
- Prompt includes relevant book excerpts as grounding context
- Graceful fallback: if LLM fails or times out, return the original template text

### Cost/latency guardrails

- Only enhance when `enhance=True` parameter is passed (default: `False` — backward compatible)
- Use `gpt-4o-mini` for cost efficiency (narrative generation, not reasoning)
- Cap at 4 LLM calls per document (executive summary, risk analysis, decision rationale, relevance notes)
- Cache enhanced sections by `(goal_hash, project_type, section_type)` to avoid re-generating

### Modified files

- `project_planning_tools.py` — add `enhance` parameter to `generate_implementation_plan`, `generate_brd`, `generate_wireframe_brief`
- `server.py` — pass through `enhance` parameter in MCP tool wrappers

### Tests

- Mock LLM responses, verify enhanced text replaces template sections
- Verify fallback returns template text on LLM error
- Verify `enhance=False` produces identical output to current behavior

---

## Sequencing

```
Session 1:  DB context manager refactor (agentic_pipeline/)
Session 2:  Test Batch 1 - core utilities
Session 3:  Test Batch 2 - tool modules
Session 4:  Test Batch 3 - discovery and learning tools
Session 5:  Test Batch 4 - planning, schemas, core
Session 6:  Planning tools LLM enhancement
```

### Why this order

- DB refactor first: small, self-contained, cleans up before we write new tests
- Test batches 1-4 next: batch 1 tests the utilities that all tools depend on, so writing those tests may surface bugs we fix before testing the tools themselves
- Planning tools LLM last: by session 6 we'll have full test coverage on `project_planning_tools.py` (batch 4), so we can enhance it with confidence

### Each session produces

- A branch, merged to main when tests pass
- Updated CLAUDE.md if architecture changed
- Session saved via claude-innit

### Exit criteria

- All 36 src/ modules have dedicated tests
- agentic_pipeline/ uses context managers consistently
- Planning tools support `enhance=True` with graceful LLM fallback
- All tests pass (current 245 + ~150-200 new)

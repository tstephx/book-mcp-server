---
title: "Dead code sweep: excerpt_utils.py, batch_export_chapters, three context managers, old validate_book_id, get_processing_status; misleading approve_book_tool(adjustments=)"
number: 17
state: closed
labels: [bug]
repo: book-mcp-server
url: https://github.com/tstephx/book-mcp-server/issues/17
created: 2026-10-30
updated: 2026-11-02
---

## Problem

Dead code with no caller and no test, plus one misleading parameter. All verified with `rg -w <symbol>` across `src/`, `agentic_pipeline/`, `scripts/`, `tests/`, `*.md` and `pyproject.toml` against `main` at 8ade59a; only the definition (and, where noted, archived docs) came back.

Delete:

- `src/utils/excerpt_utils.py` — the whole module is never imported (`extract_relevant_excerpt`, `split_paragraphs`). Replaced by query-aware excerpts per an archived design doc.
- `src/utils/file_utils.py:140` `get_chapter_excerpt()` — same replacement, no caller.
- `src/utils/batch_ops.py:150` `batch_export_chapters()` — ~90 lines, not registered as an MCP tool (`server.py` imports only `batch_semantic_search` and `get_library_statistics`).
- `src/utils/context_managers.py:46` `database_transaction()`, `:87` `batch_processing_context()`, `:123` `error_context()` — only their own docstring examples and archived QUICK-WINS/REFACTORING docs reference them. `embedding_model_context` in the same file is live; keep it.
- `src/schemas/tool_schemas.py:171` module-level `validate_book_id()` — `ref/mcp-tools.md:168` tells readers "Do not call the old validate_book_id() directly"; nothing does. (`src/utils/validators.py:16` has a same-named function that is also uncalled; only `resolve_book_id` is imported from that module.)
- `agentic_pipeline/adapters/processing_adapter.py:372` `ProcessingAdapter.get_processing_status()` — no caller, no test.

Fix the docstring or the behaviour:

- `agentic_pipeline/mcp_server.py:22` `approve_book_tool(adjustments=...)` is documented (line 34) as "Optional adjustments to apply before ingestion", but `approve_book` (`approval/actions.py:373-406`) only copies the dict into the audit row and applies nothing. The FastMCP `approve` wrapper (`agentic_mcp_server.py:61`) does not expose the parameter, so it is also unreachable from MCP. Either apply the adjustments, or reword the docstring to "recorded in the audit trail" and drop the parameter from the tool signature.

Also worth a look in the same pass (MINOR, not blocking): four input schemas re-exported from `src/schemas/__init__.py` with no caller (`SearchInput`, `BookIdInput`, `ChapterInput`, `ChapterRangeInput`); `orchestrator.py:294` `_run_embedding` is tests-only while `ref/pipeline-architecture.md:68` still names it as the EMBEDDING step; `spot_checks.pipeline_id` (`ref/db-schema.md:197`) is always NULL because neither CLI call passes it (`autonomy/spot_check.py:75`).

## Evidence

Found by the tech-debt scan, pass 2 (vault `Research/Project-Audits/2026-10-28-tech-debt-scan-wave-1-python.md`), each item re-checked by hand.

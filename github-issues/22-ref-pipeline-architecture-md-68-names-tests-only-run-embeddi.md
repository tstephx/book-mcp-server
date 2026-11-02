---
title: "ref/pipeline-architecture.md:68 names tests-only _run_embedding as the EMBEDDING step"
number: 22
state: open
labels: [documentation]
repo: book-mcp-server
url: https://github.com/tstephx/book-mcp-server/issues/22
created: 2026-11-02
updated: 2026-11-02
---

Follow-up from #17 (PR #20).

`orchestrator._run_embedding` (`agentic_pipeline/orchestrator/orchestrator.py`, ~:294) is only called from tests, but `ref/pipeline-architecture.md:68` still describes it as the EMBEDDING step. In production, embedding runs inline in `approve_book()` (APPROVED → EMBEDDING → COMPLETE).

**Fix:** either point the doc at the inline `approve_book` path, or delete `_run_embedding` and its tests.

**DOD:** doc and code agree on what runs the EMBEDDING step.

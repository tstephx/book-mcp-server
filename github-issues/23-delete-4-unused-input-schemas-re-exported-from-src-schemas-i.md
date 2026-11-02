---
title: "Delete 4 unused input schemas re-exported from src/schemas/__init__.py"
number: 23
state: open
labels: [enhancement]
repo: book-mcp-server
url: https://github.com/tstephx/book-mcp-server/issues/23
created: 2026-11-02
updated: 2026-11-02
---

Follow-up from #17 (PR #20).

`SearchInput`, `BookIdInput`, `ChapterInput` and `ChapterRangeInput` are re-exported from `src/schemas/__init__.py` but have no callers. Re-grep with `rg -w <symbol>` before deleting.

**DOD:** no callers remain, symbols and exports removed, `make lint && make test` green.

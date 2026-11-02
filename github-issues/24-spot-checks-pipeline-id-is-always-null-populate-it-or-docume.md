---
title: "spot_checks.pipeline_id is always NULL: populate it or document it"
number: 24
state: open
labels: [question]
repo: book-mcp-server
url: https://github.com/tstephx/book-mcp-server/issues/24
created: 2026-11-02
updated: 2026-11-02
---

Follow-up from #17 (PR #20).

`spot_checks.pipeline_id` (`ref/db-schema.md:197`) is never written: neither CLI call into `autonomy/spot_check.py:75` passes it.

**Decision needed:** pass `pipeline_id` at the call sites, or document the column as always NULL in `ref/db-schema.md`. Do not change the `library.db` schema (shared with other repos).

**DOD:** column is either populated (with a test) or documented as always NULL.

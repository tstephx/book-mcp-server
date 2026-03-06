---
name: sync-ref-docs
description: Audit ref/ documentation against actual codebase and fix any drift. Run after feature work to keep docs accurate.
disable-model-invocation: true
---

# Sync Reference Documentation

Audits all 4 ref docs against the codebase and reports drift.

## Steps

### 1. Audit `ref/mcp-tools.md`

**Library tools**: Grep for `@mcp.tool()` in `src/tools/` and `src/server.py`. Extract function names.
**Pipeline tools**: Grep for `@mcp.tool()` in `agentic_pipeline/mcp_server.py`. Extract function names.

Compare against tools listed in `ref/mcp-tools.md`.
- Flag tools in code but not in docs (MISSING from docs)
- Flag tools in docs but not in code (STALE in docs)

### 2. Audit `ref/db-schema.md`

Read `agentic_pipeline/db/migrations.py`:
- Extract all table names from `CREATE TABLE` statements in `MIGRATIONS`
- Extract all columns per table
- Extract versioned ALTER TABLE additions in `run_migrations()`

Compare against `ref/db-schema.md`.
- Flag tables/columns in migrations but not in docs
- Flag tables/columns in docs but not in migrations

### 3. Audit `ref/module-map.md`

List all `.py` files in `agentic_pipeline/` (recursively, excluding `__pycache__`).

Compare against modules listed in `ref/module-map.md`.
- Flag files that exist but aren't documented
- Flag documented files that no longer exist

### 4. Audit `ref/pipeline-architecture.md`

Grep for `PipelineState` enum or state constants in `agentic_pipeline/`.

Compare against states listed in `ref/pipeline-architecture.md`.
- Flag states in code but not in docs
- Flag states in docs but not in code

### 5. Report

Present a checklist:

```
## ref/mcp-tools.md
- [ ] MISSING: tool_name (in src/tools/foo.py, not in docs)
- [ ] STALE: old_tool_name (in docs, not in code)

## ref/db-schema.md
- [x] All tables match

## ref/module-map.md
- [ ] MISSING: agentic_pipeline/new_module.py

## ref/pipeline-architecture.md
- [x] All states match
```

### 6. Fix

Ask: "Which items should I fix?" Then update only the selected ref docs. Do not make changes without user confirmation.

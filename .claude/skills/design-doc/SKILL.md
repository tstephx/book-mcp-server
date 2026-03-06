---
name: design-doc
description: Create or archive a design document following the docs/active/ to docs/archive/ lifecycle.
disable-model-invocation: true
---

# Design Doc Lifecycle

Arguments: `<action: new|archive>` `<topic>`

## Action: new

### 1. Create the document

Create `docs/active/YYYY-MM-DD-<kebab-case-topic>-design.md`:

```markdown
# <Topic> Design

**Status:** Draft
**Date:** YYYY-MM-DD
**Author:** Taylor Stephens

## Problem

[What problem does this solve? Why now?]

## Proposed Solution

[High-level approach]

## Design Details

[Technical details, data flow, module changes]

## Alternatives Considered

[What else was evaluated and why it was rejected]

## Implementation Plan

- [ ] Step 1
- [ ] Step 2
- [ ] ...

## Open Questions

- [ ] Question 1
```

### 2. Update the active docs index

Edit `docs/active/README.md` — replace the `*(none currently)*` row or add a new row:

```markdown
| [YYYY-MM-DD-<topic>-design.md](YYYY-MM-DD-<topic>-design.md) | <Short description> |
```

### 3. Prompt for content

Ask the user to fill in the Problem and Proposed Solution sections. Do not write placeholder content.

## Action: archive

### 1. Find the document

Look in `docs/active/` for a file matching the topic.

### 2. Move to archive

Move the file from `docs/active/` to `docs/archive/`.

### 3. Update indexes

- Remove the row from `docs/active/README.md`. If no docs remain, restore the `*(none currently)*` placeholder.
- No index file exists for `docs/archive/` — no update needed there.

### 4. Prompt for ADR

Ask the user: "Should an ADR be created from this design doc?" If yes, invoke the `/new-adr` skill.

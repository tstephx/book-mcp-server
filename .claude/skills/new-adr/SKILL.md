---
name: new-adr
description: Create a numbered Architecture Decision Record in docs/decisions/ with consistent format and index update.
disable-model-invocation: true
---

# New ADR

Arguments: `<title>`

## Steps

### 1. Determine next number

List `docs/decisions/` to find the highest existing number. Increment by 1. Pad to 3 digits (e.g., 008).

### 2. Create the ADR file

Create `docs/decisions/<NNN>-<kebab-case-title>.md`:

```markdown
# ADR <NNN>: <Title>

**Status:** Accepted
**Date:** <YYYY-MM-DD>

## Context

[What is the problem or situation that led to this decision?]

## Decision

[What did we decide, and why?]

## Consequences

[What are the trade-offs, limitations, and follow-up implications?]
```

Ask the user to provide the Context, Decision, and Consequences content. Do not fill these in with placeholder text.

### 3. Update the index

Append a row to the table in `docs/decisions/README.md`:

```markdown
| [<NNN>](<NNN>-<kebab-case-title>.md) | <Short description> | Accepted |
```

### 4. Remind

If the decision affects daily workflow (new state, new config knob, new tool pattern), remind the user to update `CLAUDE.md`.

---
name: superpowers-development-workflow
description: Use when building features, fixing bugs, or writing code. Enforces TDD, systematic debugging, and structured planning workflows.
---

# Superpowers Development Workflow

A complete software development workflow with enforced best practices.

## When to Use

- **Building features** → Start with brainstorming, then plan, then TDD
- **Fixing bugs** → Use systematic debugging before proposing fixes
- **Writing any code** → Follow TDD (test first, watch fail, minimal code)
- **Completing work** → Verify before claiming success

## Core Workflows

### New Feature Workflow

1. **Brainstorm** - Ask questions one at a time, explore 2-3 approaches, present design in sections
2. **Write Plan** - Break into bite-sized tasks (2-5 min each) with exact file paths and code
3. **Execute with TDD** - For each task: write failing test → verify fail → minimal code → verify pass → commit

### Bug Fix Workflow

1. **Systematic Debug** - Find root cause BEFORE attempting fixes
2. **Write failing test** - Reproduce the bug in a test
3. **Minimal fix** - Only fix the root cause
4. **Verify** - Run tests, confirm fix

## The Iron Laws

### TDD Iron Law
```
NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST
```
Write code before test? Delete it. Start over.

### Debugging Iron Law
```
NO FIXES WITHOUT ROOT CAUSE INVESTIGATION FIRST
```
If 3+ fixes failed → Question architecture, don't attempt fix #4.

### Verification Iron Law
```
NO COMPLETION CLAIMS WITHOUT FRESH VERIFICATION EVIDENCE
```
Run the command. Read output. THEN claim success.

## TDD: Red-Green-Refactor

| Phase | Action | Verify |
|-------|--------|--------|
| RED | Write ONE failing test | Run it, confirm fails for expected reason |
| GREEN | Write MINIMAL code to pass | Run it, confirm passes |
| REFACTOR | Clean up while staying green | Run tests, still passing |

## Systematic Debugging Phases

| Phase | Activities | Success Criteria |
|-------|------------|------------------|
| 1. Root Cause | Read errors carefully, reproduce, check recent changes | Understand WHAT and WHY |
| 2. Pattern | Find working examples, compare differences | Identify what's different |
| 3. Hypothesis | Form theory, test ONE change | Confirmed or new hypothesis |
| 4. Implementation | Create failing test, fix, verify | Bug resolved, tests pass |

## Brainstorming Process

1. Understand project context (files, docs, commits)
2. Ask questions ONE AT A TIME
3. Prefer multiple choice questions
4. Propose 2-3 approaches with trade-offs
5. Present design in 200-300 word sections
6. Validate each section before continuing
7. Apply YAGNI ruthlessly

## Planning: Bite-Sized Tasks

Each task should be ONE action (2-5 minutes):
- "Write the failing test" - one step
- "Run it to verify it fails" - one step
- "Write minimal implementation" - one step
- "Run tests to verify pass" - one step
- "Commit" - one step

Include in each task:
- Exact file paths
- Complete code (not "add validation")
- Exact commands with expected output

## Red Flags - STOP

**TDD violations:**
- Writing code before test
- "I'll test after"
- "Too simple to test"
- Test passes immediately (means testing wrong thing)

**Debugging violations:**
- "Quick fix for now"
- "Just try changing X"
- Proposing fixes before understanding root cause
- Multiple changes at once

**Verification violations:**
- "Should work now"
- "I'm confident"
- Claiming success without running tests

## Common Rationalizations (Don't Fall For These)

| Excuse | Reality |
|--------|---------|
| "Too simple to test" | Simple code breaks. Test takes 30 seconds. |
| "I'll test after" | Tests passing immediately prove nothing. |
| "Already manually tested" | Ad-hoc ≠ systematic. Can't re-run. |
| "Quick fix for now" | Quick fixes mask root cause. |
| "I see the problem" | Seeing symptoms ≠ understanding root cause. |

## Verification Checklist

Before claiming work is complete:
- [ ] Run test/build command
- [ ] Read FULL output
- [ ] Check exit code
- [ ] Confirm zero failures
- [ ] THEN make the claim

## Key Principles

1. **Test first, always** - No exceptions without explicit permission
2. **Root cause first** - No fixes without understanding WHY
3. **Evidence before claims** - Run verification, then claim success
4. **One thing at a time** - One question, one test, one fix
5. **Delete and restart** - When TDD is skipped, delete the code
6. **YAGNI** - You Aren't Gonna Need It. Remove unnecessary features.

---

## Project-Specific Testing

For this MCP server project:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_semantic_search.py -v

# Quick server test
python test_server.py
```

## Project-Specific Verification

Before claiming a feature is complete:
```bash
# 1. Run tests
python -m pytest tests/ -v

# 2. Test the server starts
timeout 3 venv/bin/python -m src.server || true

# 3. Verify in Claude Desktop (restart required)
```

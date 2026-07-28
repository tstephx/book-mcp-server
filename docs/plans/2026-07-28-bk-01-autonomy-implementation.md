# BK-01 Unified Autonomy Decisions Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to implement this plan task-by-task.

**Goal:** Make autonomy mode, escape hatch, thresholds, eligibility rules, and the daily cap govern every automatic approval while unifying approval audit and metrics recording.

**Architecture:** `AutonomyConfig` owns a fail-closed structured decision policy. The orchestrator evaluates that policy after validation and delegates allowed decisions to `approve_book()`, which becomes the common mutation path for orchestrator, CLI, MCP, and batch approvals.

**Tech Stack:** Python 3.12, SQLite, pytest, dataclasses, existing pipeline state machine.

### Task 1: Add the decision-policy contract

**Files:**
- Modify: `agentic_pipeline/autonomy/config.py`
- Modify: `agentic_pipeline/autonomy/__init__.py`
- Test: `tests/test_autonomy_config.py`

1. Write failing tests for supervised and escape-hatch denial.
2. Add failing tests for partial mode’s global threshold.
3. Add failing tests for confident mode’s calibrated per-type threshold.
4. Add failing tests for unknown types, invalid confidence, validation failure, `needs_review`, and daily-cap exhaustion.
5. Run the tests and confirm each fails for the missing structured policy.
6. Add `AutoApprovalDecision` and the minimal `evaluate_auto_approval()` implementation.
7. Make `should_auto_approve()` delegate to the structured policy.
8. Run the policy tests and confirm they pass.

### Task 2: Route orchestration through the policy

**Files:**
- Modify: `agentic_pipeline/orchestrator/orchestrator.py`
- Modify: `tests/test_orchestrator.py`
- Modify: `tests/test_orchestrator_integration.py`

1. Replace the stale default-auto-approval expectation with a failing supervised-mode test.
2. Add failing partial-mode and escape-hatch integration tests.
3. Assert held books return a machine-readable approval reason.
4. Run the tests and confirm the old static-threshold path fails them.
5. Evaluate the autonomy policy after validation.
6. For allowed decisions, call `approve_book(..., background=False)` with policy metadata.
7. For denied decisions, return `PENDING_APPROVAL` without embedding.
8. Run the orchestrator tests and confirm they pass.

### Task 3: Unify audit and metrics recording

**Files:**
- Modify: `agentic_pipeline/approval/actions.py`
- Modify: `agentic_pipeline/autonomy/metrics.py`
- Modify: `tests/test_approval_actions.py`
- Modify: `tests/test_autonomy_metrics.py`

1. Add failing tests that human and automatic approvals both write one audit record and one metrics record.
2. Add a failing test that the processing confidence and policy metadata are preserved for automatic decisions.
3. Run the tests and confirm the missing metrics path fails.
4. Extend `approve_book()` with optional confidence and decision metadata.
5. Record the decision through `MetricsCollector` after the state transition and immutable audit write.
6. Run approval and metrics tests and confirm they pass.

### Task 4: Delegate batch approval to the common action

**Files:**
- Modify: `agentic_pipeline/batch/operations.py`
- Modify: `tests/test_batch_operations.py`

1. Add a failing test that each approved batch item receives individual audit and metrics records.
2. Preserve the existing dry-run, summary audit, concurrency skip, and embedding-result behavior.
3. Replace the duplicated transition/mark/embed sequence with `approve_book(..., background=False)`.
4. Run batch tests and confirm they pass.

### Task 5: Align documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `ref/pipeline-architecture.md`

1. Document supervised-by-default behavior.
2. Document partial and confident threshold sources.
3. Clarify that `CONFIDENCE_THRESHOLD` controls LLM fallback, not approval.
4. Document the daily-cap and fail-closed eligibility rules.

### Task 6: Verify

1. Run the focused autonomy, approval, orchestrator, batch, CLI, and MCP tests.
2. Run the broader unit suite.
3. Inspect `git diff --check`, `git status`, and the complete branch diff.
4. Use `superpowers:requesting-code-review`.
5. Address review findings with TDD.
6. Use `superpowers:verification-before-completion` before reporting completion.

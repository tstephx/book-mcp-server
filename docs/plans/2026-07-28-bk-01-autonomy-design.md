# BK-01 Unified Autonomy Decisions — Design

## Goal

Make the stored autonomy mode, escape hatch, thresholds, eligibility rules, and daily cap authoritative for every automatic book approval. Human approvals should continue to work through the CLI and MCP server, while all approvals—automatic, individual, or batch—use the same audit and autonomy-metrics recording path.

## Decision

Use `AutonomyConfig` as the single auto-approval policy owner. Extend it with a structured `evaluate_auto_approval()` result that explains whether the book is eligible, which mode and threshold were used, and why a book was held for review. Keep `should_auto_approve()` as a compatibility wrapper.

The policy is deliberately fail-closed:

- `supervised` mode never auto-approves.
- An active escape hatch behaves as supervised.
- `partial` mode uses the global database-backed `auto_approve_threshold`.
- `confident` mode requires a per-book-type calibrated threshold.
- Unknown book types, non-finite/out-of-range confidence, failed validation, `needs_review`, missing thresholds, and exhausted daily capacity all produce human review.
- The daily cap is calculated from immutable approval-audit records.

`CONFIDENCE_THRESHOLD` remains available to the ingestion adapter as its LLM-fallback trigger, but no longer authorizes approval.

## Approval data flow

After extraction validation, the orchestrator always transitions to `PENDING_APPROVAL` and asks `AutonomyConfig.evaluate_auto_approval()` for a decision. A denied decision returns the pending record with an `approval_reason`. An allowed decision calls the same `approve_book()` action used by humans, with `background=False`, the processing confidence, and the policy decision metadata.

`approve_book()` remains responsible for the state transition, approval metadata, immutable audit entry, metrics record, and embedding dispatch. Batch approval delegates to it per item rather than duplicating those writes. This makes the CLI, MCP server, batch tools, and orchestrator converge on one mutation path.

## Failure behavior

Policy lookup failures must not auto-approve. Approval audit or metrics failures occur before embedding and must be visible rather than silently claiming a fully governed decision. Existing compare-and-swap behavior continues to protect concurrent state changes.

Human rejection and rollback behavior are outside BK-01 except where existing tests require compatibility.

## Verification

Tests cover the policy matrix, orchestrator behavior, escape-hatch behavior, audit and metrics persistence, batch delegation, and existing approval contracts. The autonomy-specific suite must pass. The broad unit suite will also be run; existing offline `tiktoken` download failures will be reported separately if the sandbox remains network-restricted.

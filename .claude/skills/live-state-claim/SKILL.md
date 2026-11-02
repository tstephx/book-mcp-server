---
name: live-state-claim
description: Claim or release the session-claim ledger around a runbook that writes shared live state (library.db, the launchd worker). Use as /live-state-claim claim <slug>, /live-state-claim release <slug> "<outcome>", or /live-state-claim status.
disable-model-invocation: true
---

# Live-State Claim

Wraps `~/.claude/hooks/session_claims.py` for runbooks that mutate shared live state:
`library.db` (rebuilds, `doctor --fix`, `rechunk`, approvals, reingest) or the
`com.taylorstephens.agentic-pipeline-worker` launchd job.

Why: on 2026-11-02 the live `fts-rebuild` runbook was executed with no ledger entry, and a
second session had to rediscover it was already done from file mtimes and `launchctl`
(book-mcp-server#28). A claim stops a peer starting the same write; a release with an
outcome note leaves the trace.

Requires the user-global session-claims hook at `~/.claude/hooks/session_claims.py` (it lives
outside this repo). If that file is missing, stop and say so; do not improvise a substitute.

## Arguments

`$ARGUMENTS` is one of:

- `claim <slug>` — before the first write
- `release <slug> "<outcome>"` — when done, stood down, or aborted
- `status` — list active claims for this repo

`<slug>` is a short lowercase kebab-case name for the runbook (letters, digits, hyphens only),
e.g. `fts-rebuild`; reject anything else. The ledger branch label is `live-state-<slug>`, so
the same slug must be passed to `claim` and `release`.
Repo is always `tstephx/book-mcp-server`.

## claim

1. Check first:

   ```bash
   python3 ~/.claude/hooks/session_claims.py check --repo tstephx/book-mcp-server --branch live-state-<slug>
   ```

   - exit 0: clear, continue.
   - exit 1: a **live** session holds it. Stop, report who, and do not write anything.
   - exit 2: stale claim from a dead session. Note it, continue.

2. Also look for a different live-state claim on the same DB, since a rebuild and a
   `doctor --fix` contend for the same write lock even under different slugs:

   ```bash
   python3 ~/.claude/hooks/session_claims.py list --repo tstephx/book-mcp-server --status active
   ```

   Runbook claims are the ones this skill makes: an active `tstephx/book-mcp-server` claim with
   **no `issue`** and a label starting `live-state-`. Ignore claims that carry an issue number
   (those are feature branches, e.g. `live-state-claim-28`) and claims from your own
   `session_id`. If any other active runbook claim exists, stop and report it.

3. Claim:

   ```bash
   python3 ~/.claude/hooks/session_claims.py claim --repo tstephx/book-mcp-server --branch live-state-<slug>
   ```

4. Run the already-done check before writing: for a runbook with a kickoff prompt, look in
   `git log --oneline -20`, the backup dir for a recent `*.backup-*`, and
   `launchctl print gui/$(id -u)/com.taylorstephens.agentic-pipeline-worker` start time. If
   the work is already done, release with outcome `already done: <evidence>` and stop.

## release

Always release, including on failure or abort. The outcome is one line a stranger could act
on: what ran, where the backup is, the counts, the worker pid.

```bash
python3 ~/.claude/hooks/session_claims.py release --repo tstephx/book-mcp-server \
  --branch live-state-<slug> --outcome "<outcome>"
```

Example outcome: `fts-rebuild done: backup library.db.backup-fts-20261102-124502, entries_before=8083 entries_after=8294 errors=0, worker pid 51854`.

If release exits 1 ("no matching active claim"), the claim was never made, was already
released, or belongs to a different session. Release matches on `session_id`, so only the
session that claimed can release; a successor session cannot release a predecessor's claim. A
crashed session's claim reads as stale (check exit 2, non-blocking), so the successor can
proceed and claim afresh. Say which case applies; do not claim retroactively.

## status

```bash
python3 ~/.claude/hooks/session_claims.py list --repo tstephx/book-mcp-server --status active
```

Report label, status, claimed_at, and outcome for each row.

## Adding this to a kickoff prompt

Any kickoff prompt that writes live state should name `/live-state-claim claim <slug>` before
its first mutating step and `/live-state-claim release <slug> "<outcome>"` as its last, and
list the backup path and rebuild counts as required outcome content.

## Do not

- Skip `release` because the run failed. A failed run is exactly when the trace matters.
- Use this for read-only work or for feature branches; the preflight skill covers those.
- Hand-edit `~/.claude/logs/session-claims.json`.

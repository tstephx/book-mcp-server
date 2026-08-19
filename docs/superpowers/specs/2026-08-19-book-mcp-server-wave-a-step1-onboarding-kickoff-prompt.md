# Onboard book-mcp-server as Wave A's MCP-servers category representative (Task 5 Step 1)

## Where things stand

`open-wave-a` was decided 2026-08-19 (`docs/plans/2026-07-28-portfolio-onboarding-migration-sequence.md`,
commit `e817234`). Task 5 ("Migrate Wave A — Similar, Active, Lower Risk")
Step 1 requires: "Inspect one category representative — Prove the
proposed profile, verify-wrapper pattern, audit state, and release
contract on one repository before continuing that category." `whatbox-portfolio-mcp`
was dropped from the MCP-servers category this same session (archived
`lifecycle: reference` 2026-08-18) — that category is now 4 repos:
`book-mcp-server`, `career-coach-mcp`, `career-kb-mcp`, `my-mcp-portfolio`.

Step 1 maps onto the plan's "## Canonical Per-Repository Transaction"
section (same file, "Use this protocol for the canary and every migration
candidate"). `book-mcp-server` was chosen as the MCP-servers category
representative: largest sub-category, fully un-onboarded, no
`taylor-dev-core` presence at all before this session.

**Already done this session, from canonical `_Workspace`** (transaction
steps 1 and 4):

- **Step 1 (read-only preflight):** registry entry confirmed
  (`registry.yaml`: `book-mcp-server`, path
  `/Users/taylorstephens/Dev/_Projects/book-mcp-server`, category
  `mcp-servers`, lifecycle `active`). Branch `main`. Pre-existing,
  unrelated dirty state to preserve: `.config/wt.toml` (modified) and
  `github-issues/` (untracked) — do not touch, do not commit. An active
  worktree exists at
  `/Users/taylorstephens/Dev/_worktrees/book-mcp-server/feat-self-generating-repo-docs`
  (branch `feat/self-generating-repo-docs`) — unrelated, leave alone. No
  `AGENTS.md`; `CLAUDE.md` exists (16KB). **A canonical verify command
  already exists**: `scripts/pre-push-verify.sh`, executable, wired as
  the actual `.git/hooks/pre-push` symlink — runs `uv sync --locked`,
  ruff lint, and the pytest unit suite (mirrors `ci.yml`'s `lint`/`test`
  jobs; integration tests needing `OPENAI_API_KEY` are deliberately
  skipped, matching CI). This means the verify-wrapper pattern likely
  does **not** need a new `scripts/ai-verify.sh` — the plan's rule is "add
  a thin wrapper only when the proposed `commands.verify` is a bare PATH
  command," and this one already names a real repository-relative
  executable. No repository-owned audit command was found (no
  Claude-config/plugin audit script) — expect `commands.audit: null`,
  recorded `not-applicable`, unless this session's own `/repo-onboard`
  investigation finds one it should use instead. Risk context: this repo
  has real production characteristics — a launchd worker
  (`scripts/com.taylorstephens.agentic-pipeline-worker.plist`) and a
  SQLite DB (a `PreToolUse` hook already blocks direct edits to `.db`
  files); it needs real API keys at runtime (not read by this session).
- **Step 4 (release enablement), both halves approved and applied:**
  `enable-taylor-dev-core.rb --apply book-mcp-server` (bootstrap,
  raw-checkout path) then `promote-harness-release.rb --apply
  book-mcp-server <current-digest>` (redirect to the immutable
  materialized release), same two-step sequence as the pilot cohort
  (`fast-mail` `06d96fe`→`25d6ca9`). Committed together as `book-mcp-server`
  `d1383c5` ("Enable and promote taylor-dev-core to the current
  materialized release"), pushed to `origin/main`
  (`fc29ba2..d1383c5`, pre-push hook green: 663 passed, 3 skipped).
  `harness-release-status.rb --consumer-settings … --session-receipt …`
  confirmed `active_marketplace_state: active`, `session_state: current`
  immediately after.

**Re-derive before trusting any of this** — time has passed since it was
written. Re-check `book-mcp-server`'s `git log -3` / `git status`, re-run
`harness-release-status.rb` for it, and re-confirm the current release
digest fresh (never reuse a digest cited above without reconfirming it,
same rule as every prior promotion in this plan).

## What this session does

This session should be **launched from `book-mcp-server`'s own root**
(`/Users/taylorstephens/Dev/_Projects/book-mcp-server`) — the
`/repo-onboard`, `/repo-verify`, `/repo-status`, `/repo-drift`,
`/repo-handoff` skills only exist in an interactive session rooted there
with `taylor-dev-core` enabled (confirmed by this workspace's own
"Dispatched-agent skill gap" precedent — a dispatched subagent does not
get them). Being freshly started after the promotion commit above also
satisfies transaction Step 7 ("start a fresh session... an old session is
`reload-required`, not current") — confirm this session's own receipt
reads `session_state: current` rather than assuming it.

Complete the remaining transaction steps (2–3, 5–10):

1. **Step 2 — resolve profile and command contracts.** Run `/repo-onboard`
   in proposal mode (not `--replace`, not apply). Present the proposed
   profile to the owner for approval or override — a wrong or unexpected
   proposal is not a plugin defect, owner override is the designed
   mechanism (per `docs/superpowers/specs/2026-07-27-taylor-dev-core-subproject-2-pilot-design.md`'s
   per-repository pilot procedure, step 2 — `fast-mail`'s own pilot
   needed an owner override to `mcp-server`). Confirm `commands.verify`
   resolves to `scripts/pre-push-verify.sh` (already a real
   repository-relative executable — only author a wrapper if
   `/repo-onboard`'s own detection disagrees and proposes a bare PATH
   command instead). Confirm `commands.audit` — `null`/`not-applicable`
   with rationale unless a real bounded audit command turns up. Never add
   a placeholder audit.
2. **Step 3 — compile and review task context.** State the migration task
   contract (this kickoff prompt is a starting point, not the final
   contract — fill in `docs/ai-engineering/task-context-packet.md`'s
   "Start contract" fields for this specific onboarding if useful) and
   run `/repo-context`. Review selected/omitted sources, authority/trust/
   freshness, contradictions/unknowns, allowed reads/writes, exact
   verification/rollback commands. Reject or recompile a stale or
   incomplete envelope rather than proceeding past one.
3. **Step 5 — propose onboarding.** Run `/repo-onboard` in proposal mode.
   Show profile, verify/audit commands, manifest/rule/ignore/wrapper
   changes, transactional rollback behavior, and files that remain
   untouched. Get separate, explicit owner approval before apply — do not
   chain proposal and apply.
4. **Step 6 — apply transactionally.** Use `/repo-onboard`'s own apply
   mode only (a deterministic helper) — never hand-edit the target files
   to bypass it. On failure it must preserve every prior file hash,
   remove staging files, keep the previous release active, emit a
   bounded failure, and the candidate is marked `blocked` — do not
   improvise past a failed apply.
5. **Step 8 — verify behavior, in this order:** `/repo-status` →
   `/repo-verify` → `/repo-drift` → `/repo-context` on one bounded real
   task. Run `scripts/pre-push-verify.sh` directly if independent
   confirmation of a skill result is needed. No automatic retry or
   remediation on a failure — report it instead.
6. **Step 9 — record bounded evidence.** Repository id, wave, profile,
   lifecycle; before/after commit ids; release, component-manifest,
   context-schema, and source-snapshot digests; settings and manifest
   states; verification/audit/context/continuity outcomes; any
   unauthorized or safety events; local commit state; decision, owner,
   blocker, next gate.
7. **Commit onboarding changes locally in `book-mcp-server`. Do not
   push.** Matches the original pilot cohort's own precedent (onboarded
   locally, pushed only as a later, separately approved step) and this
   plan's "Enablement, onboarding, release selection, commit, push, and
   publication are separate approvals" constraint — if the owner's Step
   10 decision below turns out to be `revise` or `rollback`, an unpushed
   local commit is easy to amend or drop; a pushed one is not.
8. **Step 10 — decide.** Present `retain` / `revise` / `rollback` /
   `blocked` as the resulting decision point for the owner — do not
   choose unilaterally, same discipline this session used for the
   `website-portfolio` rehearsal and the `open-wave-a` decision.
9. **Record the outcome in `_Workspace`.** This session may `cd` into
   `/Users/taylorstephens/Dev/_Workspace` via Bash (its filesystem tools
   aren't gated by which repo it launched in, only the `/repo-*` skills
   are) to append a new dated entry to
   `docs/superpowers/specs/2026-07-28-portfolio-onboarding-migration-outcome.md`
   recording this pilot's evidence (same rigor as the 2026-08-19
   rollback-rehearsal entry already in that file — evidence, not a
   summary), matching the style of Task 4/5's other entries. Run
   `concurrent-session-preflight` before that `_Workspace` commit and
   again immediately before pushing it (peer sessions have been active in
   `_Workspace` all day). This push is separately approved from
   `book-mcp-server`'s own (unpushed) onboarding commit — pushing the
   evidence record doesn't require also pushing the still-pending
   onboarding commit.

## Constraints carried over

- Touch only `book-mcp-server` (its own onboarding-related files, per
  `/repo-onboard`'s own declared write set) and `_Workspace`'s outcome
  doc. No other registered consumer, no `_Workspace` registry or policy
  files, no manual edits to `book-mcp-server`'s settings.json outside the
  deterministic helpers already used.
- Preserve `.config/wt.toml`'s existing dirty state, `github-issues/`
  untracked content, and the `feat-self-generating-repo-docs` worktree —
  don't run a broad `git add`.
- Every apply-mode step (`/repo-onboard` apply included) needs its
  proposal reviewed and explicitly approved first — never chain proposal
  and apply without a real pause.
- Do not push `book-mcp-server`'s onboarding commit in this session — it
  stays local pending the Step 10 decision.
- Do not declare `retain`/`revise`/`rollback` unilaterally — record the
  evidence and present the decision.

## Caution

Re-derive current state before trusting anything above as still true:
`git log --oneline -5` and `git status` in both `_Workspace` and
`book-mcp-server`, and re-run `harness-release-status.rb` for
`book-mcp-server`. Run `concurrent-session-preflight` before starting (an
ad hoc, no-issue-number edit to a shared-ish repo, same trigger class as
this session's earlier `website-portfolio`/`epub-pdf-splitter` work) and
again immediately before the `_Workspace` outcome-doc push.

Never hardcode a `tree_digest` into a command from this file — read it
fresh (`harness-release-status.rb`), not as a literal copied value,
because a new release approval between now and execution would make a
hardcoded value silently wrong.

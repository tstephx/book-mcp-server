# Claude Code Classifier Provider + Pipeline Hygiene — Design Spec

**Date:** 2026-07-16
**Status:** Approved design, pending spec review
**Scope:** Backlog batch — Project D (classifier provider), PDF completeness check, rejection-reason surfacing. Item 1 of the batch (book recovery) is an ops runbook (§5), not code.

## 0. Decisions (binding)

1. **ClaudeCodeProvider is primary; OpenAI is fallback.** `CLASSIFIER_PROVIDER=claude-code|openai` forces that single provider with no fallback.
2. **The two source-gone books keep their live chunks**, flagged in `<db-dir>/doctor/awaiting-redownload.md`; dropping their epubs in the watch dir later heals them via normal dedup/reingest.
3. **Rejection reasons are already persisted** (`approval_audit.reason` — verified live); the remaining work is display-only in `status`.
4. **The 4 recoverable books are reingested AFTER the provider code merges** — those reingests run through the real launchd worker and double as the mandated worker-context smoke test for the new provider.

## 1. Section A — ClaudeCodeProvider (Project D)

**New file:** `agentic_pipeline/agents/providers/claude_code_provider.py`

```
class ClaudeCodeProvider(LLMProvider):
    name -> "claude-code"
    classify(text, metadata=None) -> BookProfile
```

- Builds the same prompt as OpenAIProvider: `load_prompt("classify")` +
  `.format(text=text[:40000])` (same truncation).
- Invokes `subprocess.run(["claude", "-p", prompt], capture_output=True,
  text=True, timeout=120)` — list args, no shell. ~14s/book measured in
  the spike; 120s is generous headroom.
- Parses the response with a fence-tolerant JSON parser (```json blocks
  or bare `{...}` — same pattern as `retrieval_eval._parse_fenced_json`),
  then feeds the dict through the same BookProfile construction used by
  `OpenAIProvider._parse_response` (hoist shared parsing into
  `providers/base.py` as `parse_profile_json(content) -> BookProfile` so
  both providers share one implementation instead of duplicating it).
- Any failure — nonzero exit, timeout, missing binary (OSError),
  unparseable output — raises `RuntimeError` with a diagnostic message.
  `ClassifierAgent.classify` already catches broad `Exception` from the
  primary and falls back; no agent-side error handling changes.

**`ClassifierAgent` changes (`agentic_pipeline/agents/classifier.py`):**

- Default order becomes: primary `ClaudeCodeProvider()`, fallback
  `OpenAIProvider()`. (`AnthropicProvider` remains in the codebase but
  leaves the default chain — it needs an API key, which defeats the
  purpose.)
- `CLASSIFIER_PROVIDER` env var, read in `_get_primary`/`_get_fallback`:
  `claude-code` or `openai` forces that provider as primary AND disables
  the fallback (single-provider mode, for debugging/cost control).
  Unset → the new default chain. Invalid value → ValueError at first
  classify (fail loud, not silently wrong).
- Explicit constructor-injected `primary=`/`fallback=` (used by tests)
  keep precedence over the env var, preserving current test behavior.

**Non-goals:** no embedding-provider change (OpenAI embeddings stay —
they must match the 3072-dim index); no changes to cache/confidence/
retry logic in the agent; no Ollama.

## 2. Section B — PDF completeness check

**File:** `agentic_pipeline/validation/extraction_validator.py`

`count_source_words(source_path)` gains a `.pdf` branch:

- Lazy `import fitz` (PyMuPDF, already an installed dependency).
- `sum(len(page.get_text().split()) for page in doc)` over the document;
  close the doc.
- Any failure (import error, corrupt file, encrypted PDF) → log warning,
  return `None` — the existing "skip the check" semantics. The EPUB
  branch and the `None` contract are unchanged; Check 8
  (`MIN_SOURCE_COVERAGE = 0.5`) applies to PDFs with no further changes.
- Docstring updated: PDF is no longer listed as unsupported.

**Known limitation (documented, accepted):** image-only/scanned PDFs
yield near-zero extractable text, making a coverage ratio meaningless
(a tiny denominator lets any extraction look complete). Guard: when the
PDF yields fewer than 100 words total, return `None` — "can't establish
a count", identical to today's skip semantics.

## 3. Section C — surface rejection reasons in `status`

**File:** `agentic_pipeline/cli.py` (`status` command, line ~353)

For pipelines in `rejected` (or after a rollback), `status` additionally
prints the most recent matching audit row:

```
state: rejected
rejected by human:cli — "duplicate of existing live copy (doctor recovery batch)"
```

Implementation: one query on `approval_audit` (`WHERE pipeline_id = ?
AND action IN ('rejected','rolled_back') ORDER BY id DESC LIMIT 1`),
printed only when a row exists. No schema change; `reject_book` and
`_record_audit` are untouched (verified already persisting).

## 4. Testing (TDD, house rules)

All tests on tmp DBs (`monkeypatch.setenv("AGENTIC_PIPELINE_DB", …)`);
no network, no live `claude`/OpenAI calls.

- **Provider (`tests/test_claude_code_provider.py`):** mock
  `subprocess.run` — success (fenced and bare JSON) → BookProfile fields
  typed correctly; nonzero exit / timeout / FileNotFoundError /
  garbage output → RuntimeError. Prompt passed as list args (no shell).
- **Agent order (`tests/test_classifier.py` additions):** default chain
  is claude-code→openai (inspect provider names, classify with primary
  mocked to fail → fallback used); `CLASSIFIER_PROVIDER=openai` → openai
  primary and NO fallback attempted; invalid value → ValueError;
  constructor injection still wins over env.
- **Shared parser:** `parse_profile_json` unit tests (valid, fenced,
  missing fields, non-dict JSON) — both providers routed through it.
- **PDF (`tests/test_extraction_validator.py` additions):** generate a
  real small PDF with fitz in-test → exact word count; sub-100-word PDF
  → None; corrupt file → None; EPUB behavior regression-pinned.
- **CLI (`tests/test_cli*.py`):** status on a rejected pipeline with an
  audit row prints actor + reason; status on non-rejected prints no
  reason line.

## 5. Ops runbook (after merge — item 1 of the batch)

1. Restart the launchd worker (`launchctl kickstart -k ...`) so it runs
   the new provider code.
2. `agentic-pipeline reingest <pipeline-id>` for the 4 recoverable books
   (Supercharged Coding with GenAI, Building Neo4j-Powered Applications,
   Automating Workflows with GitHub Actions, Mastering Git — epubs
   verified on disk in `~/Documents/_ebooks/Github/`). These classify
   through ClaudeCodeProvider in the real worker environment — the
   smoke test of decision 4. Verify each lands in `pending_approval`
   with a non-unknown book_type and provider `claude-code` in its
   profile; approve; chunks regenerate under the fixed chunker.
3. If any classify falls back to OpenAI, inspect why (claude CLI auth in
   the launchd context is the expected suspect) before trusting
   subscription billing as the steady state.
4. Write `<db-dir>/doctor/awaiting-redownload.md` listing the 2 gone
   books (Designing Experiences; Git Mastery: Accelerated Crash Course)
   with their book ids and original filenames.
5. `agentic-pipeline doctor` — expect clean; the 4 reingested books stop
   appearing as carried/missing-file chapters.

## 6. Risks

- **Launchd worker may lack claude CLI auth** → every classify falls
  back to OpenAI silently. Mitigated by the runbook's explicit provider
  check on the 4 smoke-test books (step 3), not by hope.
- **`claude -p` latency (~14s) in the worker loop:** classification is
  one step per book in an already-minutes-long pipeline; acceptable, and
  the 120s timeout bounds the worst case.
- **Fence-parser drift:** shared `parse_profile_json` is unit-tested for
  both providers' output shapes; no duplicated parsing to drift.

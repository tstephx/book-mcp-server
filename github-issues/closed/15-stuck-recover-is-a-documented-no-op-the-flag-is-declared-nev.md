---
title: "stuck --recover is a documented no-op: the flag is declared, never read, and StuckDetector has no recovery method"
number: 15
state: closed
labels: [bug]
repo: book-mcp-server
url: https://github.com/tstephx/book-mcp-server/issues/15
created: 2026-10-30
updated: 2026-11-02
---

## Problem

`agentic-pipeline stuck --recover` is a documented flag that does nothing.

- `agentic_pipeline/cli.py:480` declares `@click.option("--recover", is_flag=True, help="Auto-recover stuck pipelines")`; the `stuck(recover)` body never reads `recover`.
- `agentic_pipeline/health/stuck_detector.py` has only `detect()`; there is no recovery method.
- Documented as working in `docs/USER-GUIDE.md:248` (`agentic-pipeline stuck --recover`), `ref/cli-commands.md:24` (`stuck [--recover]`), and the debug-pipeline skill (`SKILL.md:80`).

Someone debugging a stuck pipeline will run the documented command, see the listing, and assume recovery was attempted.

## Suggested fix

Implement recovery (reset the stuck pipelines to a retryable state, which `reject --retry` already knows how to do), or delete the flag and the three doc mentions. Either is fine; the current state is the only wrong one.

## Evidence

Verified against `main` at 8ade59a. Found by the tech-debt scan, pass 2 (vault `Research/Project-Audits/2026-10-28-tech-debt-scan-wave-1-python.md`).

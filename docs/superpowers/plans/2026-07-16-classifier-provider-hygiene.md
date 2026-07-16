# Claude Code Classifier Provider + Pipeline Hygiene — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Classification runs on `claude -p` subscription billing (OpenAI as fallback), PDFs get source-coverage validation, and `status` shows rejection reasons.

**Architecture:** A new `ClaudeCodeProvider` slots into the existing `LLMProvider` ABC; shared JSON→BookProfile parsing hoists to `providers/base.py`. `count_source_words` gains a PyMuPDF branch. `status` reads the already-persisted `approval_audit.reason`.

**Tech Stack:** Python 3.12, subprocess (`claude -p`), PyMuPDF (`fitz`, already installed), Click/Rich CLI, pytest.

**Spec:** `docs/superpowers/specs/2026-07-16-classifier-provider-hygiene-design.md`. The §5 ops runbook (worker restart, 4 reingests as smoke test, awaiting-redownload note) executes AFTER merge — it is not a plan task.

## Global Constraints

- Branch: `feat/classifier-provider` (create from `main` before Task 1).
- Tests NEVER touch the live library DB (`monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(tmp_db))`); no live `claude`/OpenAI/network calls — mock `subprocess.run` and providers.
- Provider order, verbatim from spec: default primary `ClaudeCodeProvider`, fallback `OpenAIProvider`; `CLASSIFIER_PROVIDER=claude-code|openai` forces that single provider with NO fallback; invalid value → `ValueError` at first classify; constructor-injected `primary=`/`fallback=` take precedence over the env var.
- `claude -p` invocation: list args (`["claude", "-p", prompt]`), no shell, `timeout=120`.
- Prompt: `load_prompt("classify")` + `.format(text=text[:40000])` — same as OpenAIProvider.
- All provider failures raise `RuntimeError` (agent's broad `except Exception` handles fallback).
- PDF guard: total words < 100 → return `None` (scanned/image-only PDFs can't establish a count).
- Embedding provider is UNCHANGED (OpenAI, 3072-dim index); `AnthropicProvider` stays in the codebase but leaves the default chain.
- ruff must stay clean (`ruff check .` gates CI); all imports top-of-file; PostToolUse hook auto-formats — don't fight it.
- Do NOT edit pyproject.toml / requirements.txt / .db / .venv (hook-blocked; no new deps needed).
- Commit trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `agentic_pipeline/agents/providers/base.py` | modify | + `parse_profile_json(content) -> BookProfile` (shared, fence-tolerant) |
| `agentic_pipeline/agents/providers/openai_provider.py` | modify | `_parse_response` delegates to shared parser |
| `agentic_pipeline/agents/providers/claude_code_provider.py` | create | `ClaudeCodeProvider(LLMProvider)` via `claude -p` |
| `agentic_pipeline/agents/classifier.py` | modify | default chain claude-code→openai; `CLASSIFIER_PROVIDER` override |
| `agentic_pipeline/validation/extraction_validator.py` | modify | `.pdf` branch in `count_source_words` |
| `agentic_pipeline/cli.py` | modify | `status` prints rejection actor+reason |
| `tests/test_provider_base.py` | modify | `parse_profile_json` tests |
| `tests/test_claude_code_provider.py` | create | provider tests (mocked subprocess) |
| `tests/test_classifier.py` | modify | order/env-override tests |
| `tests/test_extraction_validator.py` | modify | PDF tests |
| `tests/test_cli.py` | modify | status rejection-reason test |

---

### Task 1: Shared `parse_profile_json` in `providers/base.py`

**Files:**
- Modify: `agentic_pipeline/agents/providers/base.py`
- Modify: `agentic_pipeline/agents/providers/openai_provider.py:64-77`
- Test: `tests/test_provider_base.py`

**Interfaces:**
- Consumes: `BookProfile.from_dict(data)` (`agentic_pipeline/agents/classifier_types.py:49`).
- Produces: `parse_profile_json(content: str) -> BookProfile` — raises `ValueError` on unparseable/non-dict JSON. Task 2's provider and the (delegating) `OpenAIProvider._parse_response` both call it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_provider_base.py`:

```python
import pytest

from agentic_pipeline.agents.providers.base import parse_profile_json


class TestParseProfileJson:
    VALID = '{"book_type": "technical", "confidence": 0.9, "suggested_tags": ["python"], "reasoning": "code-heavy"}'

    def test_bare_json(self):
        profile = parse_profile_json(self.VALID)
        assert profile.book_type.value == "technical"
        assert isinstance(profile.confidence, float)
        assert profile.confidence == 0.9

    def test_fenced_json(self):
        fenced = f"Here you go:\n```json\n{self.VALID}\n```\nDone."
        profile = parse_profile_json(fenced)
        assert profile.book_type.value == "technical"

    def test_garbage_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_profile_json("I could not classify this book.")

    def test_non_dict_json_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_profile_json('["a", "list"]')

    def test_unknown_book_type_maps_to_unknown(self):
        profile = parse_profile_json('{"book_type": "not-a-real-type", "confidence": 0.5}')
        assert profile.book_type.value == "unknown"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_provider_base.py -v`
Expected: FAIL — `ImportError: cannot import name 'parse_profile_json'`

- [ ] **Step 3: Implement the shared parser**

Append to `agentic_pipeline/agents/providers/base.py` (add `import json` at top):

```python
def parse_profile_json(content: str) -> BookProfile:
    """Parse an LLM classification response into a BookProfile.

    Tolerates markdown code fences by extracting the outermost {...}
    span. Raises ValueError on unparseable or non-dict JSON so provider
    callers surface a clean failure the agent can fall back on.
    """
    if "```" in content:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            content = content[start:end]

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}")
    if not isinstance(data, dict):
        raise ValueError(f"LLM response is not a JSON object: {type(data).__name__}")
    return BookProfile.from_dict(data)
```

- [ ] **Step 4: Delegate OpenAIProvider to it**

Replace `OpenAIProvider._parse_response` (openai_provider.py:64-77) with:

```python
    def _parse_response(self, content: str) -> BookProfile:
        """Parse JSON response into BookProfile (shared implementation)."""
        return parse_profile_json(content)
```

and change the base import (line 11) to:
`from agentic_pipeline.agents.providers.base import LLMProvider, parse_profile_json`

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_provider_base.py tests/test_openai_provider.py -v`
Expected: all PASS (openai provider tests pin the delegation didn't change behavior).

- [ ] **Step 6: Commit**

```bash
git add agentic_pipeline/agents/providers/base.py agentic_pipeline/agents/providers/openai_provider.py tests/test_provider_base.py
git commit -m "refactor: hoist BookProfile JSON parsing to shared parse_profile_json

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `ClaudeCodeProvider`

**Files:**
- Create: `agentic_pipeline/agents/providers/claude_code_provider.py`
- Test: `tests/test_claude_code_provider.py` (create)

**Interfaces:**
- Consumes: `LLMProvider`, `parse_profile_json` (Task 1), `load_prompt("classify")` from `agentic_pipeline/agents/prompts`.
- Produces: `ClaudeCodeProvider(timeout: int = 120)` with `.name == "claude-code"` and `.classify(text, metadata=None) -> BookProfile`; raises `RuntimeError` on every failure mode. Task 3 instantiates it as the default primary.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_claude_code_provider.py`:

```python
"""ClaudeCodeProvider — classification via `claude -p` (all subprocess mocked)."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from agentic_pipeline.agents.providers.claude_code_provider import ClaudeCodeProvider

VALID_JSON = '{"book_type": "technical", "confidence": 0.85, "suggested_tags": ["git"], "reasoning": "vcs"}'


def _proc(returncode=0, stdout=VALID_JSON, stderr=""):
    m = MagicMock()
    m.returncode = returncode
    m.stdout = stdout
    m.stderr = stderr
    return m


class TestClaudeCodeProvider:
    def test_name(self):
        assert ClaudeCodeProvider().name == "claude-code"

    @patch("agentic_pipeline.agents.providers.claude_code_provider.subprocess.run")
    def test_classify_success_bare_json(self, mock_run):
        mock_run.return_value = _proc()
        profile = ClaudeCodeProvider().classify("some book text")
        assert profile.book_type.value == "technical"
        assert isinstance(profile.confidence, float)
        # list args, no shell, prompt contains the text
        args, kwargs = mock_run.call_args
        cmd = args[0]
        assert cmd[0] == "claude" and cmd[1] == "-p"
        assert "some book text" in cmd[2]
        assert kwargs.get("shell") is not True
        assert kwargs["timeout"] == 120

    @patch("agentic_pipeline.agents.providers.claude_code_provider.subprocess.run")
    def test_classify_success_fenced_json(self, mock_run):
        mock_run.return_value = _proc(stdout=f"```json\n{VALID_JSON}\n```")
        assert ClaudeCodeProvider().classify("t").book_type.value == "technical"

    @patch("agentic_pipeline.agents.providers.claude_code_provider.subprocess.run")
    def test_nonzero_exit_raises_runtime_error(self, mock_run):
        mock_run.return_value = _proc(returncode=1, stdout="", stderr="not logged in")
        with pytest.raises(RuntimeError, match="exit 1"):
            ClaudeCodeProvider().classify("t")

    @patch("agentic_pipeline.agents.providers.claude_code_provider.subprocess.run")
    def test_timeout_raises_runtime_error(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=120)
        with pytest.raises(RuntimeError, match="timed out"):
            ClaudeCodeProvider().classify("t")

    @patch("agentic_pipeline.agents.providers.claude_code_provider.subprocess.run")
    def test_missing_binary_raises_runtime_error(self, mock_run):
        mock_run.side_effect = FileNotFoundError("claude")
        with pytest.raises(RuntimeError, match="unavailable"):
            ClaudeCodeProvider().classify("t")

    @patch("agentic_pipeline.agents.providers.claude_code_provider.subprocess.run")
    def test_garbage_output_raises_value_error_family(self, mock_run):
        mock_run.return_value = _proc(stdout="no json here")
        with pytest.raises((RuntimeError, ValueError)):
            ClaudeCodeProvider().classify("t")

    @patch("agentic_pipeline.agents.providers.claude_code_provider.subprocess.run")
    def test_text_truncated_to_40k(self, mock_run):
        mock_run.return_value = _proc()
        ClaudeCodeProvider().classify("x" * 50000)
        prompt = mock_run.call_args[0][0][2]
        # the prompt embeds at most 40000 chars of book text
        assert "x" * 40001 not in prompt
        assert "x" * 1000 in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_claude_code_provider.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement the provider**

Create `agentic_pipeline/agents/providers/claude_code_provider.py`:

```python
# agentic_pipeline/agents/providers/claude_code_provider.py
"""Claude Code CLI provider — classification via `claude -p`.

Uses the local claude CLI (subscription billing, no API key). Any
failure raises RuntimeError so ClassifierAgent falls back to the next
provider.
"""

import subprocess
from typing import Optional

from agentic_pipeline.agents.classifier_types import BookProfile
from agentic_pipeline.agents.providers.base import LLMProvider, parse_profile_json
from agentic_pipeline.agents.prompts import load_prompt


class ClaudeCodeProvider(LLMProvider):
    """Classification through the `claude` CLI (measured ~14s/book)."""

    def __init__(self, timeout: int = 120):
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "claude-code"

    def classify(self, text: str, metadata: Optional[dict] = None) -> BookProfile:
        prompt_template = load_prompt("classify")
        prompt = prompt_template.format(text=text[:40000])

        try:
            proc = subprocess.run(
                ["claude", "-p", prompt],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"claude -p timed out after {self.timeout}s") from e
        except OSError as e:
            raise RuntimeError(f"claude CLI unavailable: {e}") from e

        if proc.returncode != 0:
            raise RuntimeError(
                f"claude -p exit {proc.returncode}: {proc.stderr.strip()[:200]}"
            )

        return parse_profile_json(proc.stdout.strip())
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_claude_code_provider.py tests/test_provider_base.py -v`
Expected: all PASS. Also `ruff check agentic_pipeline/agents/providers/ tests/test_claude_code_provider.py` → clean.

- [ ] **Step 5: Commit**

```bash
git add agentic_pipeline/agents/providers/claude_code_provider.py tests/test_claude_code_provider.py
git commit -m "feat: ClaudeCodeProvider — classification via claude -p subscription billing

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Provider order + `CLASSIFIER_PROVIDER` override in `ClassifierAgent`

**Files:**
- Modify: `agentic_pipeline/agents/classifier.py` (lines 8-55 imports/docstring/`_get_primary`/`_get_fallback`, lines 88-94 fallback block)
- Test: `tests/test_classifier.py`

**Interfaces:**
- Consumes: `ClaudeCodeProvider` (Task 2), `OpenAIProvider`.
- Produces: default chain claude-code→openai; `CLASSIFIER_PROVIDER` semantics per Global Constraints; `_get_fallback()` may now return `None` (single-provider mode) — `classify()` handles it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_classifier.py` (it already has a `db_path` fixture and provider fakes — reuse its existing fake/mock style; the code below assumes a `db_path` fixture yielding a migrated tmp DB path, which the file's existing tests use):

```python
class _BoomProvider:
    name = "boom"

    def classify(self, text, metadata=None):
        raise RuntimeError("boom")


def test_default_chain_is_claude_code_then_openai(db_path, monkeypatch):
    monkeypatch.delenv("CLASSIFIER_PROVIDER", raising=False)
    monkeypatch.setattr(
        "agentic_pipeline.agents.providers.openai_provider.OpenAI", lambda api_key=None: object()
    )
    agent = ClassifierAgent(db_path)
    assert agent._get_primary().name == "claude-code"
    assert agent._get_fallback().name == "openai"


def test_env_override_forces_single_provider(db_path, monkeypatch):
    monkeypatch.setenv("CLASSIFIER_PROVIDER", "openai")
    monkeypatch.setattr(
        "agentic_pipeline.agents.providers.openai_provider.OpenAI", lambda api_key=None: object()
    )
    agent = ClassifierAgent(db_path)
    assert agent._get_primary().name == "openai"
    assert agent._get_fallback() is None


def test_env_override_no_fallback_attempted(db_path, monkeypatch):
    monkeypatch.setenv("CLASSIFIER_PROVIDER", "claude-code")
    agent = ClassifierAgent(db_path)
    agent.primary = _BoomProvider()
    agent._primary_initialized = True
    profile = agent.classify("text", content_hash="deadbeef" * 8)
    # single-provider mode: primary failed, NO fallback -> unknown profile
    assert profile.book_type.value == "unknown"
    assert agent._get_fallback() is None


def test_invalid_env_value_raises(db_path, monkeypatch):
    monkeypatch.setenv("CLASSIFIER_PROVIDER", "ollama")
    agent = ClassifierAgent(db_path)
    with pytest.raises(ValueError, match="CLASSIFIER_PROVIDER"):
        agent.classify("text", content_hash="cafebabe" * 8)


def test_constructor_injection_beats_env(db_path, monkeypatch):
    monkeypatch.setenv("CLASSIFIER_PROVIDER", "openai")
    injected = _BoomProvider()
    agent = ClassifierAgent(db_path, primary=injected)
    assert agent._get_primary() is injected
```

(Ensure `import pytest` and `from agentic_pipeline.agents.classifier import ClassifierAgent` exist at the top of the file — they already do for the existing tests.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_classifier.py -v`
Expected: new tests FAIL (`_get_primary().name == "openai"` today; `_get_fallback()` returns AnthropicProvider or raises on missing key); the 4 pre-existing tests still PASS (they inject providers).

- [ ] **Step 3: Implement the new chain**

In `agentic_pipeline/agents/classifier.py`:

(a) Imports — replace line 11 (`AnthropicProvider`) with:

```python
from agentic_pipeline.agents.providers.claude_code_provider import ClaudeCodeProvider
```

and add `import os` with the stdlib imports.

(b) Update the class docstring flow (lines 21-26) to:

```python
    """
    Orchestrates book classification using LLM providers.

    Flow:
    1. Check if we've seen this content hash before (cache)
    2. Try primary provider (claude-code by default — subscription billing)
    3. Try fallback provider (OpenAI by default; absent when
       CLASSIFIER_PROVIDER forces single-provider mode)
    4. Return unknown profile if all fail
    """
```

(c) Replace `_get_primary`/`_get_fallback` (lines 43-55):

```python
    _ENV_VAR = "CLASSIFIER_PROVIDER"
    _ENV_PROVIDERS = {"claude-code": ClaudeCodeProvider, "openai": OpenAIProvider}

    def _forced_provider_name(self) -> Optional[str]:
        forced = os.environ.get(self._ENV_VAR)
        if forced is None:
            return None
        if forced not in self._ENV_PROVIDERS:
            raise ValueError(
                f"{self._ENV_VAR} must be one of {sorted(self._ENV_PROVIDERS)}, got {forced!r}"
            )
        return forced

    def _get_primary(self) -> LLMProvider:
        """Get or initialize primary provider (env override wins over default)."""
        if not self._primary_initialized:
            forced = self._forced_provider_name()
            if forced is not None:
                self.primary = self._ENV_PROVIDERS[forced]()
            else:
                self.primary = ClaudeCodeProvider()
            self._primary_initialized = True
        return self.primary

    def _get_fallback(self) -> Optional[LLMProvider]:
        """Get or initialize fallback; None in forced single-provider mode."""
        if not self._fallback_initialized:
            if self._forced_provider_name() is not None:
                self.fallback = None
            else:
                self.fallback = OpenAIProvider()
            self._fallback_initialized = True
        return self.fallback
```

(d) Replace the fallback block in `classify()` (lines 88-94):

```python
        # 3. Try fallback provider (absent in forced single-provider mode)
        fallback = self._get_fallback()
        if fallback is not None:
            try:
                logger.info(f"Calling fallback provider: {fallback.name}")
                return fallback.classify(text, metadata)
            except Exception as e:
                logger.warning(f"Fallback provider ({fallback.name}) failed: {e}")
```

NOTE: `AnthropicProvider` import is removed from this file but the module
`anthropic_provider.py` and its tests stay untouched (spec: it leaves the
default chain, not the codebase).

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_classifier.py tests/test_anthropic_provider.py tests/test_cli_classify.py -v`
Expected: all PASS (anthropic provider tests pin the module still works standalone; cli_classify pins the classify CLI path).

- [ ] **Step 5: Commit**

```bash
git add agentic_pipeline/agents/classifier.py tests/test_classifier.py
git commit -m "feat: claude-code primary classifier with OpenAI fallback + env override

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: PDF branch in `count_source_words`

**Files:**
- Modify: `agentic_pipeline/validation/extraction_validator.py:73-101`
- Test: `tests/test_extraction_validator.py`

**Interfaces:**
- Consumes: PyMuPDF (`import fitz`, lazy) — already installed.
- Produces: `count_source_words` handles `.pdf`; `_count_pdf_words(path) -> Optional[int]` (module-private). `None` contract unchanged for all failure modes and sub-100-word PDFs.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_extraction_validator.py`:

```python
class TestPdfSourceWords:
    @staticmethod
    def _make_pdf(path, n_words: int):
        import fitz

        words = [f"word{i}" for i in range(n_words)]
        doc = fitz.open()
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(36, 36, 559, 806), " ".join(words), fontsize=8)
        doc.save(str(path))
        doc.close()
        return words

    def test_pdf_word_count(self, tmp_path):
        from agentic_pipeline.validation.extraction_validator import count_source_words

        pdf = tmp_path / "book.pdf"
        self._make_pdf(pdf, 150)
        count = count_source_words(str(pdf))
        assert isinstance(count, int)
        assert count == 150

    def test_pdf_under_100_words_returns_none(self, tmp_path):
        from agentic_pipeline.validation.extraction_validator import count_source_words

        pdf = tmp_path / "scan.pdf"
        self._make_pdf(pdf, 10)
        assert count_source_words(str(pdf)) is None

    def test_corrupt_pdf_returns_none(self, tmp_path):
        from agentic_pipeline.validation.extraction_validator import count_source_words

        pdf = tmp_path / "corrupt.pdf"
        pdf.write_bytes(b"not a pdf at all")
        assert count_source_words(str(pdf)) is None

    def test_unsupported_suffix_still_none(self, tmp_path):
        from agentic_pipeline.validation.extraction_validator import count_source_words

        f = tmp_path / "book.mobi"
        f.write_bytes(b"x")
        assert count_source_words(str(f)) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_extraction_validator.py::TestPdfSourceWords -v`
Expected: `test_pdf_word_count` and `test_pdf_under_100_words_returns_none` FAIL (`.pdf` currently returns None early / asserts int fails); the other two already pass — correct.

- [ ] **Step 3: Implement the PDF branch**

Replace `count_source_words` (lines 73-101) with:

```python
def count_source_words(source_path: str) -> Optional[int]:
    """Count words in a source EPUB (HTML documents) or PDF (page text).

    Returns None when the count can't be established (missing file,
    unsupported format, corrupt archive, or a PDF with fewer than 100
    extractable words — image-only scans make coverage meaningless).
    Callers treat None as "skip the check" rather than as a failure.
    """
    path = Path(source_path)
    if not path.is_file():
        return None

    suffix = path.suffix.lower()
    if suffix == ".epub":
        return _count_epub_words(path)
    if suffix == ".pdf":
        return _count_pdf_words(path)
    return None


def _count_epub_words(path: Path) -> Optional[int]:
    try:
        with zipfile.ZipFile(path) as archive:
            total = 0
            for name in archive.namelist():
                if not name.lower().endswith(_HTML_SUFFIXES):
                    continue
                try:
                    html = archive.read(name).decode("utf-8", "ignore")
                except (KeyError, OSError):
                    continue
                parser = _TextExtractor()
                parser.feed(_SCRIPT_STYLE_RE.sub(" ", html))
                total += len(" ".join(parser.chunks).split())
    except (zipfile.BadZipFile, OSError) as e:
        logger.warning(f"Could not read source words from {path.name}: {e}")
        return None

    return total


def _count_pdf_words(path: Path) -> Optional[int]:
    try:
        import fitz  # PyMuPDF — lazy: validators run in contexts without it loaded
    except ImportError:
        logger.warning("PyMuPDF not available; skipping PDF source word count")
        return None

    try:
        with fitz.open(str(path)) as doc:
            total = sum(len(page.get_text().split()) for page in doc)
    except Exception as e:
        logger.warning(f"Could not read source words from {path.name}: {e}")
        return None

    if total < 100:
        # image-only/scanned PDF: a near-zero denominator makes the
        # coverage ratio meaningless — treat as "can't establish a count"
        return None
    return total
```

(The EPUB body is byte-identical to today's, just moved into `_count_epub_words`.)

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_extraction_validator.py -v`
Expected: all PASS (pre-existing EPUB/Check-8 tests pin the refactor).

- [ ] **Step 5: Commit**

```bash
git add agentic_pipeline/validation/extraction_validator.py tests/test_extraction_validator.py
git commit -m "feat: PDF source-coverage counting via PyMuPDF with scanned-PDF guard

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Rejection reason in `status`

**Files:**
- Modify: `agentic_pipeline/cli.py` (`status` command, body starts line ~353; insert after the `approved_by` block at line ~387)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `approval_audit` table (columns `pipeline_id, action, actor, reason` verified live).
- Produces: display-only change; no schema/API changes.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli.py` (follow the file's existing CliRunner + `monkeypatch.setenv("AGENTIC_PIPELINE_DB", ...)` pattern):

```python
def test_status_shows_rejection_reason(tmp_path, monkeypatch):
    import sqlite3

    from click.testing import CliRunner

    from agentic_pipeline.cli import main
    from agentic_pipeline.db.migrations import run_migrations

    db = tmp_path / "pipeline.db"
    monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(db))
    run_migrations(db)

    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO processing_pipelines (id, source_path, content_hash, state) "
        "VALUES ('pipe-1', '/tmp/x.epub', 'hash-1', 'rejected')"
    )
    conn.execute(
        "INSERT INTO approval_audit (book_id, pipeline_id, action, actor, reason) "
        "VALUES ('', 'pipe-1', 'rejected', 'human:cli', 'duplicate of live copy')"
    )
    conn.commit()
    conn.close()

    result = CliRunner().invoke(main, ["status", "pipe-1"])
    assert result.exit_code == 0
    assert "human:cli" in result.output
    assert "duplicate of live copy" in result.output


def test_status_no_reason_line_when_not_rejected(tmp_path, monkeypatch):
    import sqlite3

    from click.testing import CliRunner

    from agentic_pipeline.cli import main
    from agentic_pipeline.db.migrations import run_migrations

    db = tmp_path / "pipeline.db"
    monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(db))
    run_migrations(db)

    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO processing_pipelines (id, source_path, content_hash, state) "
        "VALUES ('pipe-2', '/tmp/y.epub', 'hash-2', 'complete')"
    )
    conn.commit()
    conn.close()

    result = CliRunner().invoke(main, ["status", "pipe-2"])
    assert result.exit_code == 0
    assert "Rejected by" not in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cli.py -v -k "rejection_reason or no_reason_line"`
Expected: `test_status_shows_rejection_reason` FAIL (no reason in output); the negative test may already pass.

- [ ] **Step 3: Implement**

In the `status` command body, after the `approved_by` block (`if pipeline.get("approved_by"): ...`), add:

```python
    if pipeline["state"] == "rejected":
        import sqlite3

        conn = sqlite3.connect(db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            audit = conn.execute(
                "SELECT actor, reason FROM approval_audit "
                "WHERE pipeline_id = ? AND action = 'rejected' "
                "ORDER BY id DESC LIMIT 1",
                (pipeline_id,),
            ).fetchone()
        finally:
            conn.close()
        if audit:
            console.print(f"  Rejected by: {audit['actor']} — \"{audit['reason']}\"")
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_cli.py -v`
Expected: all PASS (whole file — pre-existing status tests must stay green).

- [ ] **Step 5: Commit**

```bash
git add agentic_pipeline/cli.py tests/test_cli.py
git commit -m "feat: status shows rejection actor and reason from approval_audit

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Docs sync + full-suite gate

**Files:**
- Modify: `ref/module-map.md` (add `claude_code_provider.py` row; note base.py's shared parser)
- Modify: `ref/pipeline-architecture.md` (classifier provider order + CLASSIFIER_PROVIDER, if the file documents the classifier — read it first and match its format; if it doesn't cover providers, skip it and say so in the commit body)
- Modify: `ref/cli-commands.md` (status entry: mentions rejection reason display)

**Interfaces:** consumes everything prior; produces docs matching reality + a green suite.

- [ ] **Step 1: Update docs** (read each file's format first, match exactly)

module-map addition:

```markdown
| `agentic_pipeline/agents/providers/claude_code_provider.py` | ClaudeCodeProvider — classification via `claude -p` (subscription billing); default primary since 2026-07-16 |
```

and amend the base.py row to mention `parse_profile_json` (shared response parsing for all providers).

cli-commands: in the `status` entry, append: "For rejected pipelines, prints the rejecting actor and reason from the audit trail."

- [ ] **Step 2: Full-suite gate**

Run: `source .venv/bin/activate && make test && ruff check .`
Expected: ALL tests pass (~660 total incl. ~20 new), ruff clean.

- [ ] **Step 3: Commit**

```bash
git add ref/module-map.md ref/cli-commands.md ref/pipeline-architecture.md
git commit -m "docs: sync ref docs for claude-code provider, PDF validation, status reasons

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Post-merge operational runbook (spec §5 — NOT plan tasks)

1. `launchctl kickstart -k gui/$(id -u)/com.taylorstephens.agentic-pipeline-worker`
2. Reingest the 4 recoverable books (pipeline ids from the doctor manifest); verify each classifies with provider `claude-code` (book_profile in the pipeline row / worker log) — the smoke test. Approve; chunks regenerate under the fixed chunker.
3. If any fell back to OpenAI, diagnose claude CLI auth in the launchd context before trusting subscription billing.
4. Write `<db-dir>/doctor/awaiting-redownload.md` for Designing Experiences + Git Mastery: Accelerated Crash Course (book ids from the DB).
5. `agentic-pipeline doctor` — expect clean.
```

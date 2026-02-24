# Phase 2: Classifier Agent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a model-agnostic classifier agent that uses LLMs to determine book types, with OpenAI as primary and Anthropic as fallback.

**Architecture:** Thin abstraction over LLM providers. ClassifierAgent orchestrates: check cache (existing pipeline) → try primary → try fallback → return conservative default. All providers implement the same interface.

**Tech Stack:** Python 3.12, openai>=1.0, anthropic>=0.18, pytest

---

## Prerequisites

- Working directory: `/path/to/book-mcp-server`
- Phase 1 complete (pipeline states, repository, etc.)
- API keys available: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`

---

## Task 1: Add Dependencies

**Files:**
- Modify: `requirements.txt`
- Modify: `pyproject.toml`

**Step 1: Update requirements.txt**

Add to the end of `/path/to/book-mcp-server/requirements.txt`:

```text
openai>=1.0
anthropic>=0.18
```

**Step 2: Update pyproject.toml**

Add to the dependencies list in `/path/to/book-mcp-server/pyproject.toml`:

```toml
[project]
dependencies = [
    "click>=8.0",
    "rich>=13.0",
    "watchdog>=3.0",
    "mcp>=1.0",
    "openai>=1.0",
    "anthropic>=0.18",
]
```

**Step 3: Install dependencies**

Run: `source .venv/bin/activate && pip install openai anthropic`

**Step 4: Verify installation**

Run: `python -c "import openai; import anthropic; print('OK')"`

Expected: `OK`

**Step 5: Commit**

```bash
git add requirements.txt pyproject.toml
git commit -m "chore: add openai and anthropic dependencies"
```

---

## Task 2: Data Structures (BookType, BookProfile)

**Files:**
- Create: `agentic_pipeline/agents/classifier_types.py`
- Create: `tests/test_classifier_types.py`

**Step 1: Write the failing test**

```python
# tests/test_classifier_types.py
"""Tests for classifier data types."""

import pytest


def test_book_type_enum_has_required_types():
    from agentic_pipeline.agents.classifier_types import BookType

    required = [
        "TECHNICAL_TUTORIAL", "TECHNICAL_REFERENCE", "TEXTBOOK",
        "NARRATIVE_NONFICTION", "PERIODICAL", "RESEARCH_COLLECTION", "UNKNOWN"
    ]

    for book_type in required:
        assert hasattr(BookType, book_type), f"Missing type: {book_type}"


def test_book_profile_creation():
    from agentic_pipeline.agents.classifier_types import BookProfile, BookType

    profile = BookProfile(
        book_type=BookType.TECHNICAL_TUTORIAL,
        confidence=0.85,
        suggested_tags=["python", "web"],
        reasoning="Contains code examples"
    )

    assert profile.book_type == BookType.TECHNICAL_TUTORIAL
    assert profile.confidence == 0.85
    assert profile.suggested_tags == ["python", "web"]
    assert profile.reasoning == "Contains code examples"


def test_book_profile_to_dict():
    from agentic_pipeline.agents.classifier_types import BookProfile, BookType

    profile = BookProfile(
        book_type=BookType.TEXTBOOK,
        confidence=0.72,
        suggested_tags=["economics"],
        reasoning="Academic structure"
    )

    d = profile.to_dict()

    assert d["book_type"] == "textbook"
    assert d["confidence"] == 0.72
    assert d["suggested_tags"] == ["economics"]
    assert d["reasoning"] == "Academic structure"


def test_book_profile_from_dict():
    from agentic_pipeline.agents.classifier_types import BookProfile, BookType

    d = {
        "book_type": "periodical",
        "confidence": 0.90,
        "suggested_tags": ["news", "politics"],
        "reasoning": "Article format with bylines"
    }

    profile = BookProfile.from_dict(d)

    assert profile.book_type == BookType.PERIODICAL
    assert profile.confidence == 0.90
    assert profile.suggested_tags == ["news", "politics"]
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_classifier_types.py -v`

Expected: FAIL with "No module named 'agentic_pipeline.agents.classifier_types'"

**Step 3: Write minimal implementation**

```python
# agentic_pipeline/agents/classifier_types.py
"""Data types for the classifier agent."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BookType(Enum):
    """Types of books the classifier can identify."""

    TECHNICAL_TUTORIAL = "technical_tutorial"
    TECHNICAL_REFERENCE = "technical_reference"
    TEXTBOOK = "textbook"
    NARRATIVE_NONFICTION = "narrative_nonfiction"
    PERIODICAL = "periodical"
    RESEARCH_COLLECTION = "research_collection"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str) -> "BookType":
        """Convert string to BookType, defaulting to UNKNOWN."""
        value = value.lower().strip()
        for member in cls:
            if member.value == value:
                return member
        return cls.UNKNOWN


@dataclass
class BookProfile:
    """Classification result for a book."""

    book_type: BookType
    confidence: float
    suggested_tags: list[str] = field(default_factory=list)
    reasoning: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "book_type": self.book_type.value,
            "confidence": self.confidence,
            "suggested_tags": self.suggested_tags,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BookProfile":
        """Create from dictionary."""
        return cls(
            book_type=BookType.from_string(data.get("book_type", "unknown")),
            confidence=data.get("confidence", 0.0),
            suggested_tags=data.get("suggested_tags", []),
            reasoning=data.get("reasoning", ""),
        )

    @classmethod
    def unknown(cls, reasoning: str = "Classification failed") -> "BookProfile":
        """Create an unknown profile (used for fallback)."""
        return cls(
            book_type=BookType.UNKNOWN,
            confidence=0.0,
            suggested_tags=[],
            reasoning=reasoning,
        )
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/test_classifier_types.py -v`

Expected: 4 passed

**Step 5: Commit**

```bash
git add agentic_pipeline/agents/classifier_types.py tests/test_classifier_types.py
git commit -m "feat: add BookType enum and BookProfile dataclass"
```

---

## Task 3: LLM Provider Base Class

**Files:**
- Create: `agentic_pipeline/agents/providers/__init__.py`
- Create: `agentic_pipeline/agents/providers/base.py`
- Create: `tests/test_provider_base.py`

**Step 1: Create providers directory**

Run: `mkdir -p agentic_pipeline/agents/providers && touch agentic_pipeline/agents/providers/__init__.py`

**Step 2: Write the failing test**

```python
# tests/test_provider_base.py
"""Tests for LLM provider base class."""

import pytest


def test_llm_provider_is_abstract():
    from agentic_pipeline.agents.providers.base import LLMProvider

    with pytest.raises(TypeError):
        LLMProvider()  # Can't instantiate abstract class


def test_llm_provider_defines_required_methods():
    from agentic_pipeline.agents.providers.base import LLMProvider
    import inspect

    # Check abstract methods exist
    assert hasattr(LLMProvider, 'classify')
    assert hasattr(LLMProvider, 'name')

    # Check they are abstract
    assert getattr(LLMProvider.classify, '__isabstractmethod__', False)
    assert isinstance(inspect.getattr_static(LLMProvider, 'name'), property)
```

**Step 3: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_provider_base.py -v`

Expected: FAIL with "No module named 'agentic_pipeline.agents.providers.base'"

**Step 4: Write minimal implementation**

```python
# agentic_pipeline/agents/providers/base.py
"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Optional

from agentic_pipeline.agents.classifier_types import BookProfile


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def classify(self, text: str, metadata: Optional[dict] = None) -> BookProfile:
        """
        Classify book text and return a BookProfile.

        Args:
            text: Pre-extracted book content (truncated to ~10K tokens)
            metadata: Optional hints (filename, source folder, etc.)

        Returns:
            BookProfile with classification results
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging (e.g., 'openai', 'anthropic')."""
        pass
```

**Step 5: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/test_provider_base.py -v`

Expected: 2 passed

**Step 6: Commit**

```bash
git add agentic_pipeline/agents/providers/ tests/test_provider_base.py
git commit -m "feat: add LLMProvider abstract base class"
```

---

## Task 4: Classification Prompt Template

**Files:**
- Create: `agentic_pipeline/agents/prompts/classify.txt`
- Create: `agentic_pipeline/agents/prompts/__init__.py`
- Create: `tests/test_prompts.py`

**Step 1: Create prompts directory**

Run: `mkdir -p agentic_pipeline/agents/prompts && touch agentic_pipeline/agents/prompts/__init__.py`

**Step 2: Create prompt template**

```text
# agentic_pipeline/agents/prompts/classify.txt
You are a book classifier. Analyze the provided text and classify it.

## Book Types
- technical_tutorial: Step-by-step guides teaching skills (e.g., "Learning Python")
- technical_reference: Reference manuals, documentation (e.g., "PostgreSQL Manual")
- textbook: Academic educational material with exercises (e.g., "Intro to Economics")
- narrative_nonfiction: Stories, biographies, essays (e.g., "Steve Jobs")
- periodical: Magazines, newspapers, journals
- research_collection: Academic papers, proceedings
- unknown: Cannot determine with confidence

## Instructions
1. Read the sample text carefully
2. Identify structural patterns (chapters, code blocks, bylines, etc.)
3. Choose the single best matching book type
4. Provide confidence (0.0-1.0) based on how clearly it matches
5. Suggest 3-5 topic tags
6. Explain your reasoning in 1-2 sentences

## Output Format
Return ONLY valid JSON with no additional text:
{"book_type": "...", "confidence": 0.85, "suggested_tags": ["tag1", "tag2"], "reasoning": "..."}

## Text Sample
{text}
```

**Step 3: Write the test**

```python
# tests/test_prompts.py
"""Tests for prompt loading."""

import pytest
from pathlib import Path


def test_load_classify_prompt():
    from agentic_pipeline.agents.prompts import load_prompt

    prompt = load_prompt("classify")

    assert "book classifier" in prompt.lower()
    assert "{text}" in prompt
    assert "technical_tutorial" in prompt


def test_format_classify_prompt():
    from agentic_pipeline.agents.prompts import load_prompt

    prompt = load_prompt("classify")
    formatted = prompt.format(text="Sample book content here")

    assert "Sample book content here" in formatted
    assert "{text}" not in formatted
```

**Step 4: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_prompts.py -v`

Expected: FAIL

**Step 5: Write the prompt loader**

```python
# agentic_pipeline/agents/prompts/__init__.py
"""Prompt template loading."""

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """Load a prompt template by name."""
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise ValueError(f"Prompt not found: {name}")
    return path.read_text()
```

**Step 6: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/test_prompts.py -v`

Expected: 2 passed

**Step 7: Commit**

```bash
git add agentic_pipeline/agents/prompts/ tests/test_prompts.py
git commit -m "feat: add classification prompt template"
```

---

## Task 5: OpenAI Provider

**Files:**
- Create: `agentic_pipeline/agents/providers/openai_provider.py`
- Create: `tests/test_openai_provider.py`

**Step 1: Write the failing test**

```python
# tests/test_openai_provider.py
"""Tests for OpenAI provider."""

import pytest
import json
from unittest.mock import Mock, patch


def test_openai_provider_has_correct_name():
    from agentic_pipeline.agents.providers.openai_provider import OpenAIProvider

    provider = OpenAIProvider(api_key="test-key")
    assert provider.name == "openai"


def test_openai_provider_default_model():
    from agentic_pipeline.agents.providers.openai_provider import OpenAIProvider

    provider = OpenAIProvider(api_key="test-key")
    assert provider.model == "gpt-4o-mini"


def test_openai_provider_custom_model():
    from agentic_pipeline.agents.providers.openai_provider import OpenAIProvider

    provider = OpenAIProvider(api_key="test-key", model="gpt-4o")
    assert provider.model == "gpt-4o"


def test_openai_provider_parses_valid_response():
    from agentic_pipeline.agents.providers.openai_provider import OpenAIProvider
    from agentic_pipeline.agents.classifier_types import BookType

    provider = OpenAIProvider(api_key="test-key")

    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = json.dumps({
        "book_type": "technical_tutorial",
        "confidence": 0.92,
        "suggested_tags": ["python", "programming"],
        "reasoning": "Contains code examples and exercises"
    })

    with patch.object(provider, '_client') as mock_client:
        mock_client.chat.completions.create.return_value = mock_response
        result = provider.classify("sample text")

    assert result.book_type == BookType.TECHNICAL_TUTORIAL
    assert result.confidence == 0.92
    assert "python" in result.suggested_tags


def test_openai_provider_handles_malformed_json():
    from agentic_pipeline.agents.providers.openai_provider import OpenAIProvider

    provider = OpenAIProvider(api_key="test-key")

    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "This is not JSON"

    with patch.object(provider, '_client') as mock_client:
        mock_client.chat.completions.create.return_value = mock_response

        with pytest.raises(ValueError, match="Failed to parse"):
            provider.classify("sample text")
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_openai_provider.py -v`

Expected: FAIL

**Step 3: Write minimal implementation**

```python
# agentic_pipeline/agents/providers/openai_provider.py
"""OpenAI LLM provider."""

import json
import os
from typing import Optional

from openai import OpenAI

from agentic_pipeline.agents.classifier_types import BookProfile
from agentic_pipeline.agents.providers.base import LLMProvider
from agentic_pipeline.agents.prompts import load_prompt


class OpenAIProvider(LLMProvider):
    """OpenAI-based classification provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self._client = OpenAI(api_key=self.api_key)

    @property
    def name(self) -> str:
        return "openai"

    def classify(self, text: str, metadata: Optional[dict] = None) -> BookProfile:
        """Classify text using OpenAI."""
        prompt_template = load_prompt("classify")
        prompt = prompt_template.format(text=text[:40000])  # ~10K tokens

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a book classifier. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,  # Low temperature for consistent classification
        )

        content = response.choices[0].message.content.strip()
        return self._parse_response(content)

    def _parse_response(self, content: str) -> BookProfile:
        """Parse JSON response into BookProfile."""
        # Try to extract JSON if wrapped in markdown code blocks
        if "```" in content:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                content = content[start:end]

        try:
            data = json.loads(content)
            return BookProfile.from_dict(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/test_openai_provider.py -v`

Expected: 5 passed

**Step 5: Update providers __init__.py**

```python
# agentic_pipeline/agents/providers/__init__.py
"""LLM providers for classification."""

from agentic_pipeline.agents.providers.base import LLMProvider
from agentic_pipeline.agents.providers.openai_provider import OpenAIProvider

__all__ = ["LLMProvider", "OpenAIProvider"]
```

**Step 6: Commit**

```bash
git add agentic_pipeline/agents/providers/ tests/test_openai_provider.py
git commit -m "feat: add OpenAI classification provider"
```

---

## Task 6: Anthropic Provider

**Files:**
- Create: `agentic_pipeline/agents/providers/anthropic_provider.py`
- Create: `tests/test_anthropic_provider.py`

**Step 1: Write the failing test**

```python
# tests/test_anthropic_provider.py
"""Tests for Anthropic provider."""

import pytest
import json
from unittest.mock import Mock, patch


def test_anthropic_provider_has_correct_name():
    from agentic_pipeline.agents.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(api_key="test-key")
    assert provider.name == "anthropic"


def test_anthropic_provider_default_model():
    from agentic_pipeline.agents.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(api_key="test-key")
    assert "claude" in provider.model.lower()


def test_anthropic_provider_parses_valid_response():
    from agentic_pipeline.agents.providers.anthropic_provider import AnthropicProvider
    from agentic_pipeline.agents.classifier_types import BookType

    provider = AnthropicProvider(api_key="test-key")

    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = json.dumps({
        "book_type": "narrative_nonfiction",
        "confidence": 0.88,
        "suggested_tags": ["biography", "technology"],
        "reasoning": "Narrative structure following a person's life"
    })

    with patch.object(provider, '_client') as mock_client:
        mock_client.messages.create.return_value = mock_response
        result = provider.classify("sample text")

    assert result.book_type == BookType.NARRATIVE_NONFICTION
    assert result.confidence == 0.88


def test_anthropic_provider_handles_malformed_json():
    from agentic_pipeline.agents.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(api_key="test-key")

    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = "Not valid JSON at all"

    with patch.object(provider, '_client') as mock_client:
        mock_client.messages.create.return_value = mock_response

        with pytest.raises(ValueError, match="Failed to parse"):
            provider.classify("sample text")
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_anthropic_provider.py -v`

Expected: FAIL

**Step 3: Write minimal implementation**

```python
# agentic_pipeline/agents/providers/anthropic_provider.py
"""Anthropic Claude LLM provider."""

import json
import os
from typing import Optional

from anthropic import Anthropic

from agentic_pipeline.agents.classifier_types import BookProfile
from agentic_pipeline.agents.providers.base import LLMProvider
from agentic_pipeline.agents.prompts import load_prompt


class AnthropicProvider(LLMProvider):
    """Anthropic Claude-based classification provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-haiku-20240307",
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self._client = Anthropic(api_key=self.api_key)

    @property
    def name(self) -> str:
        return "anthropic"

    def classify(self, text: str, metadata: Optional[dict] = None) -> BookProfile:
        """Classify text using Anthropic Claude."""
        prompt_template = load_prompt("classify")
        prompt = prompt_template.format(text=text[:40000])  # ~10K tokens

        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt},
            ],
            system="You are a book classifier. Return only valid JSON.",
            temperature=0.1,
        )

        content = response.content[0].text.strip()
        return self._parse_response(content)

    def _parse_response(self, content: str) -> BookProfile:
        """Parse JSON response into BookProfile."""
        # Try to extract JSON if wrapped in markdown code blocks
        if "```" in content:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                content = content[start:end]

        try:
            data = json.loads(content)
            return BookProfile.from_dict(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/test_anthropic_provider.py -v`

Expected: 4 passed

**Step 5: Update providers __init__.py**

```python
# agentic_pipeline/agents/providers/__init__.py
"""LLM providers for classification."""

from agentic_pipeline.agents.providers.base import LLMProvider
from agentic_pipeline.agents.providers.openai_provider import OpenAIProvider
from agentic_pipeline.agents.providers.anthropic_provider import AnthropicProvider

__all__ = ["LLMProvider", "OpenAIProvider", "AnthropicProvider"]
```

**Step 6: Commit**

```bash
git add agentic_pipeline/agents/providers/ tests/test_anthropic_provider.py
git commit -m "feat: add Anthropic classification provider"
```

---

## Task 7: Classifier Agent (Main Class)

**Files:**
- Create: `agentic_pipeline/agents/classifier.py`
- Create: `tests/test_classifier.py`

**Step 1: Write the failing test**

```python
# tests/test_classifier.py
"""Tests for the ClassifierAgent."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock


@pytest.fixture
def db_path():
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    yield path
    path.unlink(missing_ok=True)


class MockProvider:
    """Mock LLM provider for testing."""

    def __init__(self, response=None, should_fail=False):
        from agentic_pipeline.agents.classifier_types import BookProfile, BookType
        self.response = response or BookProfile(
            book_type=BookType.TECHNICAL_TUTORIAL,
            confidence=0.9,
            suggested_tags=["test"],
            reasoning="Test response"
        )
        self.should_fail = should_fail
        self.call_count = 0

    @property
    def name(self):
        return "mock"

    def classify(self, text, metadata=None):
        self.call_count += 1
        if self.should_fail:
            raise Exception("Mock failure")
        return self.response


def test_classifier_returns_cached_result(db_path):
    from agentic_pipeline.agents.classifier import ClassifierAgent
    from agentic_pipeline.agents.classifier_types import BookType
    from agentic_pipeline.db.pipelines import PipelineRepository
    from agentic_pipeline.pipeline.states import PipelineState

    # Setup: create pipeline with existing profile
    repo = PipelineRepository(db_path)
    pid = repo.create("/book.epub", "hash123")
    repo.update_book_profile(pid, {
        "book_type": "textbook",
        "confidence": 0.85,
        "suggested_tags": ["economics"],
        "reasoning": "Cached result"
    })

    mock_provider = MockProvider()
    agent = ClassifierAgent(db_path, primary=mock_provider)

    # Should return cached, not call provider
    result = agent.classify("any text", content_hash="hash123")

    assert result.book_type == BookType.TEXTBOOK
    assert result.confidence == 0.85
    assert mock_provider.call_count == 0  # Did not call LLM


def test_classifier_calls_primary_on_cache_miss(db_path):
    from agentic_pipeline.agents.classifier import ClassifierAgent
    from agentic_pipeline.agents.classifier_types import BookType

    mock_provider = MockProvider()
    agent = ClassifierAgent(db_path, primary=mock_provider)

    result = agent.classify("book text", content_hash="new-hash")

    assert result.book_type == BookType.TECHNICAL_TUTORIAL
    assert mock_provider.call_count == 1


def test_classifier_falls_back_on_primary_failure(db_path):
    from agentic_pipeline.agents.classifier import ClassifierAgent
    from agentic_pipeline.agents.classifier_types import BookProfile, BookType

    primary = MockProvider(should_fail=True)
    fallback = MockProvider(response=BookProfile(
        book_type=BookType.PERIODICAL,
        confidence=0.8,
        suggested_tags=["news"],
        reasoning="Fallback response"
    ))

    agent = ClassifierAgent(db_path, primary=primary, fallback=fallback)

    result = agent.classify("text", content_hash="hash456")

    assert result.book_type == BookType.PERIODICAL
    assert primary.call_count == 1
    assert fallback.call_count == 1


def test_classifier_returns_unknown_when_both_fail(db_path):
    from agentic_pipeline.agents.classifier import ClassifierAgent
    from agentic_pipeline.agents.classifier_types import BookType

    primary = MockProvider(should_fail=True)
    fallback = MockProvider(should_fail=True)

    agent = ClassifierAgent(db_path, primary=primary, fallback=fallback)

    result = agent.classify("text", content_hash="hash789")

    assert result.book_type == BookType.UNKNOWN
    assert result.confidence == 0.0
    assert "failed" in result.reasoning.lower()
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_classifier.py -v`

Expected: FAIL

**Step 3: Write minimal implementation**

```python
# agentic_pipeline/agents/classifier.py
"""Classifier Agent - orchestrates book classification."""

import logging
from pathlib import Path
from typing import Optional

from agentic_pipeline.agents.classifier_types import BookProfile
from agentic_pipeline.agents.providers.base import LLMProvider
from agentic_pipeline.agents.providers.openai_provider import OpenAIProvider
from agentic_pipeline.agents.providers.anthropic_provider import AnthropicProvider
from agentic_pipeline.db.pipelines import PipelineRepository

logger = logging.getLogger(__name__)


class ClassifierAgent:
    """
    Orchestrates book classification using LLM providers.

    Flow:
    1. Check if we've seen this content hash before (cache)
    2. Try primary provider (OpenAI by default)
    3. Try fallback provider (Anthropic by default)
    4. Return unknown profile if all fail
    """

    def __init__(
        self,
        db_path: Path,
        primary: Optional[LLMProvider] = None,
        fallback: Optional[LLMProvider] = None,
    ):
        self.db_path = db_path
        self.repo = PipelineRepository(db_path)
        self.primary = primary
        self.fallback = fallback

        # Lazy initialization of default providers
        self._primary_initialized = primary is not None
        self._fallback_initialized = fallback is not None

    def _get_primary(self) -> LLMProvider:
        """Get or initialize primary provider."""
        if not self._primary_initialized:
            self.primary = OpenAIProvider()
            self._primary_initialized = True
        return self.primary

    def _get_fallback(self) -> LLMProvider:
        """Get or initialize fallback provider."""
        if not self._fallback_initialized:
            self.fallback = AnthropicProvider()
            self._fallback_initialized = True
        return self.fallback

    def classify(
        self,
        text: str,
        content_hash: str,
        metadata: Optional[dict] = None,
    ) -> BookProfile:
        """
        Classify book text and return a BookProfile.

        Args:
            text: Pre-extracted book content
            content_hash: Hash of book content (for caching)
            metadata: Optional hints (filename, etc.)

        Returns:
            BookProfile with classification results
        """
        # 1. Check cache (existing pipeline with this hash)
        cached = self._check_cache(content_hash)
        if cached:
            logger.info(f"Cache hit for {content_hash[:8]}")
            return cached

        # 2. Try primary provider
        try:
            primary = self._get_primary()
            logger.info(f"Calling primary provider: {primary.name}")
            return primary.classify(text, metadata)
        except Exception as e:
            logger.warning(f"Primary provider ({self._get_primary().name}) failed: {e}")

        # 3. Try fallback provider
        try:
            fallback = self._get_fallback()
            logger.info(f"Calling fallback provider: {fallback.name}")
            return fallback.classify(text, metadata)
        except Exception as e:
            logger.warning(f"Fallback provider ({self._get_fallback().name}) failed: {e}")

        # 4. Return safe default
        logger.error("All providers failed, returning unknown")
        return BookProfile.unknown("Classification failed - all providers unavailable")

    def _check_cache(self, content_hash: str) -> Optional[BookProfile]:
        """Check if we have a cached classification for this hash."""
        existing = self.repo.find_by_hash(content_hash)
        if existing and existing.get("book_profile"):
            try:
                import json
                profile_data = existing["book_profile"]
                if isinstance(profile_data, str):
                    profile_data = json.loads(profile_data)
                return BookProfile.from_dict(profile_data)
            except Exception as e:
                logger.warning(f"Failed to parse cached profile: {e}")
        return None
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/test_classifier.py -v`

Expected: 4 passed

**Step 5: Update agents __init__.py**

```python
# agentic_pipeline/agents/__init__.py
"""Agents for the agentic pipeline."""

from agentic_pipeline.agents.classifier import ClassifierAgent
from agentic_pipeline.agents.classifier_types import BookProfile, BookType

__all__ = ["ClassifierAgent", "BookProfile", "BookType"]
```

**Step 6: Commit**

```bash
git add agentic_pipeline/agents/ tests/test_classifier.py
git commit -m "feat: add ClassifierAgent with caching and fallback"
```

---

## Task 8: CLI Command for Classification

**Files:**
- Modify: `agentic_pipeline/cli.py`
- Create: `tests/test_cli_classify.py`

**Step 1: Write the failing test**

```python
# tests/test_cli_classify.py
"""Tests for classify CLI command."""

import pytest
import tempfile
from pathlib import Path
from click.testing import CliRunner
from unittest.mock import patch, Mock


@pytest.fixture
def temp_db(monkeypatch):
    from agentic_pipeline.db.migrations import run_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    run_migrations(path)
    monkeypatch.setenv("AGENTIC_PIPELINE_DB", str(path))
    yield path
    path.unlink(missing_ok=True)


def test_classify_command_exists():
    from agentic_pipeline.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["classify", "--help"])

    assert result.exit_code == 0
    assert "Classify" in result.output or "classify" in result.output


def test_classify_command_with_mock_provider(temp_db):
    from agentic_pipeline.cli import main
    from agentic_pipeline.agents.classifier_types import BookProfile, BookType

    mock_profile = BookProfile(
        book_type=BookType.TECHNICAL_TUTORIAL,
        confidence=0.9,
        suggested_tags=["python"],
        reasoning="Test"
    )

    with patch('agentic_pipeline.cli.ClassifierAgent') as MockAgent:
        mock_instance = Mock()
        mock_instance.classify.return_value = mock_profile
        MockAgent.return_value = mock_instance

        runner = CliRunner()
        result = runner.invoke(main, ["classify", "--text", "Sample book content"])

    assert result.exit_code == 0
    assert "technical_tutorial" in result.output.lower()
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_cli_classify.py -v`

Expected: FAIL

**Step 3: Add classify command to CLI**

Add to `agentic_pipeline/cli.py`:

```python
@main.command()
@click.option("--text", "-t", required=True, help="Text to classify (or path to file)")
@click.option("--provider", "-p", default="openai", help="Provider to use (openai, anthropic)")
def classify(text: str, provider: str):
    """Classify book text and show the result."""
    import hashlib
    from .db.config import get_db_path
    from .agents.classifier import ClassifierAgent
    from .agents.providers.openai_provider import OpenAIProvider
    from .agents.providers.anthropic_provider import AnthropicProvider

    # Check if text is a file path
    text_path = Path(text)
    if text_path.exists():
        text = text_path.read_text()
        console.print(f"[dim]Read {len(text)} chars from {text_path}[/dim]")

    # Generate hash from text
    content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

    # Select provider
    if provider == "anthropic":
        primary = AnthropicProvider()
    else:
        primary = OpenAIProvider()

    db_path = get_db_path()
    agent = ClassifierAgent(db_path, primary=primary)

    console.print(f"[blue]Classifying with {provider}...[/blue]")

    result = agent.classify(text, content_hash=content_hash)

    console.print(f"\n[bold]Classification Result:[/bold]")
    console.print(f"  Type: [cyan]{result.book_type.value}[/cyan]")

    conf = result.confidence
    conf_style = "green" if conf >= 0.8 else "yellow" if conf >= 0.5 else "red"
    console.print(f"  Confidence: [{conf_style}]{conf:.0%}[/{conf_style}]")

    if result.suggested_tags:
        console.print(f"  Tags: {', '.join(result.suggested_tags)}")

    console.print(f"  Reasoning: [dim]{result.reasoning}[/dim]")
```

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/test_cli_classify.py -v`

Expected: 2 passed

**Step 5: Commit**

```bash
git add agentic_pipeline/cli.py tests/test_cli_classify.py
git commit -m "feat: add classify CLI command"
```

---

## Task 9: Run All Tests & Final Commit

**Step 1: Run full test suite**

Run: `source .venv/bin/activate && python -m pytest tests/ -v`

Expected: All tests pass (30+ tests)

**Step 2: Verify CLI works**

Run: `source .venv/bin/activate && python -m agentic_pipeline.cli classify --help`

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete Phase 2 - Classifier Agent

- BookType enum and BookProfile dataclass
- LLMProvider abstract base class
- OpenAI provider (gpt-4o-mini default)
- Anthropic provider (claude-3-haiku fallback)
- ClassifierAgent with caching and fallback
- CLI classify command

Ready for integration with pipeline orchestrator"
```

---

## Summary

Phase 2 delivers:

| Component | Status |
|-----------|--------|
| Data types (BookType, BookProfile) | ✅ |
| LLMProvider base class | ✅ |
| Classification prompt | ✅ |
| OpenAI provider | ✅ |
| Anthropic provider | ✅ |
| ClassifierAgent | ✅ |
| CLI classify command | ✅ |

**Total: 9 tasks, ~45 steps, ~9 commits**

**Dependencies added:** openai>=1.0, anthropic>=0.18

Next: Phase 3 - Pipeline Orchestrator (connect classifier to state machine)

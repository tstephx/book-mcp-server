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

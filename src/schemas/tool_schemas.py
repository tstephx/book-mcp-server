"""Tool input schemas - centralized validation

Following MCP best practice from Chapter 6:
"Create a schema.ts file that contains the schemas for the tools...
By using this schema, we ensure that registering a tool is consistent."

These schemas:
- Validate all tool inputs in one place
- Provide clear error messages
- Document expected input formats
- Enable type safety
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Optional
import re

# UUID validation pattern
UUID_PATTERN = r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'

class SearchInput(BaseModel):
    """Schema for keyword search operations

    Used by: search_titles tool
    """
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "docker networking",
            "limit": 10
        }
    })

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Search query text"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Maximum number of results"
    )

    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        """Ensure query is not just whitespace"""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()

class SemanticSearchInput(BaseModel):
    """Schema for semantic search operations

    Used by: semantic_search tool
    """
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "container orchestration concepts",
            "limit": 5,
            "min_similarity": 0.4
        }
    })

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Semantic search query"
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results"
    )
    min_similarity: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold (0.0-1.0)"
    )

    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        """Ensure query is not just whitespace"""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()

    @field_validator('min_similarity')
    @classmethod
    def validate_similarity(cls, v):
        """Ensure similarity is in valid range"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("min_similarity must be between 0.0 and 1.0")
        return v

class BookIdInput(BaseModel):
    """Schema for book ID operations

    Used by: get_book_info, get_table_of_contents tools
    """
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "book_id": "4e7e9727-d82d-49ca-a32c-990a6619cd75"
        }
    })

    book_id: str = Field(
        ...,
        description="UUID of the book"
    )

    @field_validator('book_id')
    @classmethod
    def validate_book_id(cls, v):
        """Ensure book_id is a valid UUID"""
        if not re.match(UUID_PATTERN, v.lower()):
            raise ValueError(f"Invalid book_id format. Expected UUID, got: {v}")
        return v.lower()

class ChapterInput(BaseModel):
    """Schema for chapter retrieval operations

    Used by: get_chapter, get_section tools
    """
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "book_id": "4e7e9727-d82d-49ca-a32c-990a6619cd75",
            "chapter_number": 6,
            "section_number": 2
        }
    })

    book_id: str = Field(
        ...,
        description="UUID of the book"
    )
    chapter_number: int = Field(
        ...,
        ge=1,
        description="Chapter number (1-indexed)"
    )
    section_number: Optional[int] = Field(
        None,
        ge=1,
        description="Section number within chapter (optional)"
    )

    @field_validator('book_id')
    @classmethod
    def validate_book_id(cls, v):
        """Ensure book_id is a valid UUID"""
        if not re.match(UUID_PATTERN, v.lower()):
            raise ValueError(f"Invalid book_id format. Expected UUID, got: {v}")
        return v.lower()

    @field_validator('chapter_number')
    @classmethod
    def validate_chapter_number(cls, v):
        """Ensure chapter number is positive"""
        if v < 1:
            raise ValueError("chapter_number must be >= 1")
        return v

class ChapterRangeInput(BaseModel):
    """Schema for chapter range operations

    Used by: list_sections tool
    """
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "book_id": "4e7e9727-d82d-49ca-a32c-990a6619cd75",
            "chapter_number": 6
        }
    })

    book_id: str = Field(
        ...,
        description="UUID of the book"
    )
    chapter_number: int = Field(
        ...,
        ge=1,
        description="Chapter number (1-indexed)"
    )

    @field_validator('book_id')
    @classmethod
    def validate_book_id(cls, v):
        """Ensure book_id is a valid UUID"""
        if not re.match(UUID_PATTERN, v.lower()):
            raise ValueError(f"Invalid book_id format. Expected UUID, got: {v}")
        return v.lower()

# Validation helper functions
def validate_limit(limit: int, min_val: int = 1, max_val: int = 20) -> int:
    """Validate and constrain limit parameter

    Args:
        limit: Limit value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Validated and constrained limit

    Raises:
        ValueError: If limit is invalid
    """
    if not isinstance(limit, int):
        raise ValueError(f"Limit must be an integer, got {type(limit)}")

    if limit < min_val:
        return min_val
    if limit > max_val:
        return max_val
    return limit

def validate_book_id(book_id: str) -> str:
    """Validate book ID format

    Args:
        book_id: Book ID to validate

    Returns:
        Validated book ID (lowercase)

    Raises:
        ValueError: If book_id is not a valid UUID
    """
    if not isinstance(book_id, str):
        raise ValueError(f"book_id must be a string, got {type(book_id)}")

    if not re.match(UUID_PATTERN, book_id.lower()):
        raise ValueError(f"Invalid book_id format. Expected UUID, got: {book_id}")

    return book_id.lower()

def validate_chapter_number(chapter_number: int) -> int:
    """Validate chapter number

    Args:
        chapter_number: Chapter number to validate

    Returns:
        Validated chapter number

    Raises:
        ValueError: If chapter_number is invalid
    """
    if not isinstance(chapter_number, int):
        raise ValueError(f"chapter_number must be an integer, got {type(chapter_number)}")

    if chapter_number < 1:
        raise ValueError("chapter_number must be >= 1")

    return chapter_number

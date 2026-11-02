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

# UUID validation pattern
UUID_PATTERN = r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$"


class SemanticSearchInput(BaseModel):
    """Schema for semantic search operations

    Used by: semantic_search tool
    """

    model_config = ConfigDict(
        json_schema_extra={"example": {"query": "container orchestration concepts", "limit": 5, "min_similarity": 0.4}}
    )

    query: str = Field(..., min_length=1, max_length=500, description="Semantic search query")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum number of results")
    min_similarity: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum similarity threshold (0.0-1.0)")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        """Ensure query is not just whitespace"""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()

    @field_validator("min_similarity")
    @classmethod
    def validate_similarity(cls, v):
        """Ensure similarity is in valid range"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("min_similarity must be between 0.0 and 1.0")
        return v


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

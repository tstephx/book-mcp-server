"""Schema validation package

Centralized input/output schemas following MCP best practice:
"Create a schema.ts file that contains the schemas for the tools"

This ensures:
- Consistent validation across all tools
- Single source of truth for data structures
- Better error messages
- Type safety
"""

from .tool_schemas import (
    SearchInput,
    SemanticSearchInput,
    BookIdInput,
    ChapterInput,
    ChapterRangeInput
)

from .response_schemas import (
    SearchResult,
    SearchResponse,
    BookInfo,
    ChapterInfo,
    ErrorResponse
)

__all__ = [
    # Input schemas
    'SearchInput',
    'SemanticSearchInput',
    'BookIdInput',
    'ChapterInput',
    'ChapterRangeInput',
    # Response schemas
    'SearchResult',
    'SearchResponse',
    'BookInfo',
    'ChapterInfo',
    'ErrorResponse'
]

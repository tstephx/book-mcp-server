"""
Input validation utilities
Follows MCP best practices for input validation
"""

import re
from typing import Optional

from ..database import execute_query

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

def validate_book_id(book_id: str) -> str:
    """
    Validate book ID format (UUID)
    
    Args:
        book_id: Book ID to validate
    
    Returns:
        Validated book ID
    
    Raises:
        ValidationError: If book ID is invalid
    """
    if not book_id:
        raise ValidationError("Book ID cannot be empty")
    
    # UUID format: 8-4-4-4-12 hex digits
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    if not re.match(uuid_pattern, book_id, re.IGNORECASE):
        raise ValidationError(f"Invalid book ID format: {book_id}")
    
    return book_id

def validate_chapter_number(chapter_number: int) -> int:
    """
    Validate chapter number
    
    Args:
        chapter_number: Chapter number to validate
    
    Returns:
        Validated chapter number
    
    Raises:
        ValidationError: If chapter number is invalid
    """
    if chapter_number < 1:
        raise ValidationError("Chapter number must be positive")
    
    if chapter_number > 1000:
        raise ValidationError("Chapter number too large (max: 1000)")
    
    return chapter_number

def validate_search_query(query: str) -> str:
    """
    Validate and sanitize search query
    
    Args:
        query: Search query to validate
    
    Returns:
        Validated search query
    
    Raises:
        ValidationError: If query is invalid
    """
    if not query or not query.strip():
        raise ValidationError("Search query cannot be empty")
    
    query = query.strip()
    
    if len(query) < 2:
        raise ValidationError("Search query must be at least 2 characters")
    
    if len(query) > 100:
        raise ValidationError("Search query too long (max: 100 characters)")
    
    return query

def validate_limit(limit: int, max_limit: Optional[int] = None) -> int:
    """
    Validate result limit
    
    Args:
        limit: Limit to validate
        max_limit: Maximum allowed limit (optional)
    
    Returns:
        Validated limit
    
    Raises:
        ValidationError: If limit is invalid
    """
    if limit < 1:
        raise ValidationError("Limit must be positive")
    
    if max_limit and limit > max_limit:
        raise ValidationError(f"Limit too large (max: {max_limit})")

    return limit


def resolve_book_id(book_id: str) -> str:
    """
    Resolve a book ID to a valid UUID.

    Accepts:
    - Full UUID (fast path, no DB lookup)
    - Slug-style string (hyphens converted to spaces, fuzzy title match)

    Returns the book's UUID. Raises ValidationError with a "did you mean?"
    message if no match found but candidates exist, or a plain error if no
    candidates exist at all.

    Args:
        book_id: UUID or slug-style book identifier

    Returns:
        Resolved UUID string

    Raises:
        ValidationError: If the book cannot be resolved
    """
    if not book_id:
        raise ValidationError("Book ID cannot be empty")

    # Fast path: already a valid UUID
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    if re.match(uuid_pattern, book_id, re.IGNORECASE):
        return book_id

    # Slug fallback: convert hyphens/underscores to spaces for title matching
    search_term = book_id.replace('-', ' ').replace('_', ' ').strip()

    # Try fuzzy LIKE match against titles
    matches = execute_query(
        "SELECT id, title FROM books WHERE LOWER(title) LIKE LOWER(?) ORDER BY title LIMIT 5",
        (f"%{search_term}%",)
    )

    if matches:
        return matches[0]['id']

    # No match — fetch candidates for did-you-mean using first word of slug
    first_word = search_term.split()[0] if search_term.split() else search_term
    candidates = execute_query(
        "SELECT id, title FROM books WHERE LOWER(title) LIKE LOWER(?) ORDER BY title LIMIT 3",
        (f"%{first_word}%",)
    )

    if candidates:
        suggestions = ", ".join(f'"{c["title"]}"' for c in candidates)
        raise ValidationError(
            f"Book not found for: '{book_id}'. Did you mean: {suggestions}? "
            f"Use search_titles() to find the correct book ID."
        )

    raise ValidationError(
        f"Book not found for: '{book_id}'. "
        f"Use search_titles() to find available books and their IDs."
    )

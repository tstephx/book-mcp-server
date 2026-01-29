"""
Input validation utilities
Follows MCP best practices for input validation
"""

import re
from typing import Optional

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

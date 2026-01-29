"""Response schemas for type safety and validation

Following MCP best practices:
- Structured response formats
- Type safety
- Documentation
- Validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class SearchResult(BaseModel):
    """Single search result from semantic or keyword search"""
    book_title: str = Field(..., description="Title of the book")
    chapter_title: str = Field(..., description="Chapter title")
    chapter_number: int = Field(..., description="Chapter number")
    similarity: Optional[float] = Field(None, description="Similarity score (semantic search only)")
    excerpt: str = Field(..., description="Text excerpt from chapter")
    
    class Config:
        schema_extra = {
            "example": {
                "book_title": "Learn Docker in a Month of Lunches",
                "chapter_title": "Container Networking",
                "chapter_number": 12,
                "similarity": 0.87,
                "excerpt": "Docker networking allows containers to communicate..."
            }
        }

class SearchResponse(BaseModel):
    """Response from search tools"""
    query: str = Field(..., description="Original search query")
    results: List[SearchResult] = Field(..., description="List of search results")
    total_found: int = Field(..., description="Total number of results found")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "docker networking",
                "results": [],
                "total_found": 5
            }
        }

class BookInfo(BaseModel):
    """Book information"""
    id: str = Field(..., description="Book UUID")
    title: str = Field(..., description="Book title")
    author: Optional[str] = Field(None, description="Book author")
    word_count: int = Field(..., description="Total word count")
    chapter_count: Optional[int] = Field(None, description="Number of chapters")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "4e7e9727-d82d-49ca-a32c-990a6619cd75",
                "title": "Learn Model Context Protocol",
                "author": "Dan Wahlin",
                "word_count": 58595,
                "chapter_count": 13
            }
        }

class ChapterInfo(BaseModel):
    """Chapter information"""
    chapter_number: int = Field(..., description="Chapter number")
    title: str = Field(..., description="Chapter title")
    word_count: int = Field(..., description="Word count")
    file_path: Optional[str] = Field(None, description="File path")
    has_sections: bool = Field(False, description="Whether chapter is split into sections")
    
    class Config:
        schema_extra = {
            "example": {
                "chapter_number": 6,
                "title": "Maintaining Clean Architecture",
                "word_count": 5555,
                "file_path": "/path/to/chapter.md",
                "has_sections": True
            }
        }

class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Book not found",
                "code": "BOOK_NOT_FOUND",
                "details": {"book_id": "invalid-uuid"}
            }
        }

class SuccessResponse(BaseModel):
    """Generic success response"""
    success: bool = Field(True, description="Operation success status")
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Operation completed successfully",
                "data": {}
            }
        }

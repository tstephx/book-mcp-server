"""
Search tools
Modular tool registration following MCP best practices
"""

from typing import TYPE_CHECKING

from ..config import Config
from ..database import execute_query, DatabaseError
from ..utils.validators import validate_search_query, validate_limit, ValidationError
from ..utils.logging import logger

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

def register_search_tools(mcp: "FastMCP") -> None:
    """
    Register all search-related tools
    
    Args:
        mcp: FastMCP server instance
    """
    
    @mcp.tool()
    def search_titles(query: str, limit: int = 10) -> str:
        """
        Search book titles, authors, and chapter titles (metadata only).

        For searching chapter *content*, use text_search or hybrid_search instead.

        Args:
            query: Search query text
            limit: Maximum number of results per category (default: 10, max: 50)
        """
        try:
            # Validate inputs
            query = validate_search_query(query)
            limit = validate_limit(limit, max_limit=Config.MAX_SEARCH_RESULTS)
            
            search_pattern = f"%{query}%"
            
            # Search in book titles and authors
            books = execute_query("""
                SELECT id, title, author 
                FROM books 
                WHERE title LIKE ? OR author LIKE ?
                ORDER BY title
                LIMIT ?
            """, (search_pattern, search_pattern, limit))
            
            # Search in chapter titles
            chapters = execute_query("""
                SELECT c.book_id, b.title as book_title, c.chapter_number, c.title as chapter_title
                FROM chapters c
                JOIN books b ON c.book_id = b.id
                WHERE c.title LIKE ?
                ORDER BY b.title, c.chapter_number
                LIMIT ?
            """, (search_pattern, limit))
            
            # Build result
            result = f"🔍 Search Results for '{query}'\n" + "="*50 + "\n\n"
            
            if books:
                result += "📚 Matching Books:\n" + "-"*50 + "\n"
                for book in books:
                    result += f"• {book['title']}"
                    if book['author']:
                        result += f" by {book['author']}"
                    result += f"\n  ID: {book['id']}\n"
            
            if chapters:
                result += "\n📄 Matching Chapters:\n" + "-"*50 + "\n"
                for chapter in chapters:
                    result += f"• {chapter['book_title']}\n"
                    result += f"  Chapter {chapter['chapter_number']}: {chapter['chapter_title']}\n"
                    result += f"  Book ID: {chapter['book_id']}\n"
            
            if not books and not chapters:
                result += "No results found.\n\n"
                result += "Try:\n"
                result += "• Using different keywords\n"
                result += "• Checking spelling\n"
                result += "• Using more general terms\n"
            else:
                result += f"\nFound {len(books)} books and {len(chapters)} chapters"
            
            logger.info(f"search_titles '{query}' returned {len(books)} books, {len(chapters)} chapters")
            return result
            
        except ValidationError as e:
            logger.warning(f"Validation error in search_titles: {e}")
            return f"Validation error: {str(e)}"
        except DatabaseError as e:
            logger.error(f"Database error in search_titles: {e}")
            return f"Error accessing database: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in search_titles: {e}")
            return f"Unexpected error: {str(e)}"

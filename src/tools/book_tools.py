"""
Book listing and information tools
Modular tool registration following MCP best practices
"""

from typing import TYPE_CHECKING

from ..database import execute_query, execute_single, DatabaseError
from ..utils.validators import validate_book_id, ValidationError
from ..utils.logging import logger

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

def register_book_tools(mcp: "FastMCP") -> None:
    """
    Register all book-related tools
    
    Args:
        mcp: FastMCP server instance
    """
    
    @mcp.tool()
    def list_books() -> str:
        """List all books in the library with their metadata"""
        try:
            books = execute_query("""
                SELECT id, title, author, word_count, book_type,
                       (SELECT COUNT(*) FROM chapters WHERE book_id = books.id) as chapter_count
                FROM books
                ORDER BY title
            """)
            
            if not books:
                return "No books found in library."
            
            result = "📚 Book Library\n" + "="*50 + "\n\n"
            
            for book in books:
                result += f"📖 {book['title']}\n"
                if book['author']:
                    result += f"   Author: {book['author']}\n"
                if book['book_type']:
                    result += f"   Type: {book['book_type']}\n"
                result += f"   Words: {book['word_count']:,}\n"
                result += f"   Chapters: {book['chapter_count']}\n"
                result += f"   ID: {book['id']}\n\n"
            
            logger.info(f"Listed {len(books)} books")
            return result
            
        except DatabaseError as e:
            logger.error(f"Database error in list_books: {e}")
            return f"Error accessing database: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in list_books: {e}")
            return f"Unexpected error: {str(e)}"
    
    @mcp.tool()
    def get_book_info(book_id: str) -> str:
        """
        Get detailed information about a specific book
        
        Args:
            book_id: The unique ID of the book
        """
        try:
            # Validate input
            book_id = validate_book_id(book_id)
            
            # Get book info
            book = execute_single("SELECT * FROM books WHERE id = ?", (book_id,))
            
            if not book:
                return f"Book with ID {book_id} not found."
            
            # Get chapters
            chapters = execute_query("""
                SELECT chapter_number, title, word_count 
                FROM chapters 
                WHERE book_id = ? 
                ORDER BY chapter_number
            """, (book_id,))
            
            result = f"📖 {book['title']}\n"
            result += "="*50 + "\n\n"

            if book['author']:
                result += f"Author: {book['author']}\n"
            result += f"Total Words: {book['word_count']:,}\n"
            result += f"Chapters: {len(chapters)}\n"
            result += f"Status: {book['processing_status']}\n"

            # Classification info
            book_type = book['book_type'] if 'book_type' in book.keys() else None
            if book_type:
                result += f"\nClassification:\n"
                result += f"  Type: {book_type}\n"
                confidence = book['classification_confidence'] if 'classification_confidence' in book.keys() else None
                if confidence is not None:
                    result += f"  Confidence: {confidence:.0%}\n"
                tags_raw = book['suggested_tags'] if 'suggested_tags' in book.keys() else None
                if tags_raw:
                    try:
                        import json
                        tags = json.loads(tags_raw)
                        if tags:
                            result += f"  Tags: {', '.join(tags)}\n"
                    except (json.JSONDecodeError, TypeError):
                        pass
                reasoning = book['classification_reasoning'] if 'classification_reasoning' in book.keys() else None
                if reasoning:
                    result += f"  Reasoning: {reasoning}\n"

            result += "\n"
            result += "Table of Contents:\n" + "-"*50 + "\n"
            for chapter in chapters:
                result += f"{chapter['chapter_number']:2d}. {chapter['title']} ({chapter['word_count']:,} words)\n"
            
            logger.info(f"Retrieved info for book: {book['title']}")
            return result
            
        except ValidationError as e:
            logger.warning(f"Validation error in get_book_info: {e}")
            return f"Validation error: {str(e)}"
        except DatabaseError as e:
            logger.error(f"Database error in get_book_info: {e}")
            return f"Error accessing database: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in get_book_info: {e}")
            return f"Unexpected error: {str(e)}"
    
    @mcp.tool()
    def get_table_of_contents(book_id: str) -> str:
        """
        Get the complete table of contents for a book
        
        Args:
            book_id: The unique ID of the book
        """
        try:
            # Validate input
            book_id = validate_book_id(book_id)
            
            # Get book
            book = execute_single("SELECT title, author FROM books WHERE id = ?", (book_id,))
            
            if not book:
                return f"Book with ID {book_id} not found."
            
            # Get chapters
            chapters = execute_query("""
                SELECT chapter_number, title, word_count 
                FROM chapters 
                WHERE book_id = ? 
                ORDER BY chapter_number
            """, (book_id,))
            
            result = f"📑 Table of Contents\n"
            result += f"{book['title']}\n"
            if book['author']:
                result += f"by {book['author']}\n"
            result += "="*50 + "\n\n"
            
            for chapter in chapters:
                result += f"{chapter['chapter_number']:2d}. {chapter['title']}\n"
                result += f"    ({chapter['word_count']:,} words)\n\n"
            
            logger.info(f"Retrieved TOC for: {book['title']}")
            return result
            
        except ValidationError as e:
            logger.warning(f"Validation error in get_table_of_contents: {e}")
            return f"Validation error: {str(e)}"
        except DatabaseError as e:
            logger.error(f"Database error in get_table_of_contents: {e}")
            return f"Error accessing database: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in get_table_of_contents: {e}")
            return f"Unexpected error: {str(e)}"

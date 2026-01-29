"""
Chapter reading tools
Modular tool registration following MCP best practices
"""

import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

from ..config import Config
from ..database import execute_single, DatabaseError
from ..utils.validators import validate_book_id, validate_chapter_number, ValidationError
from ..utils.logging import logger

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def _find_chapter_path(book_id: str, chapter_number: int, recorded_path: str) -> Tuple[Path, bool]:
    """
    Find the chapter file or folder path.

    Returns:
        Tuple of (path, is_folder) where is_folder indicates if chapter has sections
    """
    chapter_path = Path(recorded_path)

    # Check if recorded path exists (could be file or folder)
    if chapter_path.exists():
        return chapter_path, chapter_path.is_dir()

    # Try to find it in the books directory
    book_dir = Config.BOOKS_DIR / book_id / "chapters"

    if not book_dir.exists():
        raise FileNotFoundError(f"Book directory not found: {book_dir}")

    # Look for chapter file or folder by number prefix
    potential_matches = list(book_dir.glob(f"{chapter_number:02d}-*"))

    if not potential_matches:
        raise FileNotFoundError(f"Chapter {chapter_number} not found in {book_dir}")

    # Prefer folders (split chapters) over files if both exist
    for match in potential_matches:
        if match.is_dir():
            return match, True

    # Otherwise return the first .md file
    for match in potential_matches:
        if match.suffix == '.md':
            return match, False

    return potential_matches[0], potential_matches[0].is_dir()


def _read_file_content(file_path: Path, check_size: bool = True) -> str:
    """Read file content with optional size checking."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if check_size and len(content) > Config.MAX_CHAPTER_SIZE:
        raise ValueError(
            f"Content is too large ({len(content)} bytes). "
            f"Maximum allowed: {Config.MAX_CHAPTER_SIZE} bytes"
        )

    return content


def register_chapter_tools(mcp: "FastMCP") -> None:
    """
    Register all chapter-related tools

    Args:
        mcp: FastMCP server instance
    """

    @mcp.tool()
    def get_chapter(book_id: str, chapter_number: int) -> str:
        """
        Get the content of a specific chapter. For large chapters that have been
        split into sections, this returns an index listing all sections. Use
        get_section() to read individual sections.

        Args:
            book_id: The unique ID of the book
            chapter_number: The chapter number to retrieve (1-indexed)
        """
        try:
            # Validate inputs
            book_id = validate_book_id(book_id)
            chapter_number = validate_chapter_number(chapter_number)

            # Get chapter info from database
            chapter = execute_single("""
                SELECT title, file_path, word_count
                FROM chapters
                WHERE book_id = ? AND chapter_number = ?
            """, (book_id, chapter_number))

            if not chapter:
                return f"Chapter {chapter_number} not found for book ID: {book_id}"

            # Find the chapter path
            try:
                chapter_path, is_folder = _find_chapter_path(
                    book_id, chapter_number, chapter['file_path']
                )
            except FileNotFoundError as e:
                return f"Error: {str(e)}"

            # Handle folder-based chapters (split into sections)
            if is_folder:
                logger.info(f"Chapter {chapter_number} is split into sections: {chapter_path}")

                # Read the index file if it exists
                index_file = chapter_path / "_index.md"
                if index_file.exists():
                    content = _read_file_content(index_file, check_size=False)
                    return content

                # Otherwise, list the sections
                sections = sorted(chapter_path.glob("*.md"))
                sections = [s for s in sections if s.name != "_index.md"]

                lines = [
                    f"# {chapter['title']}",
                    "",
                    f"This chapter has been split into {len(sections)} sections for readability.",
                    "Use `get_section(book_id, chapter_number, section_number)` to read each section.",
                    "",
                    "## Sections",
                    ""
                ]

                for i, section in enumerate(sections, 1):
                    # Extract title from filename
                    title = section.stem
                    if title.startswith(f"{i:02d}-"):
                        title = title[3:]  # Remove number prefix
                    title = title.replace("-", " ").title()
                    lines.append(f"{i}. {title}")

                return "\n".join(lines)

            # Handle single-file chapters
            try:
                content = _read_file_content(chapter_path)
                logger.info(f"Retrieved chapter {chapter_number} from {book_id}")
                return content

            except ValueError as e:
                # File too large - check if there are sections we missed
                logger.warning(f"Chapter content exceeds size limit: {e}")
                return str(e)
            except UnicodeDecodeError:
                logger.error(f"Failed to decode chapter file: {chapter_path}")
                return f"Error: Unable to read chapter file (encoding issue)"
            except IOError as e:
                logger.error(f"IO error reading chapter file: {e}")
                return f"Error reading chapter file: {str(e)}"

        except ValidationError as e:
            logger.warning(f"Validation error in get_chapter: {e}")
            return f"Validation error: {str(e)}"
        except DatabaseError as e:
            logger.error(f"Database error in get_chapter: {e}")
            return f"Error accessing database: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in get_chapter: {e}")
            return f"Unexpected error: {str(e)}"

    @mcp.tool()
    def get_section(book_id: str, chapter_number: int, section_number: int) -> str:
        """
        Get a specific section from a chapter that has been split into multiple parts.
        Use get_chapter() first to see the list of available sections.

        Args:
            book_id: The unique ID of the book
            chapter_number: The chapter number (1-indexed)
            section_number: The section number within the chapter (1-indexed)
        """
        try:
            # Validate inputs
            book_id = validate_book_id(book_id)
            chapter_number = validate_chapter_number(chapter_number)
            if section_number < 1:
                return "Error: section_number must be >= 1"

            # Get chapter info from database
            chapter = execute_single("""
                SELECT title, file_path, word_count
                FROM chapters
                WHERE book_id = ? AND chapter_number = ?
            """, (book_id, chapter_number))

            if not chapter:
                return f"Chapter {chapter_number} not found for book ID: {book_id}"

            # Find the chapter path
            try:
                chapter_path, is_folder = _find_chapter_path(
                    book_id, chapter_number, chapter['file_path']
                )
            except FileNotFoundError as e:
                return f"Error: {str(e)}"

            # Must be a folder-based chapter
            if not is_folder:
                return (f"Chapter {chapter_number} is not split into sections. "
                       f"Use get_chapter() to read it directly.")

            # Find the section file
            section_files = sorted([
                f for f in chapter_path.glob("*.md")
                if f.name != "_index.md"
            ])

            if section_number > len(section_files):
                return (f"Section {section_number} not found. "
                       f"Chapter {chapter_number} has {len(section_files)} sections.")

            section_file = section_files[section_number - 1]

            try:
                content = _read_file_content(section_file)
                logger.info(f"Retrieved section {section_number} of chapter {chapter_number} from {book_id}")
                return content

            except ValueError as e:
                return str(e)
            except UnicodeDecodeError:
                logger.error(f"Failed to decode section file: {section_file}")
                return f"Error: Unable to read section file (encoding issue)"
            except IOError as e:
                logger.error(f"IO error reading section file: {e}")
                return f"Error reading section file: {str(e)}"

        except ValidationError as e:
            logger.warning(f"Validation error in get_section: {e}")
            return f"Validation error: {str(e)}"
        except DatabaseError as e:
            logger.error(f"Database error in get_section: {e}")
            return f"Error accessing database: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in get_section: {e}")
            return f"Unexpected error: {str(e)}"

    @mcp.tool()
    def list_sections(book_id: str, chapter_number: int) -> str:
        """
        List all sections in a chapter that has been split into multiple parts.

        Args:
            book_id: The unique ID of the book
            chapter_number: The chapter number (1-indexed)
        """
        try:
            # Validate inputs
            book_id = validate_book_id(book_id)
            chapter_number = validate_chapter_number(chapter_number)

            # Get chapter info from database
            chapter = execute_single("""
                SELECT title, file_path, word_count
                FROM chapters
                WHERE book_id = ? AND chapter_number = ?
            """, (book_id, chapter_number))

            if not chapter:
                return f"Chapter {chapter_number} not found for book ID: {book_id}"

            # Find the chapter path
            try:
                chapter_path, is_folder = _find_chapter_path(
                    book_id, chapter_number, chapter['file_path']
                )
            except FileNotFoundError as e:
                return f"Error: {str(e)}"

            if not is_folder:
                return (f"Chapter {chapter_number} is not split into sections. "
                       f"Use get_chapter() to read it directly.")

            # List section files
            section_files = sorted([
                f for f in chapter_path.glob("*.md")
                if f.name != "_index.md"
            ])

            lines = [
                f"# {chapter['title']}",
                "",
                f"**{len(section_files)} sections**",
                ""
            ]

            for i, section in enumerate(section_files, 1):
                # Try to get token count from file
                try:
                    content = section.read_text(encoding='utf-8')
                    # Look for token count in frontmatter
                    if content.startswith('---'):
                        token_match = re.search(r'tokens:\s*(\d+)', content[:500])
                        tokens = int(token_match.group(1)) if token_match else "?"
                    else:
                        tokens = len(content) // 4  # Estimate
                except:
                    tokens = "?"

                # Extract title from filename
                title = section.stem
                if re.match(r'^\d{2}-', title):
                    title = title[3:]
                title = title.replace("-", " ").title()

                lines.append(f"{i}. {title} (~{tokens} tokens)")

            return "\n".join(lines)

        except ValidationError as e:
            logger.warning(f"Validation error in list_sections: {e}")
            return f"Validation error: {str(e)}"
        except DatabaseError as e:
            logger.error(f"Database error in list_sections: {e}")
            return f"Error accessing database: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in list_sections: {e}")
            return f"Unexpected error: {str(e)}"

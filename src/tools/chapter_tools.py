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


def _estimate_tokens(text: str) -> int:
    """Estimate token count using ~4 characters per token heuristic."""
    return len(text) // 4


def _truncate_to_tokens(content: str, max_tokens: int) -> Tuple[str, bool, int]:
    """
    Truncate content to approximately max_tokens.

    Returns:
        Tuple of (truncated_content, was_truncated, total_estimated_tokens)
    """
    total_tokens = _estimate_tokens(content)

    if total_tokens <= max_tokens:
        return content, False, total_tokens

    # Truncate to approximate character count (4 chars per token)
    max_chars = max_tokens * 4

    # Find a good break point (end of paragraph or sentence)
    truncated = content[:max_chars]

    # Try to break at paragraph
    last_para = truncated.rfind('\n\n')
    if last_para > max_chars * 0.7:  # Only if we keep at least 70%
        truncated = truncated[:last_para]
    else:
        # Try to break at sentence
        last_sentence = max(
            truncated.rfind('. '),
            truncated.rfind('.\n'),
            truncated.rfind('? '),
            truncated.rfind('! ')
        )
        if last_sentence > max_chars * 0.8:  # Only if we keep at least 80%
            truncated = truncated[:last_sentence + 1]

    return truncated, True, total_tokens


# Threshold for auto-splitting (in tokens)
AUTO_SPLIT_THRESHOLD = 5000
# Target size for paragraph-based chunks (in tokens)
CHUNK_TARGET_TOKENS = 1500


def _extract_frontmatter(content: str) -> Tuple[str, str]:
    """
    Extract YAML frontmatter from content.

    Returns:
        Tuple of (frontmatter, remaining_content)
    """
    if content.startswith('---'):
        end_match = re.search(r'\n---\n', content[3:])
        if end_match:
            frontmatter_end = end_match.end() + 3
            return content[:frontmatter_end], content[frontmatter_end:]
    return '', content


def _split_by_headers(content: str) -> list[dict]:
    """
    Split content by markdown ## or ### headers only.

    Only uses actual markdown headers to avoid false positives
    from table content or other formatted text.

    Returns:
        List of sections with 'title', 'content', and 'tokens' keys
    """
    # Extract frontmatter first
    frontmatter, body = _extract_frontmatter(content)

    # Find ## or ### headers (h2 or h3)
    header_pattern = re.compile(r'^(#{2,3}) (.+)$', re.MULTILINE)
    matches = list(header_pattern.finditer(body))

    if not matches:
        return []

    sections = []

    # Content before first header (intro)
    if matches[0].start() > 0:
        intro_content = body[:matches[0].start()].strip()
        if intro_content and _estimate_tokens(intro_content) > 100:
            sections.append({
                'title': 'Introduction',
                'content': frontmatter + intro_content if frontmatter else intro_content,
                'tokens': _estimate_tokens(intro_content)
            })

    # Each header section
    for i, match in enumerate(matches):
        title = match.group(2).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        section_content = body[start:end].strip()

        sections.append({
            'title': title,
            'content': section_content,
            'tokens': _estimate_tokens(section_content)
        })

    return sections


def _split_by_paragraphs(content: str, target_tokens: int = CHUNK_TARGET_TOKENS) -> list[dict]:
    """
    Split content into chunks by paragraph boundaries.

    Returns:
        List of sections with 'title', 'content', and 'tokens' keys
    """
    # Extract frontmatter first
    frontmatter, body = _extract_frontmatter(content)

    # Split into paragraphs
    paragraphs = re.split(r'\n\n+', body)

    sections = []
    current_chunk = []
    current_tokens = 0
    chunk_num = 1

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_tokens = _estimate_tokens(para)

        # If adding this paragraph exceeds target, save current chunk
        if current_tokens > 0 and current_tokens + para_tokens > target_tokens:
            chunk_content = '\n\n'.join(current_chunk)
            # Add frontmatter to first chunk only
            if chunk_num == 1 and frontmatter:
                chunk_content = frontmatter + chunk_content

            sections.append({
                'title': f'Part {chunk_num}',
                'content': chunk_content,
                'tokens': current_tokens
            })
            current_chunk = [para]
            current_tokens = para_tokens
            chunk_num += 1
        else:
            current_chunk.append(para)
            current_tokens += para_tokens

    # Don't forget the last chunk
    if current_chunk:
        chunk_content = '\n\n'.join(current_chunk)
        if chunk_num == 1 and frontmatter:
            chunk_content = frontmatter + chunk_content
        sections.append({
            'title': f'Part {chunk_num}',
            'content': chunk_content,
            'tokens': current_tokens
        })

    return sections


def _auto_split_chapter(content: str) -> list[dict]:
    """
    Automatically split a large chapter into sections.

    Tries header-based splitting first, falls back to paragraph chunks.
    Only uses header-based if sections are well-balanced.

    Returns:
        List of sections with 'title', 'content', and 'tokens' keys
    """
    total_tokens = _estimate_tokens(content)

    # Try header-based splitting first
    sections = _split_by_headers(content)

    # Validate header-based sections:
    # - Need at least 3 sections
    # - No single section should be > 40% of total content
    # - Average section size should be reasonable
    if len(sections) >= 3:
        max_section_tokens = max(s['tokens'] for s in sections)
        if max_section_tokens < total_tokens * 0.4:
            logger.info(f"Auto-split chapter into {len(sections)} header-based sections")
            return sections

    # Fall back to paragraph-based chunking
    sections = _split_by_paragraphs(content)
    logger.info(f"Auto-split chapter into {len(sections)} paragraph-based chunks")
    return sections


def _format_section_index(chapter_title: str, sections: list[dict], book_id: str, chapter_number: int) -> str:
    """Format a section index for display."""
    total_tokens = sum(s['tokens'] for s in sections)

    lines = [
        f"# {chapter_title}",
        "",
        f"**{len(sections)} sections** | ~{total_tokens:,} total tokens",
        "",
        "This chapter has been auto-split for easier reading.",
        f"Use `get_section('{book_id}', {chapter_number}, section_number)` to read each section.",
        "",
        "## Sections",
        ""
    ]

    for i, section in enumerate(sections, 1):
        lines.append(f"{i}. **{section['title']}** (~{section['tokens']:,} tokens)")

    return "\n".join(lines)


# Cache for auto-split sections (book_id, chapter_number) -> sections list
_auto_split_cache: dict[Tuple[str, int], list[dict]] = {}


def register_chapter_tools(mcp: "FastMCP") -> None:
    """
    Register all chapter-related tools

    Args:
        mcp: FastMCP server instance
    """

    @mcp.tool()
    def get_chapter(book_id: str, chapter_number: int, max_tokens: Optional[int] = None) -> str:
        """
        Get the content of a specific chapter. For large chapters that have been
        split into sections, this returns an index listing all sections. Use
        get_section() to read individual sections.

        Args:
            book_id: The unique ID of the book
            chapter_number: The chapter number to retrieve (1-indexed)
            max_tokens: Optional maximum tokens to return. If content exceeds this,
                       it will be truncated with a note. Uses ~4 chars/token estimate.
        """
        try:
            # Validate inputs
            book_id = validate_book_id(book_id)
            chapter_number = validate_chapter_number(chapter_number)

            # Apply server-side default if caller did not specify
            if max_tokens is None:

                max_tokens = Config.DEFAULT_CHAPTER_TOKENS

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

                    # Apply token truncation if requested
                    if max_tokens is not None:
                        content, was_truncated, total_tokens = _truncate_to_tokens(content, max_tokens)
                        if was_truncated:
                            content += (
                                f"\n\n---\n"
                                f"⚠️ **Content truncated** to ~{max_tokens:,} tokens "
                                f"(of ~{total_tokens:,} total). "
                                f"Use `get_section()` to read individual sections."
                            )

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
                content = _read_file_content(chapter_path, check_size=False)
                total_tokens = _estimate_tokens(content)

                # Auto-split large chapters
                if total_tokens >= AUTO_SPLIT_THRESHOLD:
                    logger.info(
                        f"Chapter {chapter_number} has {total_tokens} tokens, "
                        f"auto-splitting (threshold: {AUTO_SPLIT_THRESHOLD})"
                    )
                    sections = _auto_split_chapter(content)

                    # Cache the sections for get_section() to use
                    _auto_split_cache[(book_id, chapter_number)] = sections

                    # Return section index
                    return _format_section_index(
                        chapter['title'], sections, book_id, chapter_number
                    )

                # Small chapter - apply token truncation if requested
                if max_tokens is not None:
                    content, was_truncated, total_tokens = _truncate_to_tokens(content, max_tokens)
                    if was_truncated:
                        truncation_note = (
                            f"\n\n---\n"
                            f"⚠️ **Content truncated** to ~{max_tokens:,} tokens "
                            f"(of ~{total_tokens:,} total). "
                            f"Use `get_section()` for large chapters or increase `max_tokens`."
                        )
                        content += truncation_note
                        logger.info(
                            f"Truncated chapter {chapter_number} from {book_id}: "
                            f"{total_tokens} -> {max_tokens} tokens"
                        )

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
    def get_section(book_id: str, chapter_number: int, section_number: int, max_tokens: Optional[int] = None) -> str:
        """
        Get a specific section from a chapter that has been split into multiple parts.
        Use get_chapter() first to see the list of available sections.

        Args:
            book_id: The unique ID of the book
            chapter_number: The chapter number (1-indexed)
            section_number: The section number within the chapter (1-indexed)
            max_tokens: Optional maximum tokens to return. If content exceeds this,
                       it will be truncated with a note.
        """
        try:
            # Validate inputs
            book_id = validate_book_id(book_id)
            chapter_number = validate_chapter_number(chapter_number)

            # Apply server-side default if caller did not specify
            if max_tokens is None:

                max_tokens = Config.DEFAULT_CHAPTER_TOKENS
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

            # Check for auto-split sections first
            cache_key = (book_id, chapter_number)
            if not is_folder and cache_key in _auto_split_cache:
                sections = _auto_split_cache[cache_key]
                if section_number > len(sections):
                    return (f"Section {section_number} not found. "
                           f"Chapter {chapter_number} has {len(sections)} auto-split sections.")

                section = sections[section_number - 1]
                content = section['content']

                # Apply token truncation if requested
                if max_tokens is not None:
                    content, was_truncated, total_tokens = _truncate_to_tokens(content, max_tokens)
                    if was_truncated:
                        content += (
                            f"\n\n---\n"
                            f"⚠️ **Content truncated** to ~{max_tokens:,} tokens "
                            f"(of ~{total_tokens:,} total). Increase `max_tokens` for more."
                        )

                logger.info(f"Retrieved auto-split section {section_number} of chapter {chapter_number}")
                return content

            # Check if it's a single-file chapter that needs auto-splitting
            if not is_folder:
                # Try to auto-split it now
                try:
                    content = _read_file_content(chapter_path, check_size=False)
                    total_tokens = _estimate_tokens(content)

                    if total_tokens >= AUTO_SPLIT_THRESHOLD:
                        sections = _auto_split_chapter(content)
                        _auto_split_cache[cache_key] = sections

                        if section_number > len(sections):
                            return (f"Section {section_number} not found. "
                                   f"Chapter {chapter_number} has {len(sections)} auto-split sections.")

                        section = sections[section_number - 1]
                        section_content = section['content']

                        if max_tokens is not None:
                            section_content, was_truncated, _ = _truncate_to_tokens(section_content, max_tokens)
                            if was_truncated:
                                section_content += (
                                    f"\n\n---\n"
                                    f"⚠️ **Content truncated** to ~{max_tokens:,} tokens."
                                )

                        return section_content
                    else:
                        return (f"Chapter {chapter_number} is not split into sections "
                               f"(only {total_tokens} tokens). Use get_chapter() to read it directly.")
                except Exception as e:
                    logger.error(f"Error auto-splitting chapter: {e}")
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

                # Apply token truncation if requested
                if max_tokens is not None:
                    content, was_truncated, total_tokens = _truncate_to_tokens(content, max_tokens)
                    if was_truncated:
                        content += (
                            f"\n\n---\n"
                            f"⚠️ **Content truncated** to ~{max_tokens:,} tokens "
                            f"(of ~{total_tokens:,} total). Increase `max_tokens` for more."
                        )
                        logger.info(
                            f"Truncated section {section_number} of chapter {chapter_number}: "
                            f"{total_tokens} -> {max_tokens} tokens"
                        )

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

            # Check for auto-split sections first
            cache_key = (book_id, chapter_number)
            if not is_folder and cache_key in _auto_split_cache:
                sections = _auto_split_cache[cache_key]
                total_tokens = sum(s['tokens'] for s in sections)

                lines = [
                    f"# {chapter['title']}",
                    "",
                    f"**{len(sections)} auto-split sections** | ~{total_tokens:,} total tokens",
                    ""
                ]

                for i, section in enumerate(sections, 1):
                    lines.append(f"{i}. **{section['title']}** (~{section['tokens']:,} tokens)")

                return "\n".join(lines)

            # Check if single-file chapter needs auto-splitting
            if not is_folder:
                try:
                    content = _read_file_content(chapter_path, check_size=False)
                    total_tokens = _estimate_tokens(content)

                    if total_tokens >= AUTO_SPLIT_THRESHOLD:
                        sections = _auto_split_chapter(content)
                        _auto_split_cache[cache_key] = sections

                        lines = [
                            f"# {chapter['title']}",
                            "",
                            f"**{len(sections)} auto-split sections** | ~{total_tokens:,} total tokens",
                            ""
                        ]

                        for i, section in enumerate(sections, 1):
                            lines.append(f"{i}. **{section['title']}** (~{section['tokens']:,} tokens)")

                        return "\n".join(lines)
                    else:
                        return (f"Chapter {chapter_number} is not split into sections "
                               f"(only {total_tokens} tokens). Use get_chapter() to read it directly.")
                except Exception as e:
                    logger.error(f"Error checking chapter for auto-split: {e}")
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
                except Exception:
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

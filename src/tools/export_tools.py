"""
Export tools for generating clean outputs and study materials
Enables creating markdown exports, study guides, and flashcards from chapters
"""

import logging
import re
from typing import TYPE_CHECKING, Optional
from datetime import datetime

from ..database import execute_single, execute_query
from ..utils.file_utils import read_chapter_content

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


def _clean_markdown(content: str) -> str:
    """Clean up markdown content for export

    - Removes YAML frontmatter
    - Normalizes headers
    - Cleans up excessive whitespace
    - Ensures consistent formatting
    """
    lines = content.split('\n')
    result_lines = []
    in_frontmatter = False
    frontmatter_count = 0

    for line in lines:
        # Handle YAML frontmatter
        if line.strip() == '---':
            frontmatter_count += 1
            if frontmatter_count == 1:
                in_frontmatter = True
                continue
            elif frontmatter_count == 2:
                in_frontmatter = False
                continue

        if in_frontmatter:
            continue

        result_lines.append(line)

    content = '\n'.join(result_lines)

    # Normalize multiple blank lines to max 2
    content = re.sub(r'\n{4,}', '\n\n\n', content)

    # Strip leading/trailing whitespace
    content = content.strip()

    return content


def _extract_key_concepts(content: str) -> list[dict]:
    """Extract key concepts from chapter content

    Looks for:
    - Definitions (terms followed by explanations)
    - Bold/emphasized terms
    - Section headers as topics
    - Code examples
    """
    concepts = []

    # Extract headers as topics
    headers = re.findall(r'^#{1,3}\s+(.+)$', content, re.MULTILINE)
    for header in headers[:10]:  # Limit to first 10
        if len(header) > 5 and not header.lower().startswith(('chapter', 'section', 'part')):
            concepts.append({
                'type': 'topic',
                'term': header.strip(),
                'context': f"Section covering: {header.strip()}"
            })

    # Extract bold terms as key vocabulary
    bold_terms = re.findall(r'\*\*([^*]+)\*\*', content)
    seen = set()
    for term in bold_terms:
        term = term.strip()
        if len(term) > 2 and len(term) < 50 and term.lower() not in seen:
            seen.add(term.lower())
            # Try to find context around the term
            pattern = re.escape(f"**{term}**")
            match = re.search(f'.{{0,100}}{pattern}.{{0,150}}', content, re.DOTALL)
            context = match.group(0) if match else ""
            context = re.sub(r'\s+', ' ', context).strip()

            concepts.append({
                'type': 'term',
                'term': term,
                'context': context[:200] if context else f"Key term: {term}"
            })

    # Extract code blocks as examples
    code_blocks = re.findall(r'```(\w*)\n(.*?)```', content, re.DOTALL)
    for i, (lang, code) in enumerate(code_blocks[:5]):  # Limit to first 5
        if len(code.strip()) > 20:
            concepts.append({
                'type': 'code_example',
                'term': f"Code Example {i+1}" + (f" ({lang})" if lang else ""),
                'context': code.strip()[:300]
            })

    return concepts


def _generate_flashcards(concepts: list[dict], chapter_title: str) -> list[dict]:
    """Generate flashcard-style Q&A pairs from concepts"""
    flashcards = []

    for concept in concepts:
        if concept['type'] == 'topic':
            flashcards.append({
                'question': f"What are the key points about {concept['term']}?",
                'answer': concept['context'],
                'category': 'topic'
            })
        elif concept['type'] == 'term':
            flashcards.append({
                'question': f"What is {concept['term']}?",
                'answer': concept['context'],
                'category': 'vocabulary'
            })
        elif concept['type'] == 'code_example':
            flashcards.append({
                'question': f"Explain this code pattern from {chapter_title}:",
                'answer': concept['context'],
                'category': 'code'
            })

    return flashcards


def register_export_tools(mcp: "FastMCP") -> None:
    """Register export tools with the MCP server"""

    @mcp.tool()
    def export_chapter_to_markdown(
        book_id: str,
        chapter_number: int,
        include_metadata: bool = True,
        clean_formatting: bool = True
    ) -> dict:
        """Export a chapter as clean, formatted markdown

        Creates a polished markdown export suitable for:
        - Reading in markdown viewers
        - Sharing or printing
        - Further processing
        - Note-taking applications

        Args:
            book_id: UUID of the book
            chapter_number: Chapter number to export
            include_metadata: Include title, book info, word count header (default: True)
            clean_formatting: Remove frontmatter and normalize whitespace (default: True)

        Returns:
            Dictionary with exported markdown content and metadata

        Examples:
            export_chapter_to_markdown("abc-123", 5)
            export_chapter_to_markdown("abc-123", 5, include_metadata=False)
        """
        try:
            # Get chapter info
            chapter = execute_single("""
                SELECT c.id, c.title, c.word_count, c.file_path, c.chapter_number,
                       b.title as book_title, b.author
                FROM chapters c
                JOIN books b ON c.book_id = b.id
                WHERE c.book_id = ? AND c.chapter_number = ?
            """, (book_id, chapter_number))

            if not chapter:
                return {
                    "error": f"Chapter {chapter_number} not found in book {book_id}",
                    "content": ""
                }

            # Read chapter content
            try:
                content = read_chapter_content(chapter['file_path'])
            except Exception as e:
                return {"error": f"Could not read chapter: {e}", "content": ""}

            # Clean formatting if requested
            if clean_formatting:
                content = _clean_markdown(content)

            # Build export
            export_parts = []

            if include_metadata:
                export_parts.append(f"# {chapter['title']}")
                export_parts.append("")
                export_parts.append(f"> From **{chapter['book_title']}**" +
                                  (f" by {chapter['author']}" if chapter['author'] else ""))
                export_parts.append(f"> Chapter {chapter['chapter_number']} | " +
                                  f"~{chapter['word_count'] or 0:,} words | " +
                                  f"Exported {datetime.now().strftime('%Y-%m-%d')}")
                export_parts.append("")
                export_parts.append("---")
                export_parts.append("")

            export_parts.append(content)

            exported_content = '\n'.join(export_parts)

            logger.info(f"Exported chapter {chapter_number} from {chapter['book_title']}")

            return {
                "book_title": chapter['book_title'],
                "chapter_title": chapter['title'],
                "chapter_number": chapter_number,
                "word_count": chapter['word_count'] or 0,
                "content": exported_content,
                "character_count": len(exported_content)
            }

        except Exception as e:
            logger.error(f"export_chapter_to_markdown error: {e}", exc_info=True)
            return {"error": str(e), "content": ""}

    @mcp.tool()
    def create_study_guide(
        book_id: str,
        chapter_number: int,
        format: str = "comprehensive"
    ) -> dict:
        """Generate a study guide with flashcards and summary from a chapter

        Creates study materials including:
        - Chapter summary
        - Key concepts and vocabulary
        - Flashcard-style Q&A pairs
        - Code examples (if present)
        - Study tips

        Args:
            book_id: UUID of the book
            chapter_number: Chapter number to create study guide for
            format: Guide format - "comprehensive" (full guide), "flashcards" (Q&A only),
                   or "summary" (brief overview)

        Returns:
            Dictionary with study guide content, flashcards, and key concepts

        Examples:
            create_study_guide("abc-123", 5)
            create_study_guide("abc-123", 5, format="flashcards")
        """
        try:
            # Validate format
            valid_formats = ['comprehensive', 'flashcards', 'summary']
            if format not in valid_formats:
                return {
                    "error": f"Invalid format. Choose from: {', '.join(valid_formats)}",
                    "content": ""
                }

            # Get chapter info
            chapter = execute_single("""
                SELECT c.id, c.title, c.word_count, c.file_path, c.chapter_number,
                       b.title as book_title, b.author, b.id as book_id
                FROM chapters c
                JOIN books b ON c.book_id = b.id
                WHERE c.book_id = ? AND c.chapter_number = ?
            """, (book_id, chapter_number))

            if not chapter:
                return {
                    "error": f"Chapter {chapter_number} not found in book {book_id}",
                    "content": ""
                }

            # Read and clean content
            try:
                raw_content = read_chapter_content(chapter['file_path'])
                content = _clean_markdown(raw_content)
            except Exception as e:
                return {"error": f"Could not read chapter: {e}", "content": ""}

            # Extract key concepts
            concepts = _extract_key_concepts(content)

            # Generate flashcards
            flashcards = _generate_flashcards(concepts, chapter['title'])

            # Get adjacent chapters for context
            adjacent = execute_query("""
                SELECT chapter_number, title
                FROM chapters
                WHERE book_id = ? AND chapter_number IN (?, ?)
                ORDER BY chapter_number
            """, (book_id, chapter_number - 1, chapter_number + 1))

            prev_chapter = next((c for c in adjacent if c['chapter_number'] == chapter_number - 1), None)
            next_chapter = next((c for c in adjacent if c['chapter_number'] == chapter_number + 1), None)

            # Build study guide content based on format
            guide_parts = []

            if format in ['comprehensive', 'summary']:
                guide_parts.append(f"# Study Guide: {chapter['title']}")
                guide_parts.append("")
                guide_parts.append(f"**Book**: {chapter['book_title']}" +
                                 (f" by {chapter['author']}" if chapter['author'] else ""))
                guide_parts.append(f"**Chapter**: {chapter_number}")
                guide_parts.append(f"**Reading Time**: ~{(chapter['word_count'] or 0) // 200} minutes")
                guide_parts.append("")

                # Navigation context
                if prev_chapter or next_chapter:
                    guide_parts.append("## Chapter Context")
                    if prev_chapter:
                        guide_parts.append(f"- **Previous**: Ch. {prev_chapter['chapter_number']} - {prev_chapter['title']}")
                    guide_parts.append(f"- **Current**: Ch. {chapter_number} - {chapter['title']}")
                    if next_chapter:
                        guide_parts.append(f"- **Next**: Ch. {next_chapter['chapter_number']} - {next_chapter['title']}")
                    guide_parts.append("")

            if format == 'comprehensive':
                # Key Topics section
                topic_concepts = [c for c in concepts if c['type'] == 'topic']
                if topic_concepts:
                    guide_parts.append("## Key Topics Covered")
                    for concept in topic_concepts[:8]:
                        guide_parts.append(f"- {concept['term']}")
                    guide_parts.append("")

                # Key Vocabulary section
                term_concepts = [c for c in concepts if c['type'] == 'term']
                if term_concepts:
                    guide_parts.append("## Key Vocabulary")
                    for concept in term_concepts[:10]:
                        guide_parts.append(f"- **{concept['term']}**: {concept['context'][:100]}...")
                    guide_parts.append("")

                # Code Examples section
                code_concepts = [c for c in concepts if c['type'] == 'code_example']
                if code_concepts:
                    guide_parts.append("## Code Examples")
                    for concept in code_concepts:
                        guide_parts.append(f"### {concept['term']}")
                        guide_parts.append("```")
                        guide_parts.append(concept['context'])
                        guide_parts.append("```")
                        guide_parts.append("")

            if format in ['comprehensive', 'flashcards']:
                guide_parts.append("## Flashcards")
                guide_parts.append("")

                for i, card in enumerate(flashcards, 1):
                    guide_parts.append(f"### Card {i} ({card['category']})")
                    guide_parts.append(f"**Q**: {card['question']}")
                    guide_parts.append("")
                    if card['category'] == 'code':
                        guide_parts.append("```")
                        guide_parts.append(card['answer'])
                        guide_parts.append("```")
                    else:
                        guide_parts.append(f"**A**: {card['answer']}")
                    guide_parts.append("")

            if format in ['comprehensive', 'summary']:
                guide_parts.append("## Study Tips")
                guide_parts.append("")
                guide_parts.append(f"1. Read through the chapter once for overview")
                guide_parts.append(f"2. Review the {len(term_concepts) if format == 'comprehensive' else 'key'} vocabulary terms")
                guide_parts.append(f"3. Practice with the {len(flashcards)} flashcards above")
                if code_concepts if format == 'comprehensive' else False:
                    guide_parts.append(f"4. Try running the code examples yourself")
                guide_parts.append(f"5. Connect concepts to {next_chapter['title'] if next_chapter else 'related topics'}")
                guide_parts.append("")
                guide_parts.append("---")
                guide_parts.append(f"*Generated from {chapter['book_title']} on {datetime.now().strftime('%Y-%m-%d')}*")

            guide_content = '\n'.join(guide_parts)

            logger.info(f"Created {format} study guide for {chapter['book_title']} Ch.{chapter_number}")

            return {
                "book_title": chapter['book_title'],
                "chapter_title": chapter['title'],
                "chapter_number": chapter_number,
                "format": format,
                "content": guide_content,
                "statistics": {
                    "topics_found": len([c for c in concepts if c['type'] == 'topic']),
                    "vocabulary_terms": len([c for c in concepts if c['type'] == 'term']),
                    "code_examples": len([c for c in concepts if c['type'] == 'code_example']),
                    "flashcards_generated": len(flashcards)
                },
                "flashcards": flashcards,
                "key_concepts": concepts
            }

        except Exception as e:
            logger.error(f"create_study_guide error: {e}", exc_info=True)
            return {"error": str(e), "content": ""}

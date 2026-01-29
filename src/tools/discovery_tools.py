"""
Discovery tools for cross-book content exploration
Enables finding related content and topic coverage across the library
"""

import logging
from typing import TYPE_CHECKING, Optional
import numpy as np
import io
import re

from ..database import get_db_connection, execute_query, execute_single
from ..utils.vector_store import find_top_k, cosine_similarity
from ..utils.context_managers import embedding_model_context
from ..utils.file_utils import read_chapter_content, get_chapter_excerpt
from ..utils.cache import get_cache

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


def _get_context(content: str, position: int, max_chars: int = 200) -> str:
    """Extract context text before a given position in content"""
    if position <= 0:
        return ""

    context_start = max(0, position - max_chars)
    context = content[context_start:position].strip()

    # Try to start at a sentence boundary
    last_period = context.rfind('.')
    if last_period > 0:
        context = context[last_period + 1:].strip()

    return context


def _load_embeddings_cached() -> tuple[np.ndarray, list[dict]] | None:
    """Load embeddings matrix with caching

    Returns cached embeddings if available, otherwise loads from database
    and caches for subsequent calls.

    Returns:
        Tuple of (embeddings_matrix, chapter_metadata) or None if no embeddings
    """
    cache = get_cache()
    cached = cache.get_embeddings()

    if cached:
        return cached

    # Load from database
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                c.id, c.book_id, c.chapter_number, c.title as chapter_title,
                c.embedding, c.file_path, c.word_count,
                b.title as book_title, b.author
            FROM chapters c
            JOIN books b ON c.book_id = b.id
            WHERE c.embedding IS NOT NULL
            ORDER BY c.id
        """)
        rows = cursor.fetchall()

    if not rows:
        return None

    # Convert embeddings from BLOB to numpy arrays
    chapter_embeddings = []
    chapter_metadata = []

    for row in rows:
        embedding = np.load(io.BytesIO(row['embedding']))
        chapter_embeddings.append(embedding)
        chapter_metadata.append({
            'id': row['id'],
            'book_id': row['book_id'],
            'book_title': row['book_title'],
            'author': row['author'],
            'chapter_title': row['chapter_title'],
            'chapter_number': row['chapter_number'],
            'word_count': row['word_count'],
            'file_path': row['file_path']
        })

    embeddings_matrix = np.vstack(chapter_embeddings)

    # Store in cache
    cache.set_embeddings(embeddings_matrix, chapter_metadata)

    return embeddings_matrix, chapter_metadata


def register_discovery_tools(mcp: "FastMCP") -> None:
    """Register discovery tools with the MCP server"""

    @mcp.tool()
    def find_related_content(
        chapter_id: str = "",
        text_snippet: str = "",
        limit: int = 5,
        exclude_same_book: bool = True
    ) -> dict:
        """Find related content across all books based on a chapter or text snippet

        Discovers conceptually similar content from OTHER books, enabling
        cross-referencing and finding alternative explanations.

        Args:
            chapter_id: ID of a chapter to find related content for
            text_snippet: OR provide a text snippet to find related content
            limit: Maximum results to return (1-20, default: 5)
            exclude_same_book: Exclude chapters from the same book (default: True)

        Returns:
            Dictionary with related chapters from other books, ranked by similarity

        Examples:
            find_related_content(chapter_id="abc-123")
            find_related_content(text_snippet="container networking fundamentals")
        """
        try:
            if not chapter_id and not text_snippet:
                return {
                    "error": "Provide either chapter_id or text_snippet",
                    "results": []
                }

            # Validate limit
            limit = max(1, min(20, limit))
            source_book_id = None

            # Get query embedding
            with embedding_model_context() as generator:
                if chapter_id:
                    # Get the chapter's embedding from database
                    chapter = execute_single("""
                        SELECT c.id, c.book_id, c.chapter_number, c.title, c.embedding,
                               b.title as book_title
                        FROM chapters c
                        JOIN books b ON c.book_id = b.id
                        WHERE c.id = ?
                    """, (chapter_id,))

                    if not chapter:
                        return {"error": f"Chapter not found: {chapter_id}", "results": []}

                    if not chapter['embedding']:
                        return {"error": "Chapter has no embedding", "results": []}

                    query_embedding = np.load(io.BytesIO(chapter['embedding']))
                    source_book_id = chapter['book_id']
                    source_info = f"Chapter {chapter['chapter_number']}: {chapter['title']} from '{chapter['book_title']}'"
                else:
                    # Generate embedding for the text snippet
                    query_embedding = generator.generate(text_snippet)
                    source_info = f"Text: \"{text_snippet[:50]}...\""

            # Load embeddings from cache or database
            cached_data = _load_embeddings_cached()
            if not cached_data:
                return {"message": "No embeddings found in database", "results": []}

            embeddings_matrix, chapter_metadata = cached_data

            # Filter out same book if requested
            if exclude_same_book and source_book_id:
                indices = [i for i, m in enumerate(chapter_metadata) if m['book_id'] != source_book_id]
                if not indices:
                    return {"message": "No other books found", "results": []}
                embeddings_matrix = embeddings_matrix[indices]
                chapter_metadata = [chapter_metadata[i] for i in indices]

            # Find top K most similar
            top_results = find_top_k(
                query_embedding,
                embeddings_matrix,
                k=limit,
                min_similarity=0.2
            )

            # Build response
            results = []
            for idx, similarity in top_results:
                metadata = chapter_metadata[idx]
                results.append({
                    'book_title': metadata['book_title'],
                    'author': metadata['author'],
                    'chapter_title': metadata['chapter_title'],
                    'chapter_number': metadata['chapter_number'],
                    'similarity': round(similarity, 3),
                    'word_count': metadata['word_count'],
                    'chapter_id': metadata['id'],
                    'book_id': metadata['book_id']
                })

            logger.info(f"Found {len(results)} related chapters for: {source_info}")

            return {
                "source": source_info,
                "results": results,
                "total_found": len(results)
            }

        except Exception as e:
            logger.error(f"find_related_content error: {e}", exc_info=True)
            return {"error": str(e), "results": []}

    @mcp.tool()
    def get_topic_coverage(
        topic: str,
        min_similarity: float = 0.3,
        include_excerpts: bool = True
    ) -> dict:
        """Get comprehensive coverage of a topic across all books

        Returns every chapter that covers the topic, ranked by relevance/depth.
        Useful for understanding how well a topic is covered in your library.

        Args:
            topic: Topic to search for (e.g., "async programming", "docker networking")
            min_similarity: Minimum similarity threshold (0.0-1.0, default: 0.3)
            include_excerpts: Include text excerpts showing coverage (default: True)

        Returns:
            Dictionary with all chapters covering the topic, grouped by book

        Examples:
            get_topic_coverage("dependency injection")
            get_topic_coverage("kubernetes orchestration", min_similarity=0.4)
        """
        try:
            if not topic or not topic.strip():
                return {"error": "Topic cannot be empty", "results": []}

            topic = topic.strip()
            min_similarity = max(0.0, min(1.0, min_similarity))

            # Generate query embedding
            with embedding_model_context() as generator:
                query_embedding = generator.generate(topic)

            # Load embeddings from cache or database
            cached_data = _load_embeddings_cached()
            if not cached_data:
                return {"message": "No embeddings found in database", "results": []}

            embeddings_matrix, chapter_metadata = cached_data

            # Calculate similarities for all chapters
            all_results = []
            for i, metadata in enumerate(chapter_metadata):
                similarity = cosine_similarity(query_embedding, embeddings_matrix[i])

                if similarity >= min_similarity:
                    result = {
                        'book_id': metadata['book_id'],
                        'book_title': metadata['book_title'],
                        'author': metadata['author'],
                        'chapter_id': metadata['id'],
                        'chapter_title': metadata['chapter_title'],
                        'chapter_number': metadata['chapter_number'],
                        'similarity': round(float(similarity), 3),
                        'word_count': metadata['word_count']
                    }

                    # Add excerpt if requested
                    if include_excerpts and metadata['file_path']:
                        result['excerpt'] = get_chapter_excerpt(metadata['file_path'], max_chars=200)

                    all_results.append(result)

            # Sort by similarity (highest first)
            all_results.sort(key=lambda x: x['similarity'], reverse=True)

            # Group by book
            books_coverage = {}
            for result in all_results:
                book_id = result['book_id']
                if book_id not in books_coverage:
                    books_coverage[book_id] = {
                        'book_title': result['book_title'],
                        'author': result['author'],
                        'chapters': [],
                        'total_words': 0,
                        'avg_similarity': 0
                    }
                books_coverage[book_id]['chapters'].append(result)
                books_coverage[book_id]['total_words'] += result['word_count'] or 0

            # Calculate average similarity per book
            for book_id in books_coverage:
                chapters = books_coverage[book_id]['chapters']
                if chapters:
                    books_coverage[book_id]['avg_similarity'] = round(
                        sum(c['similarity'] for c in chapters) / len(chapters), 3
                    )

            # Sort books by average similarity
            sorted_books = sorted(
                books_coverage.values(),
                key=lambda x: x['avg_similarity'],
                reverse=True
            )

            logger.info(f"Topic '{topic}' covered in {len(sorted_books)} books, {len(all_results)} chapters")

            return {
                "topic": topic,
                "total_chapters": len(all_results),
                "books_count": len(sorted_books),
                "coverage_by_book": sorted_books,
                "top_chapters": all_results[:10]  # Top 10 most relevant
            }

        except Exception as e:
            logger.error(f"get_topic_coverage error: {e}", exc_info=True)
            return {"error": str(e), "results": []}

    @mcp.tool()
    def extract_code_examples(
        book_id: str,
        chapter_number: int,
        language: str = ""
    ) -> dict:
        """Extract code examples from a chapter

        Parses content to extract code blocks in various formats:
        - Markdown fenced code blocks (```language ... ```)
        - Dockerfile patterns (FROM, RUN, CMD, etc.)
        - Shell commands (lines starting with $ or #)
        - Python code (def, class, import patterns)

        Args:
            book_id: UUID of the book
            chapter_number: Chapter number to extract from
            language: Optional language filter (e.g., "python", "dockerfile", "bash")

        Returns:
            Dictionary with all code blocks and their surrounding context

        Examples:
            extract_code_examples("abc-123", 5)
            extract_code_examples("abc-123", 5, language="python")
        """
        try:
            # Get chapter file path
            chapter = execute_single("""
                SELECT c.id, c.title, c.file_path, b.title as book_title
                FROM chapters c
                JOIN books b ON c.book_id = b.id
                WHERE c.book_id = ? AND c.chapter_number = ?
            """, (book_id, chapter_number))

            if not chapter:
                return {
                    "error": f"Chapter {chapter_number} not found in book {book_id}",
                    "code_blocks": []
                }

            # Read chapter content (handles split chapters)
            try:
                content = read_chapter_content(chapter['file_path'])
            except Exception as e:
                return {"error": f"Could not read chapter file: {e}", "code_blocks": []}

            code_blocks = []

            # Pattern 1: Markdown fenced code blocks
            fenced_pattern = r'```(\w*)\n(.*?)```'
            for match in re.finditer(fenced_pattern, content, re.DOTALL):
                lang = match.group(1).lower() if match.group(1) else "unknown"
                code = match.group(2).strip()

                if language and lang != language.lower():
                    continue

                context = _get_context(content, match.start(), 200)
                code_blocks.append({
                    'language': lang,
                    'code': code,
                    'context': context,
                    'line_count': len(code.split('\n')),
                    'format': 'fenced'
                })

            # Pattern 2: Dockerfile blocks (FROM followed by Docker instructions)
            if not language or language.lower() in ('dockerfile', 'docker'):
                dockerfile_pattern = r'(FROM\s+\S+.*?(?=\n\n[A-Z]|\n\n[a-z]|\Z))'
                for match in re.finditer(dockerfile_pattern, content, re.DOTALL):
                    code = match.group(1).strip()
                    # Validate it looks like a Dockerfile
                    if any(kw in code for kw in ['FROM', 'RUN', 'CMD', 'COPY', 'ENV', 'WORKDIR']):
                        context = _get_context(content, match.start(), 150)
                        code_blocks.append({
                            'language': 'dockerfile',
                            'code': code,
                            'context': context,
                            'line_count': len(code.split('\n')),
                            'format': 'inline'
                        })

            # Pattern 3: Shell command blocks (consecutive lines with $ or common commands)
            if not language or language.lower() in ('bash', 'shell', 'sh'):
                shell_pattern = r'(\$\s+[^\n]+(?:\n\$\s+[^\n]+)*)'
                for match in re.finditer(shell_pattern, content):
                    code = match.group(1).strip()
                    if len(code) > 10:  # Skip very short matches
                        context = _get_context(content, match.start(), 150)
                        code_blocks.append({
                            'language': 'bash',
                            'code': code,
                            'context': context,
                            'line_count': len(code.split('\n')),
                            'format': 'inline'
                        })

            # Pattern 4: Docker commands
            if not language or language.lower() in ('bash', 'shell', 'docker'):
                docker_cmd_pattern = r'(docker\s+(?:run|build|pull|push|exec|compose)[^\n]+(?:\n\s+[^\n]+)*)'
                for match in re.finditer(docker_cmd_pattern, content, re.IGNORECASE):
                    code = match.group(1).strip()
                    context = _get_context(content, match.start(), 150)
                    code_blocks.append({
                        'language': 'bash',
                        'code': code,
                        'context': context,
                        'line_count': len(code.split('\n')),
                        'format': 'inline'
                    })

            # Deduplicate (some patterns may overlap)
            seen = set()
            unique_blocks = []
            for block in code_blocks:
                code_hash = hash(block['code'][:100])
                if code_hash not in seen:
                    seen.add(code_hash)
                    unique_blocks.append(block)

            code_blocks = unique_blocks

            # Summary statistics
            languages = {}
            for block in code_blocks:
                lang = block['language']
                languages[lang] = languages.get(lang, 0) + 1

            logger.info(f"Extracted {len(code_blocks)} code blocks from {chapter['book_title']} Ch.{chapter_number}")

            return {
                "book_title": chapter['book_title'],
                "chapter_title": chapter['title'],
                "chapter_number": chapter_number,
                "total_blocks": len(code_blocks),
                "languages": languages,
                "code_blocks": code_blocks
            }

        except Exception as e:
            logger.error(f"extract_code_examples error: {e}", exc_info=True)
            return {"error": str(e), "code_blocks": []}

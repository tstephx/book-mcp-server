"""
Batch Operations for Library

Provides memory-efficient batch processing for:
- Multi-book searches
- Bulk exports
- Aggregate operations
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional, List

from ..database import execute_query, get_db_connection

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result of a batch operation"""
    total: int
    processed: int
    results: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'total': self.total,
            'processed': self.processed,
            'success_count': len(self.results),
            'error_count': len(self.errors),
            'results': self.results,
            'errors': self.errors if self.errors else None
        }


def batch_semantic_search(
    query: str,
    book_ids: Optional[List[str]] = None,
    max_per_book: int = 5,
    min_similarity: float = 0.3
) -> dict:
    """Search across multiple books with per-book result limits using chunk-level search

    Args:
        query: Semantic search query
        book_ids: List of book IDs to search (None = all books)
        max_per_book: Maximum results per book (default: 5)
        min_similarity: Minimum similarity threshold (default: 0.3)

    Returns:
        Aggregated results grouped by book
    """
    from .context_managers import embedding_model_context
    from .vector_store import find_top_k
    from .chunk_loader import load_chunk_embeddings, best_chunk_per_chapter

    # Get list of books to search
    if book_ids:
        placeholders = ','.join('?' * len(book_ids))
        books = execute_query(
            f"SELECT id, title FROM books WHERE id IN ({placeholders})",
            tuple(book_ids)
        )
    else:
        books = execute_query("SELECT id, title FROM books ORDER BY title")

    if not books:
        return {
            "query": query,
            "books_searched": 0,
            "total_results": 0,
            "results_by_book": []
        }

    embeddings_matrix, chunk_metadata = load_chunk_embeddings()
    if embeddings_matrix is None:
        return {
            "query": query,
            "error": "No embeddings found",
            "books_searched": 0,
            "total_results": 0,
            "results_by_book": []
        }

    results_by_book = []
    total_results = 0
    errors = []

    with embedding_model_context() as generator:
        query_embedding = generator.generate(query)

        for book in books:
            try:
                # Filter to this book's chunks
                book_indices = [
                    i for i, m in enumerate(chunk_metadata)
                    if m['book_id'] == book['id']
                ]

                if not book_indices:
                    continue

                book_embeddings = embeddings_matrix[book_indices]
                book_chunk_meta = [chunk_metadata[i] for i in book_indices]

                # Find top chunk results for this book
                top_results = find_top_k(
                    query_embedding,
                    book_embeddings,
                    k=max_per_book * 3,
                    min_similarity=min_similarity
                )

                chunk_results = []
                for idx, similarity in top_results:
                    meta = book_chunk_meta[idx]
                    chunk_results.append({**meta, 'similarity': similarity})

                # Aggregate to chapter level
                chapter_results = best_chunk_per_chapter(chunk_results)[:max_per_book]

                book_results = []
                for r in chapter_results:
                    book_results.append({
                        'chapter_number': r['chapter_number'],
                        'chapter_title': r['chapter_title'],
                        'similarity': round(r['similarity'], 3),
                        'excerpt': r.get('excerpt', '')[:300]
                    })

                if book_results:
                    results_by_book.append({
                        'book_id': book['id'],
                        'book_title': book['title'],
                        'result_count': len(book_results),
                        'chapters': book_results
                    })
                    total_results += len(book_results)

            except Exception as e:
                logger.warning(f"Error searching book {book['id']}: {e}")
                errors.append({'book_id': book['id'], 'error': str(e)})

    result = {
        "query": query,
        "books_searched": len(books),
        "total_results": total_results,
        "results_by_book": results_by_book
    }

    if errors:
        result["errors"] = errors

    logger.info(f"Batch search '{query}' found {total_results} results across {len(results_by_book)} books")

    return result


def batch_export_chapters(
    book_ids: Optional[List[str]] = None,
    format: str = "markdown",
    include_content: bool = False
) -> Iterator[str]:
    """Stream export of chapter metadata

    Memory-efficient generator for large exports.

    Args:
        book_ids: List of book IDs (None = all)
        format: "markdown" or "json"
        include_content: Include full chapter content (large!)

    Yields:
        Formatted strings for each chapter
    """
    from .file_utils import read_chapter_content

    # Build query
    if book_ids:
        placeholders = ','.join('?' * len(book_ids))
        query = f"""
            SELECT c.id, c.book_id, c.chapter_number, c.title, c.file_path, c.word_count,
                   b.title as book_title, b.author
            FROM chapters c
            JOIN books b ON c.book_id = b.id
            WHERE c.book_id IN ({placeholders})
            ORDER BY b.title, c.chapter_number
        """
        params = tuple(book_ids)
    else:
        query = """
            SELECT c.id, c.book_id, c.chapter_number, c.title, c.file_path, c.word_count,
                   b.title as book_title, b.author
            FROM chapters c
            JOIN books b ON c.book_id = b.id
            ORDER BY b.title, c.chapter_number
        """
        params = ()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)

        # Yield header
        if format == "markdown":
            yield "# Chapter Export\n\n"
            current_book = None

        for row in cursor:
            if format == "markdown":
                # Add book header when book changes
                if row['book_title'] != current_book:
                    current_book = row['book_title']
                    yield f"\n## {current_book}\n"
                    if row['author']:
                        yield f"*by {row['author']}*\n\n"
                    else:
                        yield "\n"

                yield f"### Chapter {row['chapter_number']}: {row['title']}\n"
                yield f"- Words: {row['word_count'] or 'N/A'}\n"

                if include_content:
                    try:
                        content = read_chapter_content(row['file_path'])
                        yield f"\n{content}\n\n---\n"
                    except Exception as e:
                        yield f"- *Content unavailable: {e}*\n"

                yield "\n"

            else:  # json
                data = {
                    'chapter_id': row['id'],
                    'book_id': row['book_id'],
                    'book_title': row['book_title'],
                    'author': row['author'],
                    'chapter_number': row['chapter_number'],
                    'chapter_title': row['title'],
                    'word_count': row['word_count']
                }

                if include_content:
                    try:
                        data['content'] = read_chapter_content(row['file_path'])
                    except Exception as e:
                        data['content_error'] = str(e)

                yield json.dumps(data) + "\n"


def get_library_statistics() -> dict:
    """Get comprehensive library statistics

    Returns aggregate stats useful for batch reporting.
    """
    stats = {}

    # Basic counts
    rows = execute_query("""
        SELECT
            (SELECT COUNT(*) FROM books) as book_count,
            (SELECT COUNT(*) FROM chapters) as chapter_count,
            (SELECT COALESCE(SUM(word_count), 0) FROM books) as total_words,
            (SELECT COUNT(*) FROM chapters WHERE embedding IS NOT NULL) as embedded_chapters
    """)

    if rows:
        row = rows[0]
        stats['books'] = row['book_count']
        stats['chapters'] = row['chapter_count']
        stats['total_words'] = row['total_words']
        stats['embedded_chapters'] = row['embedded_chapters']
        stats['embedding_coverage'] = round(
            row['embedded_chapters'] / row['chapter_count'] * 100, 1
        ) if row['chapter_count'] > 0 else 0

    # Per-book stats
    books = execute_query("""
        SELECT b.id, b.title, b.word_count,
               COUNT(c.id) as chapters,
               SUM(CASE WHEN c.embedding IS NOT NULL THEN 1 ELSE 0 END) as embedded
        FROM books b
        LEFT JOIN chapters c ON c.book_id = b.id
        GROUP BY b.id
        ORDER BY b.word_count DESC
    """)

    stats['books_detail'] = [
        {
            'id': b['id'],
            'title': b['title'],
            'word_count': b['word_count'],
            'chapters': b['chapters'],
            'embedded': b['embedded']
        }
        for b in books
    ]

    # FTS status
    try:
        fts_rows = execute_query("SELECT COUNT(*) as count FROM chapters_fts")
        stats['fts_indexed'] = fts_rows[0]['count'] if fts_rows else 0
    except Exception:
        stats['fts_indexed'] = 0

    # Summaries status
    try:
        sum_rows = execute_query("SELECT COUNT(*) as count FROM chapter_summaries")
        stats['summaries_generated'] = sum_rows[0]['count'] if sum_rows else 0
    except Exception:
        stats['summaries_generated'] = 0

    return stats

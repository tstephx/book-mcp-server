"""
Analytics tools for library insights and statistics
Provides deep analysis of library content, coverage, and patterns
"""

import logging
from collections import Counter
from typing import TYPE_CHECKING
import numpy as np
import io

from ..database import get_db_connection, execute_query, execute_single
from ..utils.vector_store import cosine_similarity

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Shared topic detection keywords
TOPIC_KEYWORDS = {
    'Python': ['python', 'django', 'flask', 'fastapi', 'pytest', 'pip'],
    'Data Science': ['data', 'analytics', 'pandas', 'numpy', 'visualization', 'matplotlib'],
    'Machine Learning': ['ml', 'machine learning', 'deep learning', 'neural', 'ai', 'llm', 'tensorflow', 'pytorch'],
    'Architecture': ['architecture', 'design patterns', 'clean code', 'solid', 'refactoring'],
    'DevOps': ['docker', 'kubernetes', 'k8s', 'devops', 'ci/cd', 'cloud', 'container'],
    'Linux': ['linux', 'ubuntu', 'systemd', 'kernel', 'bash', 'shell'],
    'Networking': ['networking', 'network', 'firewall', 'vpn', 'tcp', 'ip', 'http'],
    'Web Development': ['web', 'api', 'rest', 'frontend', 'backend', 'node', 'javascript'],
    'Databases': ['database', 'sql', 'nosql', 'postgresql', 'mongodb', 'redis'],
    'Security': ['security', 'encryption', 'authentication', 'oauth', 'ssl', 'tls'],
    'Quantum': ['quantum', 'qubit'],
    'Forecasting': ['forecasting', 'time series', 'prediction', 'arima'],
    'Async/Concurrency': ['async', 'await', 'concurrent', 'parallel', 'threading', 'multiprocessing'],
}


def _detect_topics(text: str) -> list:
    """Detect topics from text based on keywords"""
    text_lower = text.lower()
    detected = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            detected.append(topic)
    return detected if detected else ['General']


def register_analytics_tools(mcp: "FastMCP") -> None:
    """Register analytics tools with the MCP server"""

    @mcp.tool()
    def get_library_statistics() -> dict:
        """Get comprehensive library statistics and analytics

        Provides deep insights into your library including:
        - Word counts by topic
        - Author distribution
        - Book size distribution
        - Embedding coverage
        - Reading progress summary

        Returns:
            Dictionary with detailed library analytics

        Examples:
            get_library_statistics()
        """
        try:
            # Basic stats
            basic = execute_single("""
                SELECT
                    (SELECT COUNT(*) FROM books) as total_books,
                    (SELECT COUNT(*) FROM chapters) as total_chapters,
                    (SELECT COALESCE(SUM(word_count), 0) FROM books) as total_words,
                    (SELECT COUNT(*) FROM chapters WHERE embedding IS NOT NULL) as embedded_chapters
            """)

            # Get all books with details
            books = execute_query("""
                SELECT
                    b.id,
                    b.title,
                    b.author,
                    b.word_count,
                    b.added_date,
                    COUNT(c.id) as chapter_count,
                    SUM(CASE WHEN c.embedding IS NOT NULL THEN 1 ELSE 0 END) as embedded_count
                FROM books b
                LEFT JOIN chapters c ON c.book_id = b.id
                GROUP BY b.id
                ORDER BY b.word_count DESC
            """)

            # Word counts by topic
            topic_words = Counter()
            topic_books = Counter()
            topic_chapters = Counter()

            for book in books:
                topics = _detect_topics(book['title'])
                for topic in topics:
                    topic_words[topic] += book['word_count'] or 0
                    topic_books[topic] += 1

            # Get chapter-level topic distribution
            chapters = execute_query("SELECT title, word_count FROM chapters")
            for chapter in chapters:
                topics = _detect_topics(chapter['title'])
                for topic in topics:
                    topic_chapters[topic] += 1

            # Author distribution
            author_stats = {}
            for book in books:
                author = book['author'] or 'Unknown'
                if author not in author_stats:
                    author_stats[author] = {'books': 0, 'words': 0, 'chapters': 0}
                author_stats[author]['books'] += 1
                author_stats[author]['words'] += book['word_count'] or 0
                author_stats[author]['chapters'] += book['chapter_count']

            # Sort authors by word count
            sorted_authors = sorted(
                author_stats.items(),
                key=lambda x: x[1]['words'],
                reverse=True
            )

            # Book size distribution
            word_counts = [b['word_count'] or 0 for b in books]
            size_distribution = {
                'smallest': min(word_counts) if word_counts else 0,
                'largest': max(word_counts) if word_counts else 0,
                'average': sum(word_counts) // len(word_counts) if word_counts else 0,
                'median': sorted(word_counts)[len(word_counts)//2] if word_counts else 0,
            }

            # Categorize by size
            size_categories = {'short': 0, 'medium': 0, 'long': 0}
            for wc in word_counts:
                if wc < 50000:
                    size_categories['short'] += 1
                elif wc < 150000:
                    size_categories['medium'] += 1
                else:
                    size_categories['long'] += 1

            # Reading progress (if tables exist)
            try:
                progress = execute_single("""
                    SELECT
                        (SELECT COUNT(*) FROM reading_progress WHERE status = 'read') as chapters_read,
                        (SELECT COUNT(DISTINCT book_id) FROM reading_progress WHERE status = 'read') as books_started,
                        (SELECT COUNT(*) FROM bookmarks) as total_bookmarks
                """)
                reading_stats = {
                    'chapters_read': progress['chapters_read'] or 0,
                    'books_started': progress['books_started'] or 0,
                    'bookmarks': progress['total_bookmarks'] or 0,
                    'percent_complete': round((progress['chapters_read'] or 0) / basic['total_chapters'] * 100, 1) if basic['total_chapters'] > 0 else 0
                }
            except Exception:
                reading_stats = {'chapters_read': 0, 'books_started': 0, 'bookmarks': 0, 'percent_complete': 0}

            # Embedding coverage
            embedding_coverage = round(basic['embedded_chapters'] / basic['total_chapters'] * 100, 1) if basic['total_chapters'] > 0 else 0

            logger.info("Generated library statistics")

            return {
                "summary": {
                    "total_books": basic['total_books'],
                    "total_chapters": basic['total_chapters'],
                    "total_words": basic['total_words'],
                    "estimated_reading_hours": basic['total_words'] // 15000,  # ~250 wpm
                },
                "topic_distribution": {
                    "by_word_count": [
                        {"topic": topic, "words": count, "percent": round(count / basic['total_words'] * 100, 1)}
                        for topic, count in topic_words.most_common()
                    ],
                    "by_book_count": [
                        {"topic": topic, "books": count}
                        for topic, count in topic_books.most_common()
                    ],
                    "by_chapter_count": [
                        {"topic": topic, "chapters": count}
                        for topic, count in topic_chapters.most_common(10)
                    ]
                },
                "author_distribution": [
                    {
                        "author": author,
                        "books": stats['books'],
                        "words": stats['words'],
                        "chapters": stats['chapters']
                    }
                    for author, stats in sorted_authors[:10]
                ],
                "book_sizes": {
                    "distribution": size_distribution,
                    "categories": {
                        "short_under_50k": size_categories['short'],
                        "medium_50k_150k": size_categories['medium'],
                        "long_over_150k": size_categories['long']
                    },
                    "largest_books": [
                        {"title": b['title'], "words": b['word_count']}
                        for b in books[:5]
                    ]
                },
                "embedding_coverage": {
                    "chapters_with_embeddings": basic['embedded_chapters'],
                    "total_chapters": basic['total_chapters'],
                    "percent_covered": embedding_coverage,
                    "books_fully_embedded": sum(1 for b in books if b['embedded_count'] == b['chapter_count']),
                    "books_partially_embedded": sum(1 for b in books if 0 < b['embedded_count'] < b['chapter_count']),
                    "books_not_embedded": sum(1 for b in books if b['embedded_count'] == 0)
                },
                "reading_progress": reading_stats
            }

        except Exception as e:
            logger.error(f"get_library_statistics error: {e}", exc_info=True)
            return {"error": str(e)}

    @mcp.tool()
    def find_duplicate_coverage(
        similarity_threshold: float = 0.7,
        min_results: int = 5,
        max_results: int = 50
    ) -> dict:
        """Find chapters covering similar topics across different books

        Identifies potential duplicate or overlapping coverage by comparing
        chapter embeddings across the library. Useful for:
        - Finding alternative explanations of the same concept
        - Identifying redundant content
        - Cross-referencing perspectives from different authors

        Args:
            similarity_threshold: Minimum similarity to consider as duplicate (0.5-0.95, default: 0.7)
            min_results: Minimum pairs to return (default: 5)
            max_results: Maximum pairs to return (default: 50)

        Returns:
            Dictionary with similar chapter pairs, grouped by topic

        Examples:
            find_duplicate_coverage()  # Default threshold
            find_duplicate_coverage(similarity_threshold=0.8)  # Stricter matching
        """
        try:
            # Validate threshold
            similarity_threshold = max(0.5, min(0.95, similarity_threshold))

            # Fetch all chapter embeddings
            chapters = execute_query("""
                SELECT
                    c.id,
                    c.book_id,
                    c.chapter_number,
                    c.title as chapter_title,
                    c.word_count,
                    c.embedding,
                    b.title as book_title,
                    b.author
                FROM chapters c
                JOIN books b ON c.book_id = b.id
                WHERE c.embedding IS NOT NULL
                ORDER BY b.title, c.chapter_number
            """)

            if len(chapters) < 2:
                return {"message": "Need at least 2 chapters with embeddings", "pairs": []}

            # Load all embeddings
            embeddings = []
            metadata = []
            for ch in chapters:
                emb = np.load(io.BytesIO(ch['embedding']))
                embeddings.append(emb)
                metadata.append({
                    'id': ch['id'],
                    'book_id': ch['book_id'],
                    'book_title': ch['book_title'],
                    'author': ch['author'],
                    'chapter_number': ch['chapter_number'],
                    'chapter_title': ch['chapter_title'],
                    'word_count': ch['word_count']
                })

            # Find similar pairs (only across different books)
            similar_pairs = []
            n = len(embeddings)

            for i in range(n):
                for j in range(i + 1, n):
                    # Skip if same book
                    if metadata[i]['book_id'] == metadata[j]['book_id']:
                        continue

                    similarity = cosine_similarity(embeddings[i], embeddings[j])

                    if similarity >= similarity_threshold:
                        similar_pairs.append({
                            'similarity': round(float(similarity), 3),
                            'chapter_1': metadata[i],
                            'chapter_2': metadata[j],
                            'detected_topic': _detect_topics(
                                metadata[i]['chapter_title'] + ' ' + metadata[j]['chapter_title']
                            )[0]
                        })

            # Sort by similarity (highest first)
            similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)

            # Limit results
            if len(similar_pairs) > max_results:
                similar_pairs = similar_pairs[:max_results]
            elif len(similar_pairs) < min_results:
                # Lower threshold and try again if needed
                pass  # Keep what we have

            # Group by detected topic
            by_topic = {}
            for pair in similar_pairs:
                topic = pair['detected_topic']
                if topic not in by_topic:
                    by_topic[topic] = []
                by_topic[topic].append({
                    'similarity': pair['similarity'],
                    'chapter_1': {
                        'book': pair['chapter_1']['book_title'],
                        'author': pair['chapter_1']['author'],
                        'chapter': f"Ch.{pair['chapter_1']['chapter_number']}: {pair['chapter_1']['chapter_title']}"
                    },
                    'chapter_2': {
                        'book': pair['chapter_2']['book_title'],
                        'author': pair['chapter_2']['author'],
                        'chapter': f"Ch.{pair['chapter_2']['chapter_number']}: {pair['chapter_2']['chapter_title']}"
                    }
                })

            # Summary statistics
            books_with_overlap = set()
            for pair in similar_pairs:
                books_with_overlap.add(pair['chapter_1']['book_title'])
                books_with_overlap.add(pair['chapter_2']['book_title'])

            logger.info(f"Found {len(similar_pairs)} similar chapter pairs")

            return {
                "summary": {
                    "total_pairs_found": len(similar_pairs),
                    "similarity_threshold": similarity_threshold,
                    "topics_with_overlap": len(by_topic),
                    "books_with_overlap": len(books_with_overlap)
                },
                "by_topic": [
                    {
                        "topic": topic,
                        "overlap_count": len(pairs),
                        "pairs": pairs[:10]  # Limit per topic
                    }
                    for topic, pairs in sorted(by_topic.items(), key=lambda x: len(x[1]), reverse=True)
                ],
                "top_similar_pairs": [
                    {
                        "similarity": p['similarity'],
                        "chapter_1": f"{p['chapter_1']['book_title']} - Ch.{p['chapter_1']['chapter_number']}",
                        "chapter_2": f"{p['chapter_2']['book_title']} - Ch.{p['chapter_2']['chapter_number']}",
                        "topic": p['detected_topic']
                    }
                    for p in similar_pairs[:10]
                ],
                "recommendations": {
                    "high_overlap_topics": [
                        topic for topic, pairs in by_topic.items() if len(pairs) >= 3
                    ],
                    "suggestion": "Topics with high overlap offer multiple perspectives - great for cross-referencing and deeper understanding."
                }
            }

        except Exception as e:
            logger.error(f"find_duplicate_coverage error: {e}", exc_info=True)
            return {"error": str(e)}

    @mcp.tool()
    def get_author_insights(author_name: str = "") -> dict:
        """Get insights about authors in your library

        Analyze author contributions, writing style metrics, and topic expertise.

        Args:
            author_name: Specific author to analyze (empty = all authors)

        Returns:
            Dictionary with author analytics and insights

        Examples:
            get_author_insights()  # All authors
            get_author_insights("Elton Stoneman")  # Specific author
        """
        try:
            if author_name:
                # Specific author analysis
                books = execute_query("""
                    SELECT
                        b.id,
                        b.title,
                        b.word_count,
                        COUNT(c.id) as chapter_count,
                        AVG(c.word_count) as avg_chapter_words
                    FROM books b
                    LEFT JOIN chapters c ON c.book_id = b.id
                    WHERE b.author LIKE ?
                    GROUP BY b.id
                    ORDER BY b.title
                """, (f"%{author_name}%",))

                if not books:
                    return {"error": f"No books found for author: {author_name}"}

                # Get all chapter titles for topic analysis
                chapters = execute_query("""
                    SELECT c.title, c.word_count
                    FROM chapters c
                    JOIN books b ON c.book_id = b.id
                    WHERE b.author LIKE ?
                """, (f"%{author_name}%",))

                # Analyze topics covered
                topic_coverage = Counter()
                for ch in chapters:
                    for topic in _detect_topics(ch['title']):
                        topic_coverage[topic] += 1

                total_words = sum(b['word_count'] or 0 for b in books)
                total_chapters = sum(b['chapter_count'] for b in books)

                return {
                    "author": author_name,
                    "summary": {
                        "total_books": len(books),
                        "total_chapters": total_chapters,
                        "total_words": total_words,
                        "avg_book_length": total_words // len(books) if books else 0,
                        "avg_chapter_length": total_words // total_chapters if total_chapters else 0
                    },
                    "books": [
                        {
                            "title": b['title'],
                            "chapters": b['chapter_count'],
                            "words": b['word_count'],
                            "avg_chapter_words": round(b['avg_chapter_words'] or 0)
                        }
                        for b in books
                    ],
                    "topic_expertise": [
                        {"topic": topic, "chapters": count}
                        for topic, count in topic_coverage.most_common()
                    ]
                }

            else:
                # All authors overview
                authors = execute_query("""
                    SELECT
                        COALESCE(b.author, 'Unknown') as author,
                        COUNT(DISTINCT b.id) as book_count,
                        SUM(b.word_count) as total_words,
                        COUNT(c.id) as chapter_count
                    FROM books b
                    LEFT JOIN chapters c ON c.book_id = b.id
                    GROUP BY b.author
                    ORDER BY total_words DESC
                """)

                return {
                    "total_authors": len(authors),
                    "authors": [
                        {
                            "name": a['author'],
                            "books": a['book_count'],
                            "chapters": a['chapter_count'],
                            "words": a['total_words'],
                            "avg_words_per_book": a['total_words'] // a['book_count'] if a['book_count'] else 0
                        }
                        for a in authors
                    ],
                    "insights": {
                        "most_prolific": authors[0]['author'] if authors else None,
                        "total_unique_authors": len([a for a in authors if a['author'] != 'Unknown'])
                    }
                }

        except Exception as e:
            logger.error(f"get_author_insights error: {e}", exc_info=True)
            return {"error": str(e)}

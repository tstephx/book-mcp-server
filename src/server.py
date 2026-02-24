"""
Book Library MCP Server
Production-ready FastMCP server following MCP best practices

Architecture:
- Modular tool organization
- Proper error handling
- Configuration management
- Connection pooling
- Input validation
- Logging
"""

from mcp.server.fastmcp import FastMCP

from .config import Config
from .database import check_database_health, ensure_library_schema
from .utils.logging import logger
from .tools.book_tools import register_book_tools
from .tools.chapter_tools import register_chapter_tools
from .tools.search_tools import register_search_tools
from .tools.semantic_search_tools import register_semantic_search_tools
from .tools.hybrid_search_tools import register_hybrid_search_tools
from .tools.discovery_tools import register_discovery_tools
from .tools.reading_tools import register_reading_tools
from .tools.analytics_tools import register_analytics_tools
from .tools.export_tools import register_export_tools
from .tools.learning_tools import register_learning_tools
from .tools.project_learning_tools import register_project_learning_tools
from .tools.project_planning_tools import register_project_planning_tools
from .tools.audit_tools import register_audit_tools
from .utils.cache import get_cache

def create_server() -> FastMCP:
    """
    Create and configure the MCP server
    
    Returns:
        Configured FastMCP server instance
    """
    # Validate configuration
    try:
        Config.validate()
        if Config.DEBUG:
            logger.info(Config.display())
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise
    
    # Check database health
    health = check_database_health()
    if health["status"] != "healthy":
        logger.error(f"Database unhealthy: {health.get('error')}")
        raise RuntimeError(f"Database health check failed: {health.get('error')}")
    
    logger.info(f"Database healthy: {health['books']} books, {health['chapters']} chapters, "
               f"{health['total_words']:,} total words")

    # Ensure library-side tables and columns exist
    ensure_library_schema()

    # Create server
    mcp = FastMCP(Config.SERVER_NAME)
    
    # Register tools (modular organization)
    logger.info("Registering tools...")
    register_book_tools(mcp)
    register_chapter_tools(mcp)
    register_search_tools(mcp)
    register_semantic_search_tools(mcp)  # Semantic search tool + resource
    register_hybrid_search_tools(mcp)  # Hybrid FTS+semantic search with RRF fusion
    register_discovery_tools(mcp)  # Cross-book discovery tools
    register_reading_tools(mcp)  # Reading progress and bookmarks
    register_analytics_tools(mcp)  # Library analytics and statistics
    register_export_tools(mcp)  # Export and study guide generation
    register_learning_tools(mcp)  # PM-focused concept teaching
    register_project_learning_tools(mcp)  # Project-based learning paths
    register_project_planning_tools(mcp)  # Implementation plans and project artifacts
    register_audit_tools(mcp)  # Chapter quality audit

    # Register embedding sync tool
    @mcp.tool()
    def refresh_embeddings(force: bool = False) -> dict:
        """Refresh embeddings for changed chapters

        Detects content changes using file modification times and content
        hashes. Only regenerates embeddings for new or modified chapters,
        making it efficient for incremental updates.

        Args:
            force: If True, regenerate ALL embeddings (ignores change detection)

        Returns:
            Dictionary with:
            - status: 'updated' | 'no_updates_needed' | 'error'
            - updated: Number of embeddings regenerated
            - skipped: Number of unchanged chapters
            - errors: Number of failed chapters
            - duration_seconds: Time taken

        Examples:
            refresh_embeddings()  # Only update changed chapters
            refresh_embeddings(force=True)  # Regenerate everything
        """
        try:
            from .utils.embedding_sync import update_embeddings_incremental

            result = update_embeddings_incremental(force=force)
            logger.info(f"Embedding refresh complete: {result}")
            return result
        except Exception as e:
            logger.error(f"Embedding refresh error: {e}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e),
                'updated': 0,
                'skipped': 0,
                'errors': 1
            }

    # Register summary embeddings tool
    @mcp.tool()
    def generate_summary_embeddings(force: bool = False) -> dict:
        """Generate embeddings for chapter summaries

        Creates vector embeddings for existing chapter summaries,
        enabling future summary-based semantic search.

        Args:
            force: If True, regenerate ALL summary embeddings (default: False)

        Returns:
            Dictionary with:
            - generated: Number of embeddings created
            - skipped: Number already existing (when force=False)
            - errors: Number of failures
            - total: Total summaries processed

        Examples:
            generate_summary_embeddings()  # Only missing
            generate_summary_embeddings(force=True)  # Regenerate all
        """
        try:
            from .utils.summaries import batch_generate_summary_embeddings

            result = batch_generate_summary_embeddings(force=force)
            logger.info(f"Summary embedding generation: {result}")
            return result
        except Exception as e:
            logger.error(f"Summary embedding error: {e}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e),
                'generated': 0,
                'skipped': 0,
                'errors': 1
            }

    # Register cache management tool
    @mcp.tool()
    def get_cache_stats() -> dict:
        """Get cache statistics for monitoring and debugging

        Returns information about the caching system including:
        - Whether caching is enabled
        - Number of cached chapters and embeddings
        - Memory usage
        - Hit/miss rates

        Returns:
            Dictionary with cache statistics

        Examples:
            get_cache_stats()
        """
        cache = get_cache()
        return cache.stats()

    @mcp.tool()
    def clear_cache(cache_type: str = "all") -> dict:
        """Clear cached data

        Args:
            cache_type: What to clear - "chapters", "embeddings", "summary_embeddings", or "all" (default)

        Returns:
            Dictionary with confirmation and updated stats

        Examples:
            clear_cache()
            clear_cache("chapters")
        """
        cache = get_cache()

        if cache_type == "chapters":
            cache.clear_chapters()
            message = "Chapter cache cleared"
        elif cache_type == "embeddings":
            cache.invalidate_embeddings()
            message = "Embeddings cache invalidated"
        elif cache_type == "summary_embeddings":
            cache.invalidate_summary_embeddings()
            message = "Summary embeddings cache invalidated"
        elif cache_type == "all":
            cache.clear_all()
            message = "All caches cleared"
        else:
            return {"error": f"Invalid cache_type: {cache_type}. Use 'chapters', 'embeddings', 'summary_embeddings', or 'all'"}

        return {
            "message": message,
            "stats": cache.stats()
        }

    # Register library status tool
    @mcp.tool()
    def library_status() -> dict:
        """Get unified library status dashboard

        Shows which books are fully ready (content + embeddings for semantic
        search), which are partially ready, and which are still in the pipeline.

        Returns:
            Dictionary with:
            - overview: total books, chapters, words, embedding coverage
            - books: per-book status with embedding percentage and pipeline state
            - pipeline_summary: pipeline state counts

        Examples:
            library_status()
        """
        try:
            from agentic_pipeline.library import LibraryStatus
            from agentic_pipeline.db.config import get_db_path

            monitor = LibraryStatus(get_db_path())
            result = monitor.get_status()
            logger.info(f"Library status: {result['overview']['total_books']} books, "
                       f"{result['overview']['embedding_coverage_pct']}% embedded")
            return result
        except Exception as e:
            logger.error(f"Library status error: {e}", exc_info=True)
            return {"error": str(e)}

    # =========================================================================
    # Full-Text Search Tools
    # =========================================================================

    @mcp.tool()
    def text_search(
        query: str,
        limit: int = 10,
        book_id: str = None
    ) -> dict:
        """Search chapter content using full-text search (FTS5)

        Fast keyword/phrase search with BM25 ranking. Complements semantic
        search for exact term matching.

        Supports:
        - Phrase search: "async await"
        - Boolean: python AND async, python OR async
        - Prefix: python*
        - Negation: python NOT java

        Args:
            query: Search query (FTS5 syntax supported)
            limit: Maximum results (1-50, default: 10)
            book_id: Optional book ID to filter results

        Returns:
            Dictionary with results including highlighted excerpts

        Examples:
            text_search("async await")  # Find chapters with both terms
            text_search("\"dependency injection\"")  # Exact phrase
            text_search("docker NOT kubernetes", book_id="docker-deep-dive")
        """
        try:
            from .utils.fts_search import full_text_search
            return full_text_search(query, limit=limit, book_id=book_id)
        except Exception as e:
            logger.error(f"Text search error: {e}", exc_info=True)
            return {"error": str(e), "results": []}

    # =========================================================================
    # Batch Operations Tools
    # =========================================================================

    @mcp.tool()
    def search_all_books(
        query: str,
        max_per_book: int = 5,
        min_similarity: float = 0.3
    ) -> dict:
        """Semantic search across ALL books with per-book result limits

        Searches your entire library and groups results by book.
        Useful for broad research across multiple sources.

        Args:
            query: Semantic search query
            max_per_book: Maximum results per book (default: 5)
            min_similarity: Minimum similarity threshold (default: 0.3)

        Returns:
            Dictionary with results grouped by book

        Examples:
            search_all_books("error handling patterns")
            search_all_books("async programming", max_per_book=3)
        """
        try:
            from .utils.batch_ops import batch_semantic_search
            return batch_semantic_search(
                query,
                book_ids=None,  # All books
                max_per_book=max_per_book,
                min_similarity=min_similarity
            )
        except Exception as e:
            logger.error(f"Batch search error: {e}", exc_info=True)
            return {"error": str(e), "results_by_book": []}

    @mcp.tool()
    def get_library_stats() -> dict:
        """Get comprehensive library statistics

        Returns aggregate statistics useful for understanding your library:
        - Total books, chapters, words
        - Embedding coverage
        - FTS index status
        - Per-book breakdown

        Returns:
            Dictionary with library statistics

        Examples:
            get_library_stats()
        """
        try:
            from .utils.batch_ops import get_library_statistics
            return get_library_statistics()
        except Exception as e:
            logger.error(f"Stats error: {e}", exc_info=True)
            return {"error": str(e)}

    # =========================================================================
    # Chapter Summary Tools
    # =========================================================================

    @mcp.tool()
    def get_summary(chapter_id: str, force: bool = False) -> dict:
        """Get or generate a summary for a chapter

        Uses extractive summarization to create a concise summary
        highlighting key points from the chapter content.

        Args:
            chapter_id: Chapter ID
            force: Regenerate even if cached (default: False)

        Returns:
            Dictionary with chapter summary

        Examples:
            get_summary("clean-code-ch1")
            get_summary("docker-networking", force=True)
        """
        try:
            from .utils.summaries import generate_chapter_summary
            return generate_chapter_summary(chapter_id, force=force)
        except Exception as e:
            logger.error(f"Summary error: {e}", exc_info=True)
            return {"error": str(e)}

    @mcp.tool()
    def summarize_book(book_id: str, force: bool = False) -> dict:
        """Generate summaries for all chapters in a book

        Batch generates summaries for every chapter in the book.
        Results are cached for future use.

        Args:
            book_id: Book ID
            force: Regenerate all summaries (default: False)

        Returns:
            Dictionary with generation results

        Examples:
            summarize_book("python-cookbook")
            summarize_book("clean-architecture", force=True)
        """
        try:
            from .utils.summaries import generate_book_summaries
            return generate_book_summaries(book_id, force=force)
        except Exception as e:
            logger.error(f"Book summary error: {e}", exc_info=True)
            return {"error": str(e)}

    # Register resources
    @mcp.resource("library://catalog")
    def get_library_catalog() -> str:
        """Resource: Complete library catalog with statistics and topic distribution

        Returns comprehensive library overview including:
        - Total books, chapters, and word counts
        - Topic distribution based on book titles
        - Full book listing with metadata
        """
        from .database import execute_query, execute_single
        from collections import Counter
        import re

        try:
            # Get aggregate statistics
            stats = execute_single("""
                SELECT
                    (SELECT COUNT(*) FROM books) as book_count,
                    (SELECT COUNT(*) FROM chapters) as chapter_count,
                    (SELECT COALESCE(SUM(word_count), 0) FROM books) as total_words
            """)

            # Get all books with chapter counts
            books = execute_query("""
                SELECT
                    b.id,
                    b.title,
                    b.author,
                    b.word_count,
                    b.added_date,
                    COUNT(c.id) as chapter_count,
                    SUM(CASE WHEN c.embedding IS NOT NULL THEN 1 ELSE 0 END) as embedded_chapters
                FROM books b
                LEFT JOIN chapters c ON c.book_id = b.id
                GROUP BY b.id
                ORDER BY b.title
            """)

            # Extract topics from titles
            topic_keywords = {
                'Python': ['python', 'django', 'flask', 'fastapi'],
                'Data Science': ['data', 'analytics', 'pandas', 'numpy', 'visualization'],
                'Machine Learning': ['ml', 'machine learning', 'deep learning', 'neural', 'ai', 'llm'],
                'Architecture': ['architecture', 'design patterns', 'clean code', 'solid'],
                'DevOps': ['docker', 'kubernetes', 'k8s', 'devops', 'ci/cd', 'cloud'],
                'Web Development': ['web', 'api', 'rest', 'frontend', 'backend'],
                'Quantum': ['quantum'],
                'Forecasting': ['forecasting', 'time series', 'prediction'],
            }

            topic_counts = Counter()
            for book in books:
                title_lower = book['title'].lower()
                matched = False
                for topic, keywords in topic_keywords.items():
                    if any(kw in title_lower for kw in keywords):
                        topic_counts[topic] += 1
                        matched = True
                if not matched:
                    topic_counts['Other'] += 1

            # Build formatted output
            result = "📚 Library Catalog\n" + "="*60 + "\n\n"

            # Statistics section
            result += "📊 Statistics\n" + "-"*60 + "\n"
            result += f"Total Books: {stats['book_count']}\n"
            result += f"Total Chapters: {stats['chapter_count']}\n"
            result += f"Total Words: {stats['total_words']:,}\n"
            avg_words = stats['total_words'] // stats['book_count'] if stats['book_count'] > 0 else 0
            result += f"Average Words per Book: {avg_words:,}\n\n"

            # Topic distribution section
            result += "🏷️  Topic Distribution\n" + "-"*60 + "\n"
            for topic, count in topic_counts.most_common():
                pct = (count / len(books) * 100) if books else 0
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                result += f"{topic:20} {bar} {count:3} ({pct:.0f}%)\n"
            result += "\n"

            # Books listing section
            result += "📖 All Books\n" + "-"*60 + "\n"
            for book in books:
                result += f"\n• {book['title']}\n"
                if book['author']:
                    result += f"  Author: {book['author']}\n"
                result += f"  Chapters: {book['chapter_count']}"
                if book['embedded_chapters']:
                    result += f" ({book['embedded_chapters']} with embeddings)"
                result += f"\n  Words: {book['word_count']:,}\n"
                result += f"  ID: {book['id']}\n"

            logger.info(f"Generated library catalog with {len(books)} books")
            return result

        except Exception as e:
            logger.error(f"Error in library catalog resource: {e}", exc_info=True)
            return f"Error generating catalog: {str(e)}"

    @mcp.resource("book://catalog")
    def get_catalog() -> str:
        """Resource: Complete book catalog"""
        from .tools.book_tools import register_book_tools
        # We need to call list_books but it's decorated, so we import the function
        from .database import execute_query
        
        try:
            books = execute_query("""
                SELECT id, title, author, word_count
                FROM books
                ORDER BY title
            """)
            
            result = "📚 Book Catalog\n" + "="*50 + "\n\n"
            for book in books:
                result += f"• {book['title']}"
                if book['author']:
                    result += f" by {book['author']}"
                result += f" ({book['word_count']:,} words)\n"
                result += f"  ID: {book['id']}\n"
            
            return result
        except Exception as e:
            logger.error(f"Error in catalog resource: {e}")
            return f"Error: {str(e)}"
    
    @mcp.resource("book://{book_id}/metadata")
    def get_book_metadata(book_id: str) -> str:
        """Resource: Comprehensive book metadata for RAG context injection

        Returns rich metadata including:
        - Basic info (title, author, word count)
        - Chapter listing with titles
        - Detected topics/themes
        - Embedding coverage status

        Perfect for context injection when discussing a specific book.
        """
        from .database import execute_query, execute_single
        from .utils.validators import resolve_book_id, ValidationError

        # Topic detection keywords (shared with catalog)
        topic_keywords = {
            'Python': ['python', 'django', 'flask', 'fastapi'],
            'Data Science': ['data', 'analytics', 'pandas', 'numpy', 'visualization'],
            'Machine Learning': ['ml', 'machine learning', 'deep learning', 'neural', 'ai', 'llm'],
            'Architecture': ['architecture', 'design patterns', 'clean code', 'solid'],
            'DevOps': ['docker', 'kubernetes', 'k8s', 'devops', 'ci/cd', 'cloud'],
            'Web Development': ['web', 'api', 'rest', 'frontend', 'backend'],
            'Linux': ['linux', 'ubuntu', 'systemd', 'kernel', 'bash'],
            'Networking': ['networking', 'network', 'firewall', 'vpn', 'tcp', 'ip'],
            'Quantum': ['quantum'],
            'Forecasting': ['forecasting', 'time series', 'prediction'],
        }

        try:
            book_id = resolve_book_id(book_id)

            # Get book details
            book = execute_single("SELECT * FROM books WHERE id = ?", (book_id,))
            if not book:
                return f"Book not found: {book_id}"

            # Get chapters with embedding status
            chapters = execute_query("""
                SELECT chapter_number, title, word_count,
                       CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END as has_embedding
                FROM chapters
                WHERE book_id = ?
                ORDER BY chapter_number
            """, (book_id,))

            # Detect topics
            title_lower = book['title'].lower()
            detected_topics = []
            for topic, keywords in topic_keywords.items():
                if any(kw in title_lower for kw in keywords):
                    detected_topics.append(topic)
            if not detected_topics:
                detected_topics = ['General']

            # Build formatted output
            result = f"📖 {book['title']}\n" + "="*60 + "\n\n"

            # Basic metadata
            result += "📋 Metadata\n" + "-"*60 + "\n"
            if book['author']:
                result += f"Author: {book['author']}\n"
            result += f"Total Words: {book['word_count']:,}\n"
            result += f"Total Chapters: {len(chapters)}\n"
            result += f"Topics: {', '.join(detected_topics)}\n"
            result += f"Status: {book['processing_status']}\n"
            result += f"Added: {book['added_date']}\n"

            # Embedding status
            embedded_count = sum(1 for c in chapters if c['has_embedding'])
            if embedded_count == len(chapters):
                result += f"Embeddings: ✅ Complete ({embedded_count}/{len(chapters)})\n"
            elif embedded_count > 0:
                result += f"Embeddings: ⚠️ Partial ({embedded_count}/{len(chapters)})\n"
            else:
                result += f"Embeddings: ❌ None\n"

            # Chapter listing
            result += f"\n📑 Chapters\n" + "-"*60 + "\n"
            for ch in chapters:
                embed_icon = "✓" if ch['has_embedding'] else "○"
                result += f"{embed_icon} {ch['chapter_number']:2}. {ch['title']}"
                if ch['word_count']:
                    result += f" ({ch['word_count']:,} words)"
                result += "\n"

            result += f"\n💡 Use semantic_search() to find specific content in this book."

            logger.info(f"Generated metadata for book: {book['title']}")
            return result

        except ValidationError as e:
            return f"Validation error: {str(e)}"
        except Exception as e:
            logger.error(f"Error in metadata resource: {e}", exc_info=True)
            return f"Error: {str(e)}"

    # Curated reading collections for RAG context
    READING_COLLECTIONS = {
        'python-essentials': {
            'name': 'Python Essentials',
            'description': 'Core Python programming books covering fundamentals to advanced topics',
            'keywords': ['python', 'async', 'clean'],
        },
        'devops-stack': {
            'name': 'DevOps & Infrastructure',
            'description': 'Docker, Linux, networking, and system administration',
            'keywords': ['docker', 'linux', 'systemd', 'networking', 'ubuntu'],
        },
        'ai-ml-track': {
            'name': 'AI & Machine Learning',
            'description': 'Deep learning, LLMs, generative AI, and ML patterns',
            'keywords': ['deep learning', 'llm', 'ai', 'langchain', 'neural'],
        },
        'data-engineering': {
            'name': 'Data Engineering & Science',
            'description': 'Data processing, cleaning, analysis, and forecasting',
            'keywords': ['data', 'forecasting', 'analytics'],
        },
        'architecture': {
            'name': 'Software Architecture',
            'description': 'Clean architecture, design patterns, and best practices',
            'keywords': ['architecture', 'design', 'clean', 'patterns'],
        },
        'web-development': {
            'name': 'Web Development',
            'description': 'Frontend, backend, Node.js, and API development',
            'keywords': ['node', 'web', 'api', 'frontend'],
        },
    }

    @mcp.resource("collection://list")
    def list_collections() -> str:
        """Resource: List all available reading collections

        Returns overview of curated reading lists for different learning paths.
        Use collection://{name} to get books in a specific collection.
        """
        result = "📚 Reading Collections\n" + "="*60 + "\n\n"
        result += "Curated reading lists organized by topic and learning path.\n"
        result += "Access a collection with: collection://{collection-name}\n\n"

        for coll_id, coll in READING_COLLECTIONS.items():
            result += f"• {coll['name']} (collection://{coll_id})\n"
            result += f"  {coll['description']}\n\n"

        return result

    @mcp.resource("collection://{collection_name}")
    def get_collection(collection_name: str) -> str:
        """Resource: Get books in a curated reading collection

        Returns books matching the collection theme, with metadata and
        reading order suggestions. Perfect for context when asking
        "What should I read about X?" questions.

        Available collections:
        - python-essentials: Core Python programming
        - devops-stack: Docker, Linux, infrastructure
        - ai-ml-track: AI, ML, deep learning, LLMs
        - data-engineering: Data processing and analysis
        - architecture: Software design patterns
        - web-development: Web and API development
        """
        from .database import execute_query

        try:
            collection_name = collection_name.lower().strip()

            # Check if it's a valid collection
            if collection_name not in READING_COLLECTIONS:
                available = ", ".join(READING_COLLECTIONS.keys())
                return f"Collection '{collection_name}' not found.\n\nAvailable collections: {available}"

            collection = READING_COLLECTIONS[collection_name]

            # Get all books and filter by keywords
            books = execute_query("""
                SELECT
                    b.id,
                    b.title,
                    b.author,
                    b.word_count,
                    COUNT(c.id) as chapter_count,
                    SUM(CASE WHEN c.embedding IS NOT NULL THEN 1 ELSE 0 END) as embedded_chapters
                FROM books b
                LEFT JOIN chapters c ON c.book_id = b.id
                GROUP BY b.id
                ORDER BY b.title
            """)

            # Filter books matching collection keywords
            matching_books = []
            for book in books:
                title_lower = book['title'].lower()
                if any(kw in title_lower for kw in collection['keywords']):
                    matching_books.append(book)

            # Build output
            result = f"📚 {collection['name']}\n" + "="*60 + "\n\n"
            result += f"{collection['description']}\n\n"

            if not matching_books:
                result += "No books currently match this collection.\n"
                result += "Try adding books with relevant topics to your library.\n"
                return result

            result += f"📖 {len(matching_books)} Books in Collection\n" + "-"*60 + "\n"

            total_words = 0
            total_chapters = 0

            for i, book in enumerate(matching_books, 1):
                total_words += book['word_count'] or 0
                total_chapters += book['chapter_count']

                result += f"\n{i}. {book['title']}\n"
                if book['author']:
                    result += f"   Author: {book['author']}\n"
                result += f"   Chapters: {book['chapter_count']}, Words: {book['word_count']:,}\n"

                # Semantic search readiness
                if book['embedded_chapters'] == book['chapter_count']:
                    result += f"   Semantic Search: ✅ Ready\n"
                elif book['embedded_chapters'] > 0:
                    result += f"   Semantic Search: ⚠️ Partial ({book['embedded_chapters']}/{book['chapter_count']})\n"
                else:
                    result += f"   Semantic Search: ❌ Not indexed\n"

                result += f"   ID: {book['id']}\n"

            # Collection summary
            result += f"\n📊 Collection Summary\n" + "-"*60 + "\n"
            result += f"Total Books: {len(matching_books)}\n"
            result += f"Total Chapters: {total_chapters}\n"
            result += f"Total Words: {total_words:,}\n"
            est_hours = total_words / 15000  # ~250 words/min, 60 min
            result += f"Estimated Reading Time: {est_hours:.0f} hours\n"

            result += f"\n💡 Tips:\n"
            result += f"• Use semantic_search('topic') to find specific content\n"
            result += f"• Access book details with book://{{book_id}}/metadata\n"

            logger.info(f"Generated collection '{collection_name}' with {len(matching_books)} books")
            return result

        except Exception as e:
            logger.error(f"Error in collection resource: {e}", exc_info=True)
            return f"Error: {str(e)}"

    # =========================================================================
    # PROMPTS - Reusable workflows for common research patterns
    # =========================================================================

    @mcp.prompt()
    def research_topic(topic: str, depth: str = "overview") -> str:
        """Research a topic across your technical book library

        Creates a structured research workflow that searches across all books
        to find relevant content on any topic.

        Args:
            topic: The topic to research (e.g., "container networking", "async patterns")
            depth: Research depth - "overview" for quick summary, "deep-dive" for comprehensive

        Returns:
            A structured prompt for researching the topic using your library
        """
        from .database import execute_query

        # Get library stats for context
        books = execute_query("SELECT title, author FROM books ORDER BY title")
        book_list = "\n".join([f"- {b['title']}" + (f" by {b['author']}" if b['author'] else "") for b in books])

        if depth == "deep-dive":
            return f"""# Deep-Dive Research: {topic}

## Your Task
Conduct comprehensive research on "{topic}" using the available technical library.

## Available Resources
Your library contains {len(books)} books:
{book_list}

## Research Workflow

### Step 1: Semantic Search
Use `semantic_search("{topic}")` to find the most relevant chapters across all books.
Also try related terms and synonyms.

### Step 2: Gather Context
For each highly relevant result (similarity > 0.5):
- Note the book and chapter
- Read the full chapter using `get_chapter(book_id, chapter_number)`
- Extract key concepts, definitions, and examples

### Step 3: Cross-Reference
Look for:
- Different perspectives from different authors
- Complementary explanations
- Practical examples vs theoretical foundations
- Any contradictions or debates

### Step 4: Synthesize
Create a comprehensive summary that includes:
- Core concepts and definitions
- Key principles and patterns
- Practical applications
- Code examples (if applicable)
- Further reading recommendations from the library

## Output Format
Provide a well-structured research report with:
1. Executive Summary (2-3 sentences)
2. Key Concepts (bullet points)
3. Detailed Findings (organized by sub-topic)
4. Practical Applications
5. Source Citations (book title, chapter)
"""
        else:  # overview
            return f"""# Research Overview: {topic}

## Your Task
Provide a quick overview of "{topic}" using the available technical library.

## Available Resources
Your library contains {len(books)} technical books covering Python, DevOps, ML/AI, Architecture, and more.

## Research Workflow

### Step 1: Quick Search
Use `semantic_search("{topic}", limit=5)` to find the top relevant chapters.

### Step 2: Summarize Findings
For the top 2-3 results:
- Identify the main points
- Note which books cover this topic

### Step 3: Provide Overview
Create a concise summary (3-5 paragraphs) covering:
- What is {topic}?
- Why does it matter?
- Key concepts to understand
- Which books to read for more depth

## Output Format
- Brief introduction
- Key points (bullet list)
- Recommended reading from the library
"""

    @mcp.prompt()
    def explain_concept(concept: str, expertise_level: str = "intermediate") -> str:
        """Explain a technical concept using examples from your book library

        Creates a prompt that explains concepts at the appropriate level,
        drawing from your technical books for authoritative explanations.

        Args:
            concept: The concept to explain (e.g., "dependency injection", "docker volumes")
            expertise_level: Target audience - "beginner", "intermediate", or "advanced"

        Returns:
            A structured prompt for explaining the concept
        """
        level_context = {
            "beginner": {
                "tone": "simple, avoiding jargon, using analogies",
                "focus": "fundamental understanding and practical getting-started steps",
                "depth": "surface-level with clear examples",
                "prerequisites": "Assume no prior knowledge of this topic"
            },
            "intermediate": {
                "tone": "technical but accessible, explaining trade-offs",
                "focus": "practical application, common patterns, and best practices",
                "depth": "moderate with code examples and architectural considerations",
                "prerequisites": "Assume familiarity with basic programming concepts"
            },
            "advanced": {
                "tone": "deeply technical, discussing edge cases and optimizations",
                "focus": "advanced patterns, performance implications, and expert techniques",
                "depth": "comprehensive with advanced examples and internal workings",
                "prerequisites": "Assume strong foundation and production experience"
            }
        }

        ctx = level_context.get(expertise_level, level_context["intermediate"])

        return f"""# Explain Concept: {concept}

## Target Audience
Level: {expertise_level.upper()}
{ctx['prerequisites']}

## Your Task
Explain "{concept}" using your technical book library as the authoritative source.

## Explanation Guidelines
- **Tone**: {ctx['tone']}
- **Focus**: {ctx['focus']}
- **Depth**: {ctx['depth']}

## Workflow

### Step 1: Find Authoritative Sources
Use `semantic_search("{concept}")` to find chapters that explain this concept.

### Step 2: Gather Explanations
Read the relevant sections and note:
- How different authors define/explain it
- Examples they provide
- Common misconceptions they address

### Step 3: Synthesize Explanation
Create an explanation that:
- Defines the concept clearly
- Explains why it matters
- Provides concrete examples
- Addresses common questions/misconceptions
- Cites your sources

## Output Structure

### What is {concept}?
[Clear definition appropriate for {expertise_level} level]

### Why Does It Matter?
[Practical importance and use cases]

### How It Works
[Explanation with examples, code if relevant]

### Common Pitfalls
[Mistakes to avoid]

### Learn More
[Specific chapters/books from your library for deeper understanding]

---
*Sources: Cite specific books and chapters used*
"""

    @mcp.prompt()
    def create_reading_plan(
        goal: str,
        time_available: str = "flexible",
        current_knowledge: str = "some programming experience"
    ) -> str:
        """Create a personalized reading plan from your book library

        Generates a structured learning path drawing from your 17+ technical books,
        organized by priority and estimated reading time.

        Args:
            goal: Learning goal (e.g., "become proficient in Docker", "learn ML fundamentals")
            time_available: Time commitment (e.g., "2 hours/week", "intensive", "flexible")
            current_knowledge: Current skill level and background

        Returns:
            A structured prompt for creating a reading plan
        """
        from .database import execute_query, execute_single

        # Get library stats
        stats = execute_single("""
            SELECT
                (SELECT COUNT(*) FROM books) as book_count,
                (SELECT COUNT(*) FROM chapters) as chapter_count,
                (SELECT COALESCE(SUM(word_count), 0) FROM books) as total_words
        """)

        books = execute_query("""
            SELECT b.title, b.author, b.word_count, COUNT(c.id) as chapters
            FROM books b
            LEFT JOIN chapters c ON c.book_id = b.id
            GROUP BY b.id
            ORDER BY b.title
        """)

        book_summary = "\n".join([
            f"- {b['title']} ({b['chapters']} chapters, ~{b['word_count']//250} min read)"
            for b in books
        ])

        return f"""# Create Reading Plan

## Learning Goal
"{goal}"

## Learner Profile
- **Current Knowledge**: {current_knowledge}
- **Time Available**: {time_available}

## Your Library
{stats['book_count']} books, {stats['chapter_count']} chapters, {stats['total_words']:,} total words

### Available Books:
{book_summary}

## Your Task
Create a personalized reading plan to achieve the stated goal.

## Planning Workflow

### Step 1: Identify Relevant Books
Use `semantic_search("{goal}")` to find the most relevant content.
Review `collection://list` to see curated learning paths.

### Step 2: Assess Prerequisites
Based on "{current_knowledge}", determine:
- What foundational chapters should come first
- What can be skipped
- What requires extra attention

### Step 3: Create Structured Plan
Organize reading into phases:

## Reading Plan Structure

### Phase 1: Foundations (Week 1-2)
*Build core understanding*
- [ ] Book: [title] - Chapters X-Y (estimated time)
- [ ] Book: [title] - Chapter Z (estimated time)
Key concepts to master: [list]

### Phase 2: Core Skills (Week 3-4)
*Develop practical abilities*
- [ ] Book: [title] - Chapters X-Y
- [ ] Hands-on: [suggested practice]

### Phase 3: Advanced Topics (Week 5+)
*Deepen expertise*
- [ ] Book: [title] - Chapters X-Y
- [ ] Project: [suggested application]

### Quick Reference
Keep bookmarked for ongoing reference:
- [specific chapters for common tasks]

## Time Estimates
- Total reading time: [X hours]
- Recommended pace: [X chapters/week]
- Expected completion: [timeframe]

## Success Metrics
How to know you've achieved "{goal}":
- [ ] Can explain [concept] clearly
- [ ] Can implement [skill] independently
- [ ] Have completed [project/exercise]
"""

    @mcp.prompt()
    def compare_approaches(topic: str, book_ids: str = "") -> str:
        """Compare how different authors approach a topic

        Creates a side-by-side comparison of different perspectives,
        methodologies, or implementations from various books.

        Args:
            topic: Topic to compare (e.g., "error handling", "API design")
            book_ids: Optional comma-separated book IDs to focus on (empty = search all)

        Returns:
            A structured prompt for comparing approaches
        """
        from .database import execute_query

        books = execute_query("SELECT id, title, author FROM books ORDER BY title")

        if book_ids:
            id_list = [bid.strip() for bid in book_ids.split(",")]
            filtered = [b for b in books if b['id'] in id_list]
            book_context = "\n".join([f"- {b['title']} by {b['author']}" for b in filtered])
            search_scope = f"Focus on these {len(filtered)} books:\n{book_context}"
        else:
            search_scope = f"Search across all {len(books)} books in your library"

        return f"""# Compare Approaches: {topic}

## Your Task
Analyze and compare how different authors approach "{topic}".

## Scope
{search_scope}

## Comparison Workflow

### Step 1: Gather Perspectives
Use `semantic_search("{topic}")` to find relevant chapters.
For each result, note:
- The book/author
- Their approach or methodology
- Key principles they emphasize

### Step 2: Identify Dimensions
Determine comparison criteria:
- Theoretical foundation
- Practical implementation
- Trade-offs discussed
- Use cases recommended
- Code style/patterns

### Step 3: Analyze Differences
For each dimension:
- How do approaches differ?
- What assumptions underlie each approach?
- What contexts favor each approach?

### Step 4: Synthesize Insights
- Where do authors agree?
- Where do they diverge? Why?
- What can we learn from the differences?

## Output Format

### Overview
Brief summary of the different approaches found.

### Comparison Matrix

| Aspect | [Book 1] | [Book 2] | [Book 3] |
|--------|----------|----------|----------|
| Core Philosophy | | | |
| Key Principles | | | |
| Implementation | | | |
| Trade-offs | | | |
| Best For | | | |

### Author Perspectives

#### [Author 1] - [Book Title]
- **Approach**: [summary]
- **Key Quote**: "[relevant quote]"
- **Strengths**: [what this approach does well]
- **Limitations**: [where it falls short]

#### [Author 2] - [Book Title]
[Same structure]

### Synthesis

#### Points of Agreement
- [Where authors align]

#### Points of Divergence
- [Where they differ and why]

#### Recommendations
Based on this analysis:
- For [use case A], consider [approach X] because...
- For [use case B], consider [approach Y] because...

---
*Sources: All comparisons cite specific chapters*
"""

    logger.info(f"Server '{Config.SERVER_NAME}' v{Config.SERVER_VERSION} ready")
    return mcp

def main():
    """Main entry point"""
    try:
        mcp = create_server()
        mcp.run()
    except Exception as e:
        logger.critical(f"Server failed to start: {e}")
        raise

if __name__ == "__main__":
    main()

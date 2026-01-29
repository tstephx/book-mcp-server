#!/usr/bin/env python3
"""Test the three new features: FTS, Batch Ops, and Summaries"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.database import get_db_connection, execute_query

def test_fts_search():
    """Test full-text search"""
    print("\n=== Testing Full-Text Search ===")

    from src.utils.fts_search import full_text_search, fts_table_exists

    # Check if FTS table exists
    if not fts_table_exists():
        print("ERROR: FTS table not found")
        return False

    print("FTS table exists")

    # Test simple search
    result = full_text_search("async")
    print(f"\nSearch 'async': {result['total_found']} results")
    if result['results']:
        for r in result['results'][:3]:
            print(f"  - {r['book_title']}: Ch {r['chapter_number']} ({r['rank']:.3f})")
            if r.get('excerpt'):
                print(f"    Excerpt: {r['excerpt'][:100]}...")

    # Test phrase search
    result = full_text_search('"dependency injection"')
    print(f"\nPhrase search 'dependency injection': {result['total_found']} results")
    if result['results']:
        print(f"  First result: {result['results'][0]['book_title']}")

    # Test with book filter
    books = execute_query("SELECT id FROM books LIMIT 1")
    if books:
        result = full_text_search("chapter", book_id=books[0]['id'], limit=3)
        print(f"\nFiltered search in '{books[0]['id']}': {result['total_found']} results")

    print("\nFTS: PASSED")
    return True


def test_batch_operations():
    """Test batch operations"""
    print("\n=== Testing Batch Operations ===")

    from src.utils.batch_ops import batch_semantic_search, get_library_statistics

    # Test library statistics
    stats = get_library_statistics()
    print(f"\nLibrary Statistics:")
    print(f"  Books: {stats.get('books', 'N/A')}")
    print(f"  Chapters: {stats.get('chapters', 'N/A')}")
    print(f"  Total Words: {stats.get('total_words', 'N/A'):,}")
    print(f"  Embedding Coverage: {stats.get('embedding_coverage', 'N/A')}%")
    print(f"  FTS Indexed: {stats.get('fts_indexed', 'N/A')}")

    # Test batch semantic search
    print("\nBatch Semantic Search 'error handling'...")
    result = batch_semantic_search("error handling", max_per_book=2)

    print(f"  Books searched: {result['books_searched']}")
    print(f"  Total results: {result['total_results']}")

    if result['results_by_book']:
        for book_result in result['results_by_book'][:3]:
            print(f"\n  {book_result['book_title']}:")
            for ch in book_result['chapters'][:2]:
                print(f"    - Ch {ch['chapter_number']}: {ch['chapter_title']} ({ch['similarity']:.3f})")

    print("\nBatch Operations: PASSED")
    return True


def test_summaries():
    """Test chapter summaries"""
    print("\n=== Testing Chapter Summaries ===")

    from src.utils.summaries import generate_chapter_summary, extract_summary

    # Get a chapter to test with
    chapters = execute_query("""
        SELECT c.id, c.title, c.file_path, b.title as book_title
        FROM chapters c
        JOIN books b ON c.book_id = b.id
        WHERE c.word_count > 500
        LIMIT 1
    """)

    if not chapters:
        print("No suitable chapters found for testing")
        return False

    chapter = chapters[0]
    print(f"Testing with: {chapter['book_title']} - {chapter['title']}")

    # Test summary generation
    result = generate_chapter_summary(chapter['id'])

    if 'error' in result:
        print(f"ERROR: {result['error']}")
        return False

    print(f"\nStatus: {result.get('status', 'N/A')}")
    print(f"Summary preview:\n{result['summary'][:400]}...")

    # Test caching - second call should be faster
    import time
    start = time.time()
    result2 = generate_chapter_summary(chapter['id'])
    elapsed = time.time() - start
    print(f"\nCached retrieval: {elapsed:.3f}s (status: {result2.get('status', 'N/A')})")

    print("\nSummaries: PASSED")
    return True


def main():
    print("=" * 60)
    print("Testing New Features: FTS, Batch Ops, Summaries")
    print("=" * 60)

    results = {
        'fts': test_fts_search(),
        'batch': test_batch_operations(),
        'summaries': test_summaries()
    }

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed."))

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

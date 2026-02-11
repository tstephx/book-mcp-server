"""
Chapter Summary Generation

Provides extractive summarization for chapters:
- First paragraph as intro
- Key sentences using position and content heuristics
"""

import io
import logging
import re
from datetime import datetime
from typing import Optional, List

import numpy as np

from ..database import get_db_connection, execute_query
from .context_managers import embedding_model_context
from .file_utils import read_chapter_content

logger = logging.getLogger(__name__)


def extract_summary(
    content: str,
    max_sentences: int = 5,
    include_intro: bool = True
) -> str:
    """Extract summary from chapter content using extractive method

    Strategy:
    1. Take first paragraph as intro (if include_intro)
    2. Score remaining sentences by:
       - Position (earlier sentences score higher)
       - Length (prefer medium-length sentences)
       - Contains key markers (definitions, conclusions)

    Args:
        content: Full chapter text
        max_sentences: Maximum sentences in summary (default: 5)
        include_intro: Include first paragraph (default: True)

    Returns:
        Extracted summary text
    """
    if not content or not content.strip():
        return ""

    # Split into paragraphs
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

    if not paragraphs:
        return ""

    summary_parts = []
    remaining_sentences = max_sentences

    # Extract intro paragraph
    if include_intro and paragraphs:
        intro = paragraphs[0]
        # Skip if intro looks like a header or is too short
        if len(intro) > 50 and not intro.startswith('#'):
            summary_parts.append(intro)
            # Count sentences used
            intro_sentences = len(_split_sentences(intro))
            remaining_sentences = max(1, max_sentences - intro_sentences)
            paragraphs = paragraphs[1:]

    if remaining_sentences <= 0 or not paragraphs:
        return '\n\n'.join(summary_parts)

    # Collect and score all sentences from remaining paragraphs
    all_sentences = []
    for para_idx, para in enumerate(paragraphs):
        # Skip headers and very short paragraphs
        if para.startswith('#') or len(para) < 30:
            continue

        sentences = _split_sentences(para)
        for sent_idx, sentence in enumerate(sentences):
            if len(sentence) < 20:  # Skip very short sentences
                continue

            score = _score_sentence(sentence, para_idx, sent_idx, len(paragraphs))
            all_sentences.append((score, sentence, para_idx))

    # Sort by score (descending) and take top N
    all_sentences.sort(key=lambda x: x[0], reverse=True)
    selected = all_sentences[:remaining_sentences]

    # Re-sort selected by original position for coherent reading
    selected.sort(key=lambda x: (x[2], all_sentences.index(x)))

    summary_parts.extend([sent for _, sent, _ in selected])

    return '\n\n'.join(summary_parts)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences"""
    # Simple sentence splitter - handles common cases
    # Split on .!? followed by space and capital letter
    pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def _score_sentence(
    sentence: str,
    para_idx: int,
    sent_idx: int,
    total_paras: int
) -> float:
    """Score sentence for summary inclusion

    Higher scores = more likely to include

    Factors:
    - Position: Earlier paragraphs score higher
    - Sentence position: First sentences of paragraphs score higher
    - Length: Medium-length sentences preferred
    - Content markers: Definitions, key phrases boost score
    """
    score = 0.0

    # Position scoring (earlier is better)
    position_score = 1.0 - (para_idx / max(total_paras, 1))
    score += position_score * 3.0

    # First sentence of paragraph bonus
    if sent_idx == 0:
        score += 2.0

    # Length scoring (prefer 50-200 chars)
    length = len(sentence)
    if 50 <= length <= 200:
        score += 1.5
    elif 30 <= length <= 250:
        score += 0.5
    else:
        score -= 0.5

    # Content markers
    lower_sent = sentence.lower()

    # Definition patterns
    if any(marker in lower_sent for marker in [' is ', ' are ', ' means ', ' refers to ']):
        score += 1.0

    # Conclusion/summary patterns
    if any(marker in lower_sent for marker in ['in summary', 'in conclusion', 'importantly', 'key ', 'essential']):
        score += 1.5

    # Example patterns (slightly lower, but still useful)
    if any(marker in lower_sent for marker in ['for example', 'for instance', 'such as']):
        score += 0.5

    # Technical content indicators
    if any(marker in lower_sent for marker in ['function', 'method', 'class', 'pattern', 'approach']):
        score += 0.5

    return score


def generate_chapter_summary(chapter_id: str, force: bool = False) -> dict:
    """Generate and store summary for a chapter

    Args:
        chapter_id: Chapter ID
        force: Regenerate even if summary exists

    Returns:
        Dictionary with summary and metadata
    """
    # Check for existing summary
    if not force:
        existing = execute_query(
            "SELECT summary FROM chapter_summaries WHERE chapter_id = ?",
            (chapter_id,)
        )
        if existing and existing[0]['summary']:
            return {
                'chapter_id': chapter_id,
                'summary': existing[0]['summary'],
                'status': 'cached'
            }

    # Get chapter info
    chapter = execute_query(
        """SELECT c.id, c.title, c.file_path, c.chapter_number, b.title as book_title
           FROM chapters c
           JOIN books b ON c.book_id = b.id
           WHERE c.id = ?""",
        (chapter_id,)
    )

    if not chapter:
        return {'error': f'Chapter not found: {chapter_id}'}

    chapter = chapter[0]

    # Read content
    try:
        content = read_chapter_content(chapter['file_path'])
    except Exception as e:
        return {'error': f'Failed to read chapter: {e}'}

    # Generate summary
    summary = extract_summary(content, max_sentences=5)

    if not summary:
        return {'error': 'Could not generate summary (chapter may be too short)'}

    # Store in database
    now = datetime.now().isoformat()
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO chapter_summaries
                (chapter_id, summary, generated_at)
                VALUES (?, ?, ?)
            """, (chapter_id, summary, now))
            conn.commit()
    except Exception as e:
        logger.warning(f"Failed to store summary: {e}")
        # Return summary anyway, just not cached

    return {
        'chapter_id': chapter_id,
        'chapter_number': chapter['chapter_number'],
        'chapter_title': chapter['title'],
        'book_title': chapter['book_title'],
        'summary': summary,
        'status': 'generated'
    }


def get_chapter_summary(chapter_id: str) -> Optional[dict]:
    """Get existing summary for a chapter

    Args:
        chapter_id: Chapter ID

    Returns:
        Summary dict or None if not found
    """
    row = execute_query(
        """SELECT cs.chapter_id, cs.summary, cs.generated_at,
                  c.title, c.chapter_number, b.title as book_title
           FROM chapter_summaries cs
           JOIN chapters c ON cs.chapter_id = c.id
           JOIN books b ON c.book_id = b.id
           WHERE cs.chapter_id = ?""",
        (chapter_id,)
    )

    if not row:
        return None

    row = row[0]
    return {
        'chapter_id': row['chapter_id'],
        'chapter_number': row['chapter_number'],
        'chapter_title': row['title'],
        'book_title': row['book_title'],
        'summary': row['summary'],
        'generated_at': row['generated_at']
    }


def generate_book_summaries(book_id: str, force: bool = False) -> dict:
    """Generate summaries for all chapters in a book

    Args:
        book_id: Book ID
        force: Regenerate all summaries

    Returns:
        Summary generation results
    """
    chapters = execute_query(
        """SELECT id, title, chapter_number
           FROM chapters
           WHERE book_id = ?
           ORDER BY chapter_number""",
        (book_id,)
    )

    if not chapters:
        return {'error': f'No chapters found for book: {book_id}'}

    results = {
        'book_id': book_id,
        'total': len(chapters),
        'generated': 0,
        'cached': 0,
        'errors': []
    }

    for chapter in chapters:
        result = generate_chapter_summary(chapter['id'], force=force)

        if 'error' in result:
            results['errors'].append({
                'chapter_id': chapter['id'],
                'title': chapter['title'],
                'error': result['error']
            })
        elif result.get('status') == 'cached':
            results['cached'] += 1
        else:
            results['generated'] += 1

    logger.info(
        f"Book {book_id} summaries: {results['generated']} generated, "
        f"{results['cached']} cached, {len(results['errors'])} errors"
    )

    return results


def generate_summary_embedding(chapter_id: str, generator) -> bool:
    """Generate and store embedding for an existing summary.

    Args:
        chapter_id: Chapter ID with an existing summary
        generator: EmbeddingGenerator instance (model already loaded)

    Returns:
        True if embedding was generated and stored successfully
    """
    # Read summary from DB
    rows = execute_query(
        "SELECT summary FROM chapter_summaries WHERE chapter_id = ?",
        (chapter_id,)
    )
    if not rows or not rows[0]['summary']:
        logger.warning(f"No summary found for chapter {chapter_id}")
        return False

    summary_text = rows[0]['summary']

    # Generate embedding
    try:
        embedding = generator.generate(summary_text)
    except Exception as e:
        logger.error(f"Failed to generate embedding for {chapter_id}: {e}")
        return False

    # Serialize to BLOB
    buf = io.BytesIO()
    np.save(buf, embedding)
    embedding_blob = buf.getvalue()

    model_name = getattr(generator, 'model_name', 'all-MiniLM-L6-v2')

    # Store in DB
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE chapter_summaries
                SET embedding = ?, embedding_model = ?
                WHERE chapter_id = ?
            """, (embedding_blob, model_name, chapter_id))
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to store summary embedding for {chapter_id}: {e}")
        return False

    return True


def batch_generate_summary_embeddings(force: bool = False) -> dict:
    """Generate embeddings for all summaries that don't have one.

    Args:
        force: If True, regenerate all embeddings (not just missing ones)

    Returns:
        Stats dict: {generated, skipped, errors, total}
    """
    # Count total summaries and those already with embeddings
    all_summaries = execute_query(
        "SELECT chapter_id FROM chapter_summaries WHERE summary IS NOT NULL"
    )
    total = len(all_summaries)

    if force:
        rows = all_summaries
        skipped = 0
    else:
        rows = execute_query(
            "SELECT chapter_id FROM chapter_summaries WHERE summary IS NOT NULL AND embedding IS NULL"
        )
        skipped = total - len(rows)

    if not rows:
        return {'generated': 0, 'skipped': skipped, 'errors': 0, 'total': total, 'status': 'no_updates_needed'}

    generated = 0
    errors = 0

    with embedding_model_context() as generator:
        for row in rows:
            if generate_summary_embedding(row['chapter_id'], generator):
                generated += 1
            else:
                errors += 1

            if generated % 50 == 0 and generated > 0:
                logger.info(f"Generated {generated}/{total} summary embeddings...")

    logger.info(f"Summary embeddings: {generated} generated, {skipped} skipped, {errors} errors")

    return {
        'generated': generated,
        'skipped': skipped,
        'errors': errors,
        'total': total,
        'status': 'updated' if generated > 0 else 'no_updates_needed'
    }

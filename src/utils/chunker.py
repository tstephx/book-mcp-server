"""Paragraph-grouping chunker for sub-chapter retrieval.

Splits chapter text into ~500-word chunks along paragraph boundaries,
with one-paragraph overlap between adjacent chunks for context continuity.
"""

import re


def chunk_chapter(
    text: str,
    target_words: int = 500,
    min_chunk_words: int = 100,
) -> list[dict]:
    """Split chapter text into chunks along paragraph boundaries.

    Args:
        text: Full chapter text.
        target_words: Target words per chunk (~500).
        min_chunk_words: Minimum words for a standalone chunk.

    Returns:
        List of dicts with keys: chunk_index, content, word_count.
    """
    text = text.strip()
    if not text:
        return []

    total_words = len(text.split())

    if total_words <= int(target_words * 1.2):
        return [{"chunk_index": 0, "content": text, "word_count": total_words}]

    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return [{"chunk_index": 0, "content": text, "word_count": total_words}]

    chunks = []
    current_paragraphs: list[str] = []
    current_words = 0
    prev_last_paragraph: str | None = None

    for para in paragraphs:
        para_words = len(para.split())

        if current_words + para_words > target_words and current_words >= min_chunk_words:
            chunks.append(_make_chunk(len(chunks), current_paragraphs))

            prev_last_paragraph = current_paragraphs[-1]
            current_paragraphs = [prev_last_paragraph]
            current_words = len(prev_last_paragraph.split())

        current_paragraphs.append(para)
        current_words += para_words

    if current_paragraphs:
        if chunks and current_words < min_chunk_words:
            if prev_last_paragraph and current_paragraphs[0] == prev_last_paragraph:
                current_paragraphs = current_paragraphs[1:]
            if current_paragraphs:
                last = chunks[-1]
                merged_content = last["content"] + "\n\n" + "\n\n".join(current_paragraphs)
                chunks[-1] = _make_chunk(last["chunk_index"], None, merged_content)
        else:
            chunks.append(_make_chunk(len(chunks), current_paragraphs))

    return chunks


def _split_paragraphs(text: str) -> list[str]:
    """Split text on double-newlines, filtering blanks."""
    raw = re.split(r"\n\n+", text)
    return [p.strip() for p in raw if p.strip()]


def _make_chunk(
    index: int,
    paragraphs: list[str] | None = None,
    content: str | None = None,
) -> dict:
    if content is None:
        content = "\n\n".join(paragraphs)
    return {
        "chunk_index": index,
        "content": content,
        "word_count": len(content.split()),
    }

"""Paragraph-grouping chunker for sub-chapter retrieval.

Splits chapter text into ~500-word chunks along paragraph boundaries,
with one-paragraph overlap between adjacent chunks for context continuity.
Enforces a hard token limit using tiktoken to prevent API truncation.
"""

import re
import tiktoken

# Lazy-loaded encoding
_encoding = None


def _get_encoding():
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.encoding_for_model("text-embedding-3-large")
    return _encoding


def _count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(_get_encoding().encode(text))


def _split_at_sentences(text: str, max_tokens: int) -> list[str]:
    """Split text at sentence boundaries to stay under max_tokens.

    Args:
        text: Text to split
        max_tokens: Maximum tokens per chunk

    Returns:
        List of sentence-split chunks, each under max_tokens
    """
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = _count_tokens(sentence)

        # If single sentence exceeds limit, hard truncate
        if sentence_tokens > max_tokens:
            if current:
                chunks.append(' '.join(current))
                current = []
                current_tokens = 0

            # Truncate to fit
            enc = _get_encoding()
            tokens = enc.encode(sentence)[:max_tokens]
            chunks.append(enc.decode(tokens))
            continue

        if current_tokens + sentence_tokens > max_tokens and current:
            chunks.append(' '.join(current))
            current = [sentence]
            current_tokens = sentence_tokens
        else:
            current.append(sentence)
            current_tokens += sentence_tokens

    if current:
        chunks.append(' '.join(current))

    return chunks


def chunk_chapter(
    text: str,
    target_words: int = 500,
    min_chunk_words: int = 100,
    max_tokens: int = 8000,
) -> list[dict]:
    """Split chapter text into chunks along paragraph boundaries.

    Args:
        text: Full chapter text.
        target_words: Target words per chunk (~500).
        min_chunk_words: Minimum words for a standalone chunk.
        max_tokens: Hard token limit (default: 8000, buffer below 8191 API limit).

    Returns:
        List of dicts with keys: chunk_index, content, word_count, token_count.
    """
    text = text.strip()
    if not text:
        return []

    total_words = len(text.split())

    if total_words <= int(target_words * 1.2):
        token_count = _count_tokens(text)
        if token_count > max_tokens:
            # Single-chunk chapter exceeds token limit, split it
            sub_chunks = _split_at_sentences(text, max_tokens)
            return [
                {
                    "chunk_index": i,
                    "content": chunk,
                    "word_count": len(chunk.split()),
                    "token_count": _count_tokens(chunk)
                }
                for i, chunk in enumerate(sub_chunks)
            ]
        return [{
            "chunk_index": 0,
            "content": text,
            "word_count": total_words,
            "token_count": token_count
        }]

    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        token_count = _count_tokens(text)
        if token_count > max_tokens:
            sub_chunks = _split_at_sentences(text, max_tokens)
            return [
                {
                    "chunk_index": i,
                    "content": chunk,
                    "word_count": len(chunk.split()),
                    "token_count": _count_tokens(chunk)
                }
                for i, chunk in enumerate(sub_chunks)
            ]
        return [{
            "chunk_index": 0,
            "content": text,
            "word_count": total_words,
            "token_count": token_count
        }]

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

    # Post-process: split any chunks that exceed max_tokens
    final_chunks = []
    for chunk in chunks:
        if chunk["token_count"] > max_tokens:
            # Split oversized chunk at sentence boundaries
            sub_chunks = _split_at_sentences(chunk["content"], max_tokens)
            for i, sub_content in enumerate(sub_chunks):
                final_chunks.append({
                    "chunk_index": len(final_chunks),
                    "content": sub_content,
                    "word_count": len(sub_content.split()),
                    "token_count": _count_tokens(sub_content)
                })
        else:
            # Re-index to maintain sequential indices
            chunk["chunk_index"] = len(final_chunks)
            final_chunks.append(chunk)

    return final_chunks


def _split_paragraphs(text: str) -> list[str]:
    """Split text on double-newlines, filtering blanks."""
    raw = re.split(r"\n\n+", text)
    return [p.strip() for p in raw if p.strip()]


def _make_chunk(
    index: int,
    paragraphs: list[str] | None = None,
    content: str | None = None,
) -> dict:
    """Create a chunk dict with token count.

    Args:
        index: Chunk index
        paragraphs: List of paragraphs to join (if content is None)
        content: Pre-joined content string

    Returns:
        Chunk dict with chunk_index, content, word_count, token_count
    """
    if content is None:
        content = "\n\n".join(paragraphs)

    token_count = _count_tokens(content)

    return {
        "chunk_index": index,
        "content": content,
        "word_count": len(content.split()),
        "token_count": token_count,
    }

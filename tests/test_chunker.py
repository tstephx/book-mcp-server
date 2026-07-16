"""Tests for paragraph-grouping chunker."""

from src.utils.chunker import chunk_chapter, sentence_windows


class TestChunkChapter:
    def test_short_chapter_single_chunk(self):
        """Chapters under 600 words become a single chunk."""
        text = "This is a short chapter. " * 20  # ~100 words
        chunks = chunk_chapter(text, target_words=500)
        assert len(chunks) == 1
        assert chunks[0]["chunk_index"] == 0
        assert chunks[0]["content"] == text.strip()
        assert chunks[0]["word_count"] > 0

    def test_long_chapter_multiple_chunks(self):
        """Long chapters split into ~500-word chunks."""
        paragraphs = []
        for i in range(5):
            paragraphs.append(f"Paragraph {i}. " + "word " * 199)
        text = "\n\n".join(paragraphs)

        chunks = chunk_chapter(text, target_words=500)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert chunk["word_count"] <= 700

    def test_overlap_includes_last_paragraph(self):
        """Adjacent chunks share a boundary paragraph for context."""
        paragraphs = []
        for i in range(6):
            paragraphs.append(f"Topic {i}. " + "detail " * 149)
        text = "\n\n".join(paragraphs)

        chunks = chunk_chapter(text, target_words=400)
        assert len(chunks) >= 2
        chunk_0_last_line = chunks[0]["content"].split("\n\n")[-1]
        assert chunk_0_last_line in chunks[1]["content"]

    def test_empty_input(self):
        """Empty or whitespace-only input returns empty list."""
        assert chunk_chapter("") == []
        assert chunk_chapter("   \n\n  ") == []

    def test_chunk_fields(self):
        """Each chunk dict has required fields."""
        text = "Hello world. " * 50
        chunks = chunk_chapter(text, target_words=500)
        for chunk in chunks:
            assert "chunk_index" in chunk
            assert "content" in chunk
            assert "word_count" in chunk
            assert isinstance(chunk["chunk_index"], int)
            assert isinstance(chunk["word_count"], int)
            assert len(chunk["content"]) > 0

    def test_no_empty_chunks(self):
        """Chunker never produces empty chunks."""
        text = "\n\n".join(["Short."] * 3 + ["Long paragraph. " * 100] + ["Short."] * 3)
        chunks = chunk_chapter(text, target_words=500)
        for chunk in chunks:
            assert chunk["word_count"] > 0
            assert chunk["content"].strip() != ""

    def test_token_count_added(self):
        """All chunks include token_count field."""
        text = "This is a test. " * 100
        chunks = chunk_chapter(text, target_words=500)
        for chunk in chunks:
            assert "token_count" in chunk
            assert isinstance(chunk["token_count"], int)
            assert chunk["token_count"] > 0

    def test_single_paragraph_exceeds_token_limit(self):
        """Single oversized paragraph gets split at sentence boundaries."""
        # Create a single paragraph with many sentences that exceeds 8000 tokens
        sentence = "This is a sentence with many words to inflate token count. " * 20
        paragraph = sentence * 80  # ~96K tokens worth
        chunks = chunk_chapter(paragraph, target_words=500, max_tokens=1000)

        # Should split into multiple chunks
        assert len(chunks) > 1
        # Each chunk should be under token limit
        for chunk in chunks:
            assert chunk["token_count"] <= 1000

    def test_code_block_as_single_paragraph(self):
        """Large code block (single paragraph) gets split if needed."""
        code = "def function():\n    return 42\n" * 500  # Large code block
        chunks = chunk_chapter(code, target_words=500, max_tokens=2000)

        # Should handle gracefully
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk["token_count"] <= 2000

    def test_merged_chunk_stays_under_limit(self):
        """Final merged chunk respects token limit."""
        # Create scenario where final chunk gets merged
        paragraphs = ["Para " + "word " * 50] * 3 + ["Tiny"]
        text = "\n\n".join(paragraphs)
        chunks = chunk_chapter(text, target_words=200, max_tokens=5000)

        for chunk in chunks:
            assert chunk["token_count"] <= 5000


def _numbered_sentences(n: int) -> str:
    """n sentences, 7 words each, individually identifiable."""
    return " ".join(f"This is numbered sentence {i:04d} padded body." for i in range(1, n + 1))


class TestSentenceWindows:
    def test_wall_of_text_produces_sized_windows(self):
        # ~2,520 words, zero double-newlines
        text = _numbered_sentences(360)
        windows = sentence_windows(text, target_words=500, overlap_sentences=2, min_chunk_words=100)
        assert len(windows) >= 4
        for w in windows:
            assert 300 <= len(w.split()) <= 700, f"window out of range: {len(w.split())} words"

    def test_consecutive_windows_overlap_by_two_sentences(self):
        import re as _re

        def _nums(s):
            return _re.findall(r"numbered sentence (\d{4})", s)

        text = _numbered_sentences(200)
        windows = sentence_windows(text, target_words=500, overlap_sentences=2, min_chunk_words=100)
        assert len(windows) >= 2
        for prev, nxt in zip(windows, windows[1:]):
            # window i+1 opens with the last 2 sentences of window i
            assert _nums(nxt)[:2] == _nums(prev)[-2:]

    def test_short_final_window_merges_into_predecessor(self):
        # 52 sentences x 7 words = 364 words... need >target to split: use target=300
        text = _numbered_sentences(52)
        windows = sentence_windows(text, target_words=300, overlap_sentences=2, min_chunk_words=100)
        # tail after the first window is < 100 new words -> merged, single window
        assert len(windows) == 1
        assert "0052" in windows[0]

    def test_single_giant_sentence_yields_one_window(self):
        text = " ".join(["word"] * 1500)  # no sentence boundaries at all
        windows = sentence_windows(text, target_words=500, overlap_sentences=2, min_chunk_words=100)
        assert len(windows) == 1

    def test_empty_text_returns_empty(self):
        assert sentence_windows("") == []
        assert sentence_windows("   \n  ") == []

    def test_coverage_no_content_lost(self):
        text = _numbered_sentences(300)
        windows = sentence_windows(text, target_words=500, overlap_sentences=2, min_chunk_words=100)
        joined = " ".join(windows)
        for i in range(1, 301):
            assert f"{i:04d}" in joined

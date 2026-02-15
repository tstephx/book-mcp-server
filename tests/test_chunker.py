"""Tests for paragraph-grouping chunker."""

from src.utils.chunker import chunk_chapter


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

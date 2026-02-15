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

"""Tests that get_chapter and get_section apply a default token cap."""
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys


LONG_SECTION_CONTENT = "word " * 10000  # ~10000 tokens, over DEFAULT_CHAPTER_TOKENS (8000)


def _register_tools(mcp):
    """Import and register chapter tools, returning captured tool functions."""
    captured = {}

    def tool_decorator(func):
        captured[func.__name__] = func
        return func

    mcp.tool.return_value = tool_decorator

    from src.tools.chapter_tools import register_chapter_tools
    register_chapter_tools(mcp)
    return captured


def test_get_section_truncates_by_default(tmp_path):
    """get_section should truncate to DEFAULT_CHAPTER_TOKENS when max_tokens is not given."""
    chapter_dir = tmp_path / "ch01"
    chapter_dir.mkdir()
    section_file = chapter_dir / "01-section.md"
    section_file.write_text(LONG_SECTION_CONTENT)

    mcp = MagicMock()
    captured = _register_tools(mcp)

    with patch("src.tools.chapter_tools.execute_single") as mock_db, \
         patch("src.tools.chapter_tools._find_chapter_path") as mock_path, \
         patch("src.tools.chapter_tools.validate_book_id", return_value="bookid123"), \
         patch("src.tools.chapter_tools.validate_chapter_number", return_value=1):

        mock_db.return_value = {"title": "Ch1", "file_path": str(chapter_dir / "ch01.md"), "word_count": 10000}
        mock_path.return_value = (chapter_dir, True)  # is_folder=True

        result = captured["get_section"]("bookid123", 1, 1)  # no max_tokens
        assert "truncated" in result.lower(), (
            f"Expected truncation note in result, but got {len(result)} chars with no truncation notice. "
            f"Content preview: {result[:200]}"
        )


def test_get_section_explicit_max_tokens_overrides_default(tmp_path):
    """Explicit max_tokens=500 should produce a very short result."""
    chapter_dir = tmp_path / "ch01"
    chapter_dir.mkdir()
    section_file = chapter_dir / "01-section.md"
    section_file.write_text(LONG_SECTION_CONTENT)

    mcp = MagicMock()
    captured = _register_tools(mcp)

    with patch("src.tools.chapter_tools.execute_single") as mock_db, \
         patch("src.tools.chapter_tools._find_chapter_path") as mock_path, \
         patch("src.tools.chapter_tools.validate_book_id", return_value="bookid123"), \
         patch("src.tools.chapter_tools.validate_chapter_number", return_value=1):

        mock_db.return_value = {"title": "Ch1", "file_path": str(chapter_dir / "ch01.md"), "word_count": 10000}
        mock_path.return_value = (chapter_dir, True)

        result = captured["get_section"]("bookid123", 1, 1, max_tokens=500)
        assert len(result) < 8000, f"Expected result < 8000 chars with max_tokens=500, got {len(result)}"


def test_get_chapter_truncates_by_default(tmp_path):
    """get_chapter (folder with _index.md) should truncate to DEFAULT_CHAPTER_TOKENS by default."""
    chapter_dir = tmp_path / "ch01"
    chapter_dir.mkdir()
    index_file = chapter_dir / "_index.md"
    index_file.write_text(LONG_SECTION_CONTENT)

    mcp = MagicMock()
    captured = _register_tools(mcp)

    with patch("src.tools.chapter_tools.execute_single") as mock_db, \
         patch("src.tools.chapter_tools._find_chapter_path") as mock_path, \
         patch("src.tools.chapter_tools.validate_book_id", return_value="bookid123"), \
         patch("src.tools.chapter_tools.validate_chapter_number", return_value=1):

        mock_db.return_value = {"title": "Ch1", "file_path": str(chapter_dir / "ch01.md"), "word_count": 10000}
        mock_path.return_value = (chapter_dir, True)

        result = captured["get_chapter"]("bookid123", 1)  # no max_tokens
        assert "truncated" in result.lower(), (
            f"Expected truncation note in result, but got {len(result)} chars with no truncation notice. "
            f"Content preview: {result[:200]}"
        )


def test_get_chapter_explicit_max_tokens_overrides_default(tmp_path):
    """Explicit max_tokens=500 on get_chapter folder path should produce a short result."""
    chapter_dir = tmp_path / "ch01"
    chapter_dir.mkdir()
    index_file = chapter_dir / "_index.md"
    index_file.write_text(LONG_SECTION_CONTENT)

    mcp = MagicMock()
    captured = _register_tools(mcp)

    with patch("src.tools.chapter_tools.execute_single") as mock_db, \
         patch("src.tools.chapter_tools._find_chapter_path") as mock_path, \
         patch("src.tools.chapter_tools.validate_book_id", return_value="bookid123"), \
         patch("src.tools.chapter_tools.validate_chapter_number", return_value=1):

        mock_db.return_value = {"title": "Ch1", "file_path": str(chapter_dir / "ch01.md"), "word_count": 10000}
        mock_path.return_value = (chapter_dir, True)

        result = captured["get_chapter"]("bookid123", 1, max_tokens=500)
        assert len(result) < 8000, f"Expected result < 8000 chars with max_tokens=500, got {len(result)}"

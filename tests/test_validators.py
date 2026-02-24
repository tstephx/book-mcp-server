"""Tests for resolve_book_id slug resolution and did-you-mean errors."""
from unittest.mock import patch
import pytest
from src.utils.validators import resolve_book_id, ValidationError


VALID_UUID = "438ed6e4-d90f-4996-9dd3-460c45fbba67"
VALID_UUID_2 = "922e5dd7-defd-421e-92a4-dedc6bf87275"


def test_resolve_book_id_passes_valid_uuid_through():
    """A valid UUID bypasses DB lookup entirely."""
    with patch("src.utils.validators.execute_query") as mock_db:
        result = resolve_book_id(VALID_UUID)
    assert result == VALID_UUID
    mock_db.assert_not_called()


def test_resolve_book_id_slug_resolves_to_uuid():
    """A slug is converted to spaces and matched against book titles."""
    with patch("src.utils.validators.execute_query") as mock_query:
        mock_query.return_value = [
            {"id": VALID_UUID, "title": "Agentic Design Patterns A Hands On Guide"}
        ]
        result = resolve_book_id("agentic-design-patterns-a-hands-on-guide")
    assert result == VALID_UUID


def test_resolve_book_id_slug_single_match_returns_uuid():
    """Single fuzzy match returns that book's UUID."""
    with patch("src.utils.validators.execute_query") as mock_query:
        mock_query.return_value = [
            {"id": VALID_UUID_2, "title": "Arduino Programming Essentials"}
        ]
        result = resolve_book_id("arduino-programming-essentials")
    assert result == VALID_UUID_2


def test_resolve_book_id_no_match_raises_with_suggestions():
    """No match raises ValidationError with did-you-mean suggestions."""
    with patch("src.utils.validators.execute_query") as mock_query:
        # First call (exact match): returns empty
        # Second call (broad search for suggestions): returns candidates
        mock_query.side_effect = [
            [],  # no fuzzy match
            [
                {"id": VALID_UUID, "title": "Agentic Design Patterns"},
                {"id": VALID_UUID_2, "title": "Agentic AI Foundations"},
            ],
        ]
        with pytest.raises(ValidationError) as exc_info:
            resolve_book_id("agentic-something-unknown")
    assert "did you mean" in str(exc_info.value).lower()
    assert "Agentic Design Patterns" in str(exc_info.value)


def test_resolve_book_id_no_match_no_suggestions_raises_plain_error():
    """No match and no suggestions raises a plain invalid-ID error."""
    with patch("src.utils.validators.execute_query") as mock_query:
        mock_query.side_effect = [[], []]  # no matches at all
        with pytest.raises(ValidationError) as exc_info:
            resolve_book_id("completely-unknown-slug")
    assert "not found" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()


def test_resolve_book_id_empty_raises():
    """Empty string raises immediately."""
    with pytest.raises(ValidationError):
        resolve_book_id("")


def test_resolve_book_id_multiple_matches_returns_first():
    """Multiple matches returns the first (highest relevance) result."""
    with patch("src.utils.validators.execute_query") as mock_query:
        mock_query.return_value = [
            {"id": VALID_UUID, "title": "Docker Deep Dive"},
            {"id": VALID_UUID_2, "title": "Docker Networking Guide"},
        ]
        result = resolve_book_id("docker-deep-dive")
    assert result == VALID_UUID


def test_get_book_info_accepts_slug(tmp_path):
    """get_book_info should resolve slug IDs, not reject them."""
    with patch("src.tools.book_tools.resolve_book_id", return_value=VALID_UUID) as mock_resolve, \
         patch("src.tools.book_tools.execute_single") as mock_db, \
         patch("src.tools.book_tools.execute_query") as mock_chapters:
        mock_db.return_value = {
            "id": VALID_UUID, "title": "Docker Deep Dive", "author": "Nigel Poulton",
            "word_count": 50000, "added_date": "2024-01-01", "processing_status": "complete",
            "file_path": None, "language": "en", "description": None
        }
        mock_chapters.return_value = []
        from src.tools.book_tools import register_book_tools
        from unittest.mock import MagicMock
        mcp = MagicMock()
        captured = {}
        def tool_decorator(func):
            captured[func.__name__] = func
            return func
        mcp.tool.return_value = tool_decorator
        register_book_tools(mcp)
        result = captured["get_book_info"]("docker-deep-dive")
        mock_resolve.assert_called_once_with("docker-deep-dive")
        assert "Docker" in result or VALID_UUID in result

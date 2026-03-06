"""Tests for analytics_tools."""

from src.tools.analytics_tools import _detect_topics, TOPIC_KEYWORDS


def test_detect_topics_python():
    """_detect_topics should detect Python topic from keywords."""
    result = _detect_topics("A guide to python programming with django")
    assert "Python" in result


def test_detect_topics_multiple():
    """_detect_topics should detect multiple topics."""
    result = _detect_topics("Building docker containers for machine learning models")
    assert "DevOps" in result
    assert "Machine Learning" in result


def test_detect_topics_general_fallback():
    """_detect_topics should return General when no keywords match."""
    result = _detect_topics("A completely unrelated topic about cooking")
    assert result == ["General"]


def test_detect_topics_case_insensitive():
    """_detect_topics should be case-insensitive."""
    result = _detect_topics("KUBERNETES deployment strategies")
    assert "DevOps" in result


def test_topic_keywords_completeness():
    """TOPIC_KEYWORDS should cover all expected categories."""
    expected = {
        "Python", "Data Science", "Machine Learning", "Architecture",
        "DevOps", "Linux", "Networking", "Web Development",
        "Databases", "Security", "Quantum", "Forecasting", "Async/Concurrency",
    }
    assert set(TOPIC_KEYWORDS.keys()) == expected

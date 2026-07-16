"""escape_fts_query must render any user text safe for FTS5 MATCH."""

import sqlite3

import pytest

from src.utils.fts_search import escape_fts_query


def _match(escaped: str, rows: list[str]) -> list[str]:
    """Run a real FTS5 MATCH against an in-memory table; raises on bad syntax."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE t USING fts5(content)")
    conn.executemany("INSERT INTO t (content) VALUES (?)", [(r,) for r in rows])
    return [r[0] for r in conn.execute("SELECT content FROM t WHERE t MATCH ?", (escaped,))]


class TestEscapeFtsQuery:
    def test_apostrophe_no_longer_breaks_match(self):
        escaped = escape_fts_query("Jakob's law")
        hits = _match(escaped, ["the Jakob's law of interfaces", "unrelated text"])
        assert hits == ["the Jakob's law of interfaces"]

    def test_period_term_is_safe(self):
        escaped = escape_fts_query("zone.js change detection")
        # must not raise fts5 syntax error
        _match(escaped, ["angular zone.js change detection internals"])

    def test_operators_preserved(self):
        escaped = escape_fts_query("python AND async")
        assert " AND " in escaped
        hits = _match(escaped, ["python async io", "python only", "async only"])
        assert hits == ["python async io"]

    def test_prefix_star_preserved(self):
        escaped = escape_fts_query("pytho*")
        hits = _match(escaped, ["python rocks", "java rocks"])
        assert hits == ["python rocks"]

    def test_quoted_input_passes_through_unchanged(self):
        assert escape_fts_query('"exact phrase"') == '"exact phrase"'

    def test_legacy_special_chars_still_safe(self):
        for q in ["foo-bar", "a:b", "(paren)", "caret^"]:
            _match(escape_fts_query(q), ["foo-bar a:b (paren) caret^ text"])

    def test_types_and_empty(self):
        assert isinstance(escape_fts_query("x"), str)
        assert escape_fts_query("   ") == ""

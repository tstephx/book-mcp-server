"""Smoke tests for all MCP tools and CLI commands.

Tests that every tool returns without crashing. No output quality assertions.

Run:
    python -m pytest tests/test_smoke_mcp.py -v -m "not slow"       # fast only
    python -m pytest tests/test_smoke_mcp.py -v                      # full suite
    python -m pytest tests/test_smoke_mcp.py -v -k "pipeline"        # pipeline only
    python -m pytest tests/test_smoke_mcp.py -v -k "cli"             # CLI only
"""

import os
import sqlite3
import subprocess
import sys

import pytest

# ---------------------------------------------------------------------------
# Constants & skip conditions
# ---------------------------------------------------------------------------

DB_PATH = os.path.expanduser("~/_Projects/book-ingestion-python/data/library.db")
BOOKS_DIR = os.path.expanduser("~/_Projects/book-ingestion-python/data/books")
HAS_DB = os.path.exists(DB_PATH)
HAS_OPENAI = bool(os.environ.get("OPENAI_API_KEY", ""))

needs_openai = pytest.mark.skipif(not HAS_OPENAI, reason="No OPENAI_API_KEY")
slow = pytest.mark.slow

pytestmark = pytest.mark.skipif(not HAS_DB, reason="Library database not found")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server():
    """Create FastMCP server with all book-library tools registered."""
    os.environ.setdefault("BOOK_DB_PATH", DB_PATH)
    os.environ.setdefault("BOOKS_DIR", BOOKS_DIR)
    from src.server import create_server

    return create_server()


@pytest.fixture(scope="module")
def test_data():
    """Resolve book_id, chapter_id, and chapter_number from the real DB."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    book = conn.execute("SELECT id FROM books ORDER BY title LIMIT 1").fetchone()
    chapter = conn.execute(
        "SELECT id, chapter_number FROM chapters "
        "WHERE book_id = ? ORDER BY chapter_number LIMIT 1",
        (book["id"],),
    ).fetchone()

    conn.close()

    return {
        "book_id": book["id"],
        "chapter_id": chapter["id"],
        "chapter_number": chapter["chapter_number"],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Sentinel values replaced with fixture data at runtime
_BOOK_ID = "BOOK_ID"
_CH_NUM = "CH_NUM"
_CHAPTER_ID = "CHAPTER_ID"


def _call_tool(server, name, **kwargs):
    """Call a FastMCP tool's underlying function by name."""
    tool = server._tool_manager._tools.get(name)
    if tool is None:
        pytest.fail(f"Tool '{name}' not registered on server")
    return tool.fn(**kwargs)


def _resolve(params: dict, test_data: dict) -> dict:
    """Replace sentinel values in params with real fixture data."""
    mapping = {
        _BOOK_ID: test_data["book_id"],
        _CH_NUM: test_data["chapter_number"],
        _CHAPTER_ID: test_data["chapter_id"],
    }
    return {k: mapping.get(v, v) for k, v in params.items()}


# ===================================================================
# Book Library MCP Tools
# ===================================================================

BOOK_TOOLS = [
    # --- Core Library ---
    ("list_books", {}),
    ("get_book_info", {"book_id": _BOOK_ID}),
    ("get_table_of_contents", {"book_id": _BOOK_ID}),
    # --- Chapter Reading ---
    ("get_chapter", {"book_id": _BOOK_ID, "chapter_number": _CH_NUM}),
    ("get_section", {"book_id": _BOOK_ID, "chapter_number": _CH_NUM, "section_number": 1}),
    ("list_sections", {"book_id": _BOOK_ID, "chapter_number": _CH_NUM}),
    # --- Search (no OpenAI) ---
    ("search_books", {"query": "docker"}),
    ("text_search", {"query": "docker", "limit": 5}),
    # --- Discovery (no OpenAI) ---
    ("extract_code_examples", {"book_id": _BOOK_ID, "chapter_number": _CH_NUM}),
    # --- Reading Progress (read-only) ---
    ("get_reading_progress", {"book_id": _BOOK_ID}),
    ("get_bookmarks", {"book_id": _BOOK_ID}),
    # --- Analytics ---
    ("get_library_statistics", {}),
    ("get_author_insights", {}),
    # --- Export ---
    ("export_chapter_to_markdown", {"book_id": _BOOK_ID, "chapter_number": _CH_NUM}),
    ("create_study_guide", {"book_id": _BOOK_ID, "chapter_number": _CH_NUM, "format": "summary"}),
    # --- Learning (template-only) ---
    ("list_project_templates", {}),
    # --- Planning (template-only) ---
    ("list_implementation_templates", {}),
    ("get_phase_prompts", {"goal": "Build a VPS on Hetzner"}),
    ("list_architecture_templates", {}),
    # --- Summaries ---
    ("get_summary", {"chapter_id": _CHAPTER_ID}),
    # --- System/Admin ---
    ("library_status", {}),
    ("get_library_stats", {}),
    ("get_cache_stats", {}),
    ("audit_chapter_quality", {"severity": "bad"}),
]

BOOK_TOOLS_OPENAI = [
    # --- Search (needs OpenAI) ---
    pytest.param("semantic_search", {"query": "docker containers", "limit": 3},
                 marks=[needs_openai], id="semantic_search"),
    pytest.param("hybrid_search", {"query": "docker containers", "limit": 5},
                 marks=[needs_openai], id="hybrid_search"),
    # --- Discovery (needs OpenAI) ---
    pytest.param("find_related_content",
                 {"text_snippet": "container networking fundamentals", "limit": 3},
                 marks=[needs_openai, slow], id="find_related_content"),
    pytest.param("get_topic_coverage", {"topic": "docker", "include_excerpts": False},
                 marks=[needs_openai, slow], id="get_topic_coverage"),
    # --- Search (OpenAI + slow) ---
    pytest.param("search_all_books",
                 {"query": "docker containers", "max_per_book": 1},
                 marks=[needs_openai, slow], id="search_all_books"),
    # --- Learning (OpenAI + slow) ---
    pytest.param("teach_concept",
                 {"concept": "git branching", "depth": "executive"},
                 marks=[needs_openai, slow], id="teach_concept"),
    pytest.param("generate_learning_path",
                 {"goal": "Build a VPS on Hetzner", "depth": "quick", "include_concepts": False},
                 marks=[needs_openai, slow], id="generate_learning_path"),
    # --- Planning (OpenAI + slow) ---
    pytest.param("generate_implementation_plan",
                 {"goal": "Build a VPS on Hetzner"},
                 marks=[needs_openai, slow], id="generate_implementation_plan"),
    pytest.param("generate_brd",
                 {"goal": "Build a VPS on Hetzner", "template_style": "lean"},
                 marks=[needs_openai, slow], id="generate_brd"),
    pytest.param("generate_wireframe_brief",
                 {"goal": "Build a VPS on Hetzner", "audience": "executive"},
                 marks=[needs_openai, slow], id="generate_wireframe_brief"),
    pytest.param("analyze_project",
                 {"goal": "Build a VPS on Hetzner", "mode": "overview"},
                 marks=[needs_openai, slow], id="analyze_project"),
]

BOOK_TOOLS_SLOW = [
    # --- Analytics (slow, no OpenAI) ---
    pytest.param("find_duplicate_coverage", {"similarity_threshold": 0.8, "max_results": 5},
                 marks=[slow], id="find_duplicate_coverage"),
    pytest.param("summarize_book", {"book_id": _BOOK_ID},
                 marks=[slow], id="summarize_book"),
]

ALL_BOOK_TOOLS = BOOK_TOOLS + BOOK_TOOLS_OPENAI + BOOK_TOOLS_SLOW


@pytest.mark.parametrize(
    "tool_name,params",
    ALL_BOOK_TOOLS,
    ids=[t[0] for t in BOOK_TOOLS] + [None] * (len(BOOK_TOOLS_OPENAI) + len(BOOK_TOOLS_SLOW)),
)
def test_book_library_smoke(server, test_data, tool_name, params):
    """Each book library MCP tool returns without error."""
    resolved = _resolve(params, test_data)
    result = _call_tool(server, tool_name, **resolved)
    assert result is not None


# ===================================================================
# Mutating book-library tools (lifecycle tests)
# ===================================================================


def test_reading_progress_lifecycle(server, test_data):
    """mark_as_reading → mark_as_read round-trip works."""
    bid = test_data["book_id"]
    ch = test_data["chapter_number"]

    result = _call_tool(server, "mark_as_reading", book_id=bid, chapter_number=ch)
    assert result is not None

    result = _call_tool(
        server, "mark_as_read", book_id=bid, chapter_number=ch, notes="SMOKE_TEST"
    )
    assert result is not None

    progress = _call_tool(server, "get_reading_progress", book_id=bid)
    assert progress is not None


def test_bookmark_lifecycle(server, test_data):
    """add_bookmark → get_bookmarks → remove_bookmark round-trip works."""
    bid = test_data["book_id"]
    ch = test_data["chapter_number"]

    added = _call_tool(
        server, "add_bookmark",
        book_id=bid, chapter_number=ch, title="SMOKE_TEST", note="auto-cleanup",
    )
    assert added is not None

    bookmarks = _call_tool(server, "get_bookmarks", book_id=bid)
    assert bookmarks is not None

    # Clean up — find our smoke test bookmark and remove it
    if isinstance(bookmarks, dict):
        bm_list = bookmarks.get("bookmarks", [])
    else:
        bm_list = bookmarks if isinstance(bookmarks, list) else []

    for bm in bm_list:
        bm_id = bm.get("id") or bm.get("bookmark_id")
        if bm.get("title") == "SMOKE_TEST" and bm_id is not None:
            removed = _call_tool(server, "remove_bookmark", bookmark_id=bm_id)
            assert removed is not None
            break


def test_clear_cache(server):
    """clear_cache returns confirmation without error."""
    result = _call_tool(server, "clear_cache", cache_type="chapters")
    assert result is not None
    assert "error" not in result


# Embedding management tools — skipped by default (OpenAI + slow + known issues)

@pytest.mark.skipif(not HAS_OPENAI, reason="No OPENAI_API_KEY")
@pytest.mark.slow
def test_refresh_embeddings_smoke(server):
    """refresh_embeddings returns status dict."""
    result = _call_tool(server, "refresh_embeddings", force=False)
    assert result is not None
    assert "status" in result or "error" in result


@pytest.mark.skipif(not HAS_OPENAI, reason="No OPENAI_API_KEY")
@pytest.mark.slow
def test_generate_summary_embeddings_smoke(server):
    """generate_summary_embeddings returns dict (may error if migration missing)."""
    result = _call_tool(server, "generate_summary_embeddings", force=False)
    assert result is not None


# ===================================================================
# Pipeline MCP Tools
# ===================================================================

PIPELINE_TOOLS = [
    ("review_pending_books", {}, "review_pending_books"),
    ("get_pipeline_health", {}, "get_pipeline_health"),
    ("get_stuck_pipelines", {}, "get_stuck_pipelines"),
    ("batch_approve_tool", {"execute": False}, "batch_approve_tool"),
    ("batch_reject_tool", {"reason": "smoke test preview", "execute": False}, "batch_reject_tool"),
    ("get_audit_log", {"last_days": 7, "limit": 5}, "get_audit_log"),
    ("get_autonomy_status", {}, "get_autonomy_status"),
    ("get_autonomy_readiness", {}, "get_autonomy_readiness"),
    ("backfill_library", {"dry_run": True}, "backfill_library"),
    ("validate_library", {}, "validate_library"),
]


@pytest.mark.parametrize(
    "name,params,fn_name",
    PIPELINE_TOOLS,
    ids=[t[0] for t in PIPELINE_TOOLS],
)
def test_pipeline_smoke(name, params, fn_name):
    """Each pipeline MCP tool returns without error."""
    from agentic_pipeline import mcp_server

    fn = getattr(mcp_server, fn_name)
    result = fn(**params)
    assert result is not None


# ===================================================================
# CLI Commands
# ===================================================================

CLI_COMMANDS = [
    (["version"],),
    (["init"],),
    (["pending"],),
    (["strategies"],),
    (["health"],),
    (["health", "--json"],),
    (["stuck"],),
    (["library-status"],),
    (["library-status", "--json"],),
    (["audit", "--last", "5"],),
    (["validate"],),
    (["validate", "--json"],),
    (["backfill", "--dry-run"],),
    (["autonomy", "status"],),
    (["spot-check", "--list"],),
    (["batch-approve"],),
    (["batch-reject", "--reason", "smoke test preview"],),
]


@pytest.mark.parametrize("args", CLI_COMMANDS, ids=[" ".join(a[0]) for a in CLI_COMMANDS])
def test_cli_smoke(args):
    """Each CLI command exits with code 0."""
    cmd = [sys.executable, "-m", "agentic_pipeline.cli"] + args[0]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, (
        f"CLI command failed: {' '.join(cmd)}\n"
        f"stderr: {result.stderr[:500]}"
    )

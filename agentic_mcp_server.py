#!/usr/bin/env python3
"""
MCP Server wrapper for Agentic Pipeline tools.

Exposes book processing, approval, health monitoring, and autonomy
management tools to Claude via MCP.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from fastmcp import FastMCP

# Import all the tool functions
from agentic_pipeline.mcp_server import (
    # Core approval tools
    review_pending_books,
    approve_book_tool,
    reject_book_tool,
    rollback_book_tool,
    # Processing
    process_book,
    get_pipeline_status,
    # Health monitoring (Phase 4)
    get_pipeline_health,
    get_stuck_pipelines,
    # Batch operations (Phase 4)
    batch_approve_tool,
    batch_reject_tool,
    # Audit (Phase 4)
    get_audit_log,
    # Autonomy (Phase 5)
    get_autonomy_status,
    set_autonomy_mode,
    activate_escape_hatch_tool,
    get_autonomy_readiness,
)

# Create MCP server
mcp = FastMCP("agentic-pipeline")

# Register tools with cleaner names for Claude

@mcp.tool()
def pending_books(sort_by: str = "priority") -> dict:
    """Get all books pending approval. Returns queue with stats."""
    return review_pending_books(sort_by=sort_by)

@mcp.tool()
def approve(pipeline_id: str, actor: str = "mcp:claude") -> dict:
    """Approve a book for ingestion into the library."""
    return approve_book_tool(pipeline_id=pipeline_id, actor=actor)

@mcp.tool()
def reject(pipeline_id: str, reason: str, retry: bool = False, actor: str = "mcp:claude") -> dict:
    """Reject a book. Set retry=True to queue for reprocessing."""
    return reject_book_tool(pipeline_id=pipeline_id, reason=reason, actor=actor, retry=retry)

@mcp.tool()
def rollback(pipeline_id: str, reason: str, actor: str = "mcp:claude") -> dict:
    """Rollback an approved book from the library."""
    return rollback_book_tool(pipeline_id=pipeline_id, reason=reason, actor=actor)

@mcp.tool()
def process(book_path: str) -> dict:
    """Process a book file through the pipeline. Returns pipeline_id and classification."""
    return process_book(book_path)

@mcp.tool()
def status(pipeline_id: str) -> dict:
    """Get status of a pipeline run including state, book_type, confidence."""
    return get_pipeline_status(pipeline_id)

@mcp.tool()
def health() -> dict:
    """Get pipeline health: active, queued, stuck counts and alerts."""
    return get_pipeline_health()

@mcp.tool()
def stuck() -> list:
    """Get list of pipelines that appear stuck."""
    return get_stuck_pipelines()

@mcp.tool()
def batch_approve(
    min_confidence: float = None,
    book_type: str = None,
    max_count: int = 50,
    execute: bool = False
) -> dict:
    """
    Approve books matching filters.
    Set execute=True to apply (otherwise preview only).
    """
    return batch_approve_tool(
        min_confidence=min_confidence,
        book_type=book_type,
        max_count=max_count,
        execute=execute
    )

@mcp.tool()
def batch_reject(
    reason: str,
    book_type: str = None,
    max_confidence: float = None,
    max_count: int = 50,
    execute: bool = False
) -> dict:
    """
    Reject books matching filters.
    Set execute=True to apply (otherwise preview only).
    """
    return batch_reject_tool(
        book_type=book_type,
        max_confidence=max_confidence,
        reason=reason,
        max_count=max_count,
        execute=execute
    )

@mcp.tool()
def audit(
    book_id: str = None,
    actor: str = None,
    action: str = None,
    last_days: int = 7,
    limit: int = 100
) -> list:
    """Query audit trail. Filter by book_id, actor, action, or time range."""
    return get_audit_log(
        book_id=book_id,
        actor=actor,
        action=action,
        last_days=last_days,
        limit=limit
    )

@mcp.tool()
def autonomy_status() -> dict:
    """Get current autonomy mode, escape hatch status, and 30-day metrics."""
    return get_autonomy_status()

@mcp.tool()
def set_autonomy(mode: str) -> dict:
    """Change autonomy mode: 'supervised', 'partial', or 'confident'."""
    return set_autonomy_mode(mode)

@mcp.tool()
def escape_hatch(reason: str) -> dict:
    """Emergency: immediately revert to supervised mode."""
    return activate_escape_hatch_tool(reason)

@mcp.tool()
def autonomy_readiness() -> dict:
    """Check if system is ready to advance to next autonomy level."""
    return get_autonomy_readiness()


if __name__ == "__main__":
    mcp.run()

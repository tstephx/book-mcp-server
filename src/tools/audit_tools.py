"""
Chapter quality audit tools
Modular tool registration following MCP best practices
"""

from typing import TYPE_CHECKING

from ..utils.logging import logger

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def register_audit_tools(mcp: "FastMCP") -> None:
    """Register chapter quality audit tools."""

    @mcp.tool()
    def audit_chapter_quality(
        book_id: str = "",
        severity: str = "all",
    ) -> dict:
        """Audit chapter quality across the library or for a specific book

        Analyzes chapter data and flags issues:
        - Over-fragmentation: too many tiny chapters (front matter, appendices split out)
        - Under-fragmentation: too few massive chapters (chapter detection failures)
        - Title quality: generic names, PDF artifacts, non-chapter content
        - EPUB TOC mismatch: DB chapter count vs original EPUB table of contents

        Args:
            book_id: Optional book UUID. If empty, audits all books.
            severity: Filter results - "all", "warning" (warning+bad), or "bad" only

        Returns:
            Dictionary with:
            - summary: counts by severity (good/warning/bad)
            - books: per-book audit results with issues list

        Examples:
            audit_chapter_quality()  # All books
            audit_chapter_quality(severity="bad")  # Only problem books
            audit_chapter_quality(book_id="abc-123")  # Single book deep dive
        """
        try:
            import sys
            from pathlib import Path

            # Import the audit module from scripts/
            scripts_dir = Path(__file__).resolve().parent.parent.parent / "scripts"
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))

            from audit_chapters import audit_library

            result = audit_library(
                book_id=book_id if book_id else None,
                severity_filter=severity,
            )
            book_count = len(result.get("books", []))
            logger.info(
                f"Chapter audit complete: {result['summary']['total']} books, "
                f"{result['summary']['bad']} bad, {result['summary']['warning']} warning"
            )
            return result

        except Exception as e:
            logger.error(f"Chapter audit error: {e}", exc_info=True)
            return {"error": str(e), "summary": {}, "books": []}

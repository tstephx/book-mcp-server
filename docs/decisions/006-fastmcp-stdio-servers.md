# ADR 006: Two Separate FastMCP stdio Servers

**Status:** Accepted
**Date:** 2026-01-28

## Context

The project has two distinct tool surfaces:
- **Book library** — read-only search/read/learning tools for Claude Desktop users
- **Agentic pipeline** — approval, health, autonomy management for operators

Options: one combined server, or two separate servers.

## Decision

Two separate FastMCP stdio servers in the same repo:
- `server.py` → `src/server.py` (book library, Claude Desktop)
- `agentic_mcp_server.py` → `agentic_pipeline/mcp_server.py` (pipeline, operators)

## Consequences

- Clean separation of concerns — library tools never mix with pipeline tools
- Each server can be configured independently in `.mcp.json`
- Two processes to manage if both are needed simultaneously
- Shared codebase means one repo to maintain and deploy
- stdio transport means no port conflicts or network config needed

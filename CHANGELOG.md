# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- `schema_migrations` version table in SQLite — ALTER TABLE migrations now run exactly once and are tracked by name
- `PRAGMA foreign_keys = ON` enforced on every database connection
- Processing and embedding timeout enforcement via `ThreadPoolExecutor` in orchestrator
- `@pytest.mark.integration` marker for tests requiring external services (real DB, OpenAI key)
- `@pytest.mark.e2e` marker for CLI end-to-end tests
- GitHub Actions CI workflow (`ci.yml`) — unit tests, coverage, changelog validation, ruff lint
- `Makefile` with `test`, `test-cov`, `test-fast`, `test-integration`, `test-e2e` targets
- `pytest-cov` and `pytest-xdist` as dev dependencies

### Fixed
- Audit trail hardcoded `autonomy_mode = "supervised"` — now reads actual mode from `AutonomyConfig`
- Invalid `PROCESSING -> FAILED` state transition replaced with valid `PROCESSING -> REJECTED` on force-fallback retry error
- `_seen_paths` set growth (unbounded in-memory) — documented; DB-backed idempotency via `content_hash` is authoritative

---

## [0.5.0] — 2026-02-25

### Added
- `--force-fallback` flag on `reingest` CLI command
- Unified book processing architecture (see `docs/archive/2026-02-25-unified-book-processing-architecture.md`)
- Slug-based book ID resolution across all book tools

### Fixed
- Strip `//` comments from OpenAI JSON response before parsing
- LLM fallback adapter switched from Anthropic to OpenAI
- Split embedding batches by token count to stay under OpenAI 300k limit
- Double tokenization eliminated in embedding pipeline
- Guard `sources[0]` access in learning tools against empty sources

---

## [0.4.0] — 2026-01-29

### Added
- Phase 5: Confident Autonomy — per-type calibrated thresholds, spot-check system
- Batch processing session support
- `permanently_failed` count in health monitor

### Fixed
- 24h window for failure rate calculation
- CLI alignment and `get()` fallback handling

---

## [0.3.0] — 2025-01-28

### Added
- Phase 4: Production hardening — stuck detection, health monitor, escape hatch
- Phase 3: Orchestrator with full state machine
- Phase 2: Classifier agent with AI-powered book type detection
- Phase 1: Foundation — pipeline DB, state machine, approval queue

---

## [0.1.0] — Initial Release

### Added
- Book library MCP server (`server.py`) with search, read, and learning tools
- Agentic processing pipeline MCP server (`agentic_mcp_server.py`)
- CLI (`agentic-pipeline`) with approval, health, autonomy management commands
- SQLite + WAL mode backend
- OpenAI `text-embedding-3-large` semantic search
- Hybrid search (RRF: FTS5 + semantic vector)

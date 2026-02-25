# Architecture Decision Records

Decisions are numbered and immutable. Once recorded, they are not edited —
if a decision is reversed, a new ADR is added superseding the old one.

| # | Decision | Status |
|---|----------|--------|
| [001](001-sqlite-wal-mode.md) | SQLite with WAL mode for pipeline database | Accepted |
| [002](002-inline-embedding-on-approval.md) | Inline embedding during approval (not separate worker) | Accepted |
| [003](003-openai-text-embedding-3-large.md) | OpenAI text-embedding-3-large for semantic search | Accepted |
| [004](004-processing-adapter-lazy-import.md) | ProcessingAdapter wraps book-ingestion as lazy-imported library | Accepted |
| [005](005-rrf-hybrid-search.md) | RRF hybrid search combining FTS5 + semantic vectors | Accepted |
| [006](006-fastmcp-stdio-servers.md) | Two separate FastMCP stdio servers (library + pipeline) | Accepted |
| [007](007-slug-id-resolution.md) | Slug ID resolution — accept partial titles in place of UUIDs | Accepted |

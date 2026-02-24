# Manual Test Plan: Book MCP Server

**System:** Book Library MCP Server + Agentic Pipeline MCP Server + CLI
**Scope:** Full system — all MCP tools, all CLI commands
**Last updated:** 2026-02-11

---

## Execution Guide

### Part 1 — AUTOMATED

Part 1 smoke tests are automated in `tests/test_smoke_mcp.py`.

```bash
# Fast smoke (no OpenAI, no slow)
python -m pytest tests/test_smoke_mcp.py -v -m "not slow"

# Full smoke (requires OPENAI_API_KEY)
python -m pytest tests/test_smoke_mcp.py -v

# Just pipeline tools
python -m pytest tests/test_smoke_mcp.py -v -k "pipeline"

# Just CLI commands
python -m pytest tests/test_smoke_mcp.py -v -k "cli"
```

### Parts 2 & 3 — Manual Sessions

**Session structure:** Run Part 2 and Part 3 as **separate Claude sessions**.
Even after removing Part 1, 57 quality assertions + 15 edge cases can push context limits.

**Checkpointing:** After each tool group, write results to:
`docs/test-results/YYYY-MM-DD-partN.md`

Example checkpoint format:
```markdown
### semantic_search — 6/6 PASS
- [x] Returns exactly 5 results
- [x] Each result has book_title, chapter_title, similarity
- [x] Similarity scores are between 0.0 and 1.0
- [x] Results sorted by similarity descending
- [x] Top result is Docker-related
- [x] No results below min_similarity threshold
```

**Output suppression:** Only include full MCP response output for **FAILURES**.
For passing checks, just log the checkbox line.

---

## Prerequisites

### Test Fixture

The library contains 173 books with 3,172 chapters and 100% embedding coverage.
Pick a known book to use as the test fixture throughout:

> **Test Book:** Use the first book returned by `list_books()`.
> All tools return formatted text (str), not structured dicts. Capture values manually from the output.

```
BOOK_ID    = <copy book id from list_books() output — alphanumeric string>
BOOK_TITLE = <copy title from list_books() output>
CH_NUM     = 1
CHAPTER_ID = <copy chapter id UUID from get_summary() or DB directly — NOT available in TOC output>
```

> **Test Query:** `"docker containers"`
> **Test Goal:** `"Build a VPS on Hetzner to host my portfolio"`
> **Test Concept:** `"git branching"`

### Environment

- Python venv activated: `source .venv/bin/activate`
- Database exists at `~/_Projects/book-ingestion-python/data/library.db`
- `OPENAI_API_KEY` set (needed for embedding/semantic operations)
- `AGENTIC_PIPELINE_DB` set or default path valid

---

***

## Part 1: Smoke Tests *(AUTOMATED — see `tests/test_smoke_mcp.py`)*

> **Run `python -m pytest tests/test_smoke_mcp.py -v` instead of testing these manually.**

Quick pass — every tool returns a non-error response. No output quality judgment.

### **Book Library MCP Tools**

**Core Library (3)**

```
list_books() — returns str (formatted book list)
get_book_info(book_id=BOOK_ID) — returns str with title, author, word_count
get_table_of_contents(book_id=BOOK_ID) — returns str with chapter list
```

**Chapter Reading (3)**

```
get_chapter(book_id=BOOK_ID, chapter_number=CH_NUM) — returns str with chapter content or section index
get_section(book_id=BOOK_ID, chapter_number=CH_NUM, section_number=1) — returns str with section content (or error if not split)
list_sections(book_id=BOOK_ID, chapter_number=CH_NUM) — returns str with section list (or empty if not split)
```

**Search (5)**

```
search_titles(query="docker") — returns str with book and chapter matches
semantic_search(query="docker containers", limit=3) — returns str with similarity-ranked results
hybrid_search(query="docker containers", limit=5) — returns str with RRF-fused results (rrf_score field)
text_search(query="docker", limit=5) — returns str with FTS5 keyword matches
search_all_books(query="docker containers", max_per_book=2) — returns str grouped by book
```

**Discovery (3)**

```
find_related_content(text_snippet="container networking fundamentals") — returns str with cross-book matches
get_topic_coverage(topic="docker") — returns str with all chapters covering topic
extract_code_examples(book_id=BOOK_ID, chapter_number=CH_NUM) — returns str with code blocks
```

**Reading Progress (6)**

```
mark_as_reading(book_id=BOOK_ID, chapter_number=CH_NUM) — returns str confirmation
mark_as_read(book_id=BOOK_ID, chapter_number=CH_NUM, notes="test") — returns str confirmation
get_reading_progress(book_id=BOOK_ID) — returns str with chapter statuses
add_bookmark(book_id=BOOK_ID, chapter_number=CH_NUM, title="Test bookmark") — returns str confirmation
get_bookmarks(book_id=BOOK_ID) — returns str with bookmarks list
remove_bookmark(bookmark_id=<from add_bookmark result>) — returns str confirmation
```

**Analytics (3)**

```
get_library_statistics() — returns str with word counts, author distribution
find_duplicate_coverage(similarity_threshold=0.7) — returns str with similar chapter pairs
get_author_insights() — returns str with author analytics
```

**Export (2)**

```
export_chapter_to_markdown(book_id=BOOK_ID, chapter_number=CH_NUM) — returns str with markdown content
create_study_guide(book_id=BOOK_ID, chapter_number=CH_NUM, format="summary") — returns str with study content
```

**Learning (3)**

```
teach_concept(concept="git branching", depth="executive") — returns str with analogy + content
generate_learning_path(goal="Build a VPS on Hetzner", depth="quick") — returns str with phases
list_project_templates() — returns str with template list
```

**Project Planning (7)**

```
generate_implementation_plan(goal="Build a VPS on Hetzner") — returns str with phases + milestones
list_implementation_templates() — returns str with template list
get_phase_prompts(goal="Build a VPS on Hetzner") — returns str with prompts by phase
generate_brd(goal="Build a VPS on Hetzner", template_style="lean") — returns str with BRD document
generate_wireframe_brief(goal="Build a VPS on Hetzner", audience="executive") — returns str with architecture brief
list_architecture_templates() — returns str with template list
analyze_project(goal="Build a VPS on Hetzner", mode="overview") — returns str with analysis
```

**Summaries (2)**

```
get_summary(chapter_id=CHAPTER_ID) — returns str with extractive summary
summarize_book(book_id=BOOK_ID) — returns str with generation results
```

**System/Admin (5)**

```
library_status() — returns str with overview, books, pipeline_summary
get_library_stats() — returns str with aggregate statistics
get_cache_stats() — returns str with cache hit/miss rates
clear_cache(cache_type="chapters") — returns str confirmation + stats
audit_chapter_quality(severity="bad") — returns str with audit results
```

**Embedding Management (2)**

```
refresh_embeddings() — returns str with status, updated/skipped counts
generate_summary_embeddings() — returns str with generated/skipped counts
```

---

### **Agentic Pipeline MCP Tools (17)**

**Pipeline Operations (3)**

```
pending_books() — returns dict with queue + stats
health() — returns dict with active/queued/stuck counts
stuck() — returns list of stuck pipelines
```

**Approval Actions (3)**

```
approve(pipeline_id=<from pending>) — returns dict with success + state
reject(pipeline_id=<from pending>, reason="test rejection") — returns dict with success
rollback(pipeline_id=<from approved>, reason="test rollback") — returns dict with success
```

**Batch Operations (2)**

```
batch_approve(execute=False) — returns dict with would_approve count (preview)
batch_reject(reason="low quality", execute=False) — returns dict with would_reject count (preview)
```

**Audit (1)**

```
audit(last_days=7) — returns list of audit entries
```

**Autonomy (4)**

```
autonomy_status() — returns dict with mode, escape_hatch, metrics
set_autonomy(mode="supervised") — returns dict with success
escape_hatch(reason="test") — returns dict with success
autonomy_readiness() — returns dict with readiness assessment
```

**Pipeline Management (2)**

```
process(book_path="/path/to/test.epub") — returns str with pipeline_id (requires real file)
status(pipeline_id=<known id>) — returns str with state, confidence
```

*Note: `process` and `status` require a real book file / valid pipeline ID. Skip if no test fixture available.*

---

### **CLI Commands (20)**

```bash
agentic-pipeline version — prints version string
agentic-pipeline init — initializes DB (idempotent)
agentic-pipeline pending — lists pending books table
agentic-pipeline status <PIPELINE_ID> — shows pipeline details
agentic-pipeline strategies — lists processing strategies
agentic-pipeline health — shows health dashboard
agentic-pipeline health --json — outputs JSON health report
agentic-pipeline stuck — lists stuck pipelines
agentic-pipeline library-status — shows library dashboard
agentic-pipeline library-status --json — outputs JSON library report
agentic-pipeline audit --last 5 — shows recent audit entries
agentic-pipeline validate — checks library quality
agentic-pipeline validate --json — outputs JSON quality report
agentic-pipeline backfill --dry-run — previews untracked books
agentic-pipeline autonomy status — shows autonomy mode
agentic-pipeline spot-check --list — lists pending spot-checks
agentic-pipeline batch-approve — previews batch approve (dry-run by default; add --execute to apply)
agentic-pipeline batch-reject --reason="test" — previews batch reject (dry-run by default)
agentic-pipeline classify --text "Chapter 1: Introduction to Python..." — classifies sample text
agentic-pipeline escape-hatch "test" — activates escape hatch (CAUTION: changes state)
```

**Smoke Test Totals: 44 book-library + 10 pipeline + 17 CLI = 71 automated tests**
*(see `tests/test_smoke_mcp.py`)*

---

***

## Part 2: Quality Validation

Deeper checks on high-value tools where output correctness matters.

---

### list_books
Run: `list_books()`
- [ ] Returns at least 170 books (library had 173 as of 2026-02-11; count grows over time)
- [ ] Output includes id, title, author, word_count for each book (formatted text — verify visually)
- [ ] Books are sorted alphabetically by title
- [ ] No duplicate book IDs visible in results

### get_book_info
Run: `get_book_info(book_id=BOOK_ID)`
- [ ] Returns matching `id` equal to BOOK_ID
- [ ] `title` is a non-empty string
- [ ] `chapter_count` matches what `get_table_of_contents` returns
- [ ] `word_count` is a positive integer
- [ ] `added_date` is a valid date string

### semantic_search
Run: `semantic_search(query="docker containers", limit=5)`
- [ ] Returns exactly 5 results (or fewer if library has fewer matches)
- [ ] Each result has `book_title`, `chapter_title`, `similarity` fields
- [ ] Similarity scores are between 0.0 and 1.0
- [ ] Results are sorted by similarity descending
- [ ] Top result is from a Docker-related book (not a random topic)
- [ ] No results below `min_similarity` threshold (0.3 default)

Run: `semantic_search(query="docker containers", limit=5, min_similarity=0.8)`
- [ ] Returns fewer results than the 0.3 threshold query
- [ ] All results have similarity >= 0.8

### hybrid_search
Run: `hybrid_search(query="docker containers", limit=5, diverse=False)`
- [ ] Results include both keyword and semantic relevance scores
- [ ] Results include an `rrf_score` combined ranking field
- [ ] Top results are Docker-related

Run: `hybrid_search(query="docker containers", limit=5, diverse=True)`
- [ ] Results come from more distinct books than the non-diverse query
- [ ] Does NOT return 5 chapters from the same book

### text_search
Run: `text_search(query="docker", limit=5)`
- [ ] Returns results containing the literal word "docker"
- [ ] Results include highlighted excerpts with the search term
- [ ] BM25 ranking puts most relevant results first

Run: `text_search(query="\"dependency injection\"", limit=5)`
- [ ] Returns results with the exact phrase "dependency injection"
- [ ] Does NOT return results that have "dependency" and "injection" separately

### get_chapter
Run: `get_chapter(book_id=BOOK_ID, chapter_number=CH_NUM)`
- [ ] Returns chapter content as a string (or section index if split)
- [ ] Content length is > 100 characters (not empty/stub)
- [ ] Chapter title is included in the response
- [ ] `max_tokens` parameter truncates content when set to a low value

### teach_concept
Run: `teach_concept(concept="git branching", depth="executive")`
- [ ] Response includes a business/organizational analogy
- [ ] Content is concise (executive = 2-min read, roughly < 500 words of main content)
- [ ] References at least one book from the library as a source
- [ ] Does NOT use deep technical jargon without explanation
- [ ] `related_concepts` suggests what to learn next

Run: `teach_concept(concept="git branching", depth="practitioner")`
- [ ] Content is significantly longer than the executive version
- [ ] Includes technical detail appropriate for decision-making
- [ ] Still includes the business analogy framing

### generate_learning_path
Run: `generate_learning_path(goal="Build a VPS on Hetzner to host my portfolio", depth="comprehensive")`
- [ ] Returns multiple phases (at least 3)
- [ ] Each phase has topics and reading recommendations
- [ ] Reading recommendations reference actual books in the library (not invented titles)
- [ ] Includes time estimates
- [ ] `concept_briefs` are populated (include_concepts=True by default)
- [ ] Phases are ordered logically (foundations before advanced)

### generate_brd
Run: `generate_brd(goal="Build a VPS on Hetzner", template_style="standard")`
- [ ] Contains sections: problem statement, scope, requirements, success metrics, risks
- [ ] Requirements are specific and measurable (not vague)
- [ ] Includes stakeholder analysis
- [ ] Technical requirements section is present (include_technical=True default)

Run: `generate_brd(goal="Build a VPS on Hetzner", template_style="lean")`
- [ ] Output is noticeably shorter than "standard"
- [ ] Still contains core sections (problem, scope, requirements)

### analyze_project
Run: `analyze_project(goal="Build a VPS on Hetzner", mode="overview")`
- [ ] Returns `project_type` classification
- [ ] Returns `complexity` estimate (simple/moderate/complex)
- [ ] `recommendations` list is populated with next steps
- [ ] Does NOT generate full artifacts in overview mode

### get_topic_coverage
Run: `get_topic_coverage(topic="docker")`
- [ ] Groups results by book
- [ ] Each result has chapter number and similarity score
- [ ] Excerpts are included (default include_excerpts=True)
- [ ] Results span multiple books (Docker content exists in several books)

### audit_chapter_quality
Run: `audit_chapter_quality(severity="all")`
- [ ] Returns summary with counts by severity (good/warning/bad)
- [ ] Per-book results include an issues list
- [ ] Issue types include fragmentation and title quality checks
- [ ] Summary counts add up to total books audited

### library_status
Run: `library_status()`
- [ ] `overview.total_books` matches `list_books()` count
- [ ] `overview.embedding_coverage_pct` is 100 (known state)
- [ ] `overview.total_chapters` is approximately 3,220
- [ ] `books` list has per-book embedding percentage
- [ ] `pipeline_summary` shows state distribution

### health (Pipeline)
Run: `health()`
- [ ] Returns `active`, `queued`, `stuck`, `completed_24h`, `failed` counts
- [ ] All counts are non-negative integers
- [ ] `alerts` is a list (possibly empty)
- [ ] `stuck` is a list (possibly empty)

### autonomy_status (Pipeline)
Run: `autonomy_status()`
- [ ] Returns current `mode` (one of: supervised, partial, confident)
- [ ] `escape_hatch_active` is a boolean
- [ ] `metrics_30d` includes `total_processed`, `auto_approved`, `human_approved`, `human_rejected`
- [ ] All metric values are non-negative integers

**Quality Validation Totals: 17 tools, 78 assertions**

---

***

## Part 3: Edge Cases

Verify graceful error handling on bad inputs.

```
get_book_info(book_id="nonexistent-uuid-1234") — returns error dict, not crash
get_book_info(book_id="") — returns error or validation message
get_chapter(book_id=BOOK_ID, chapter_number=99999) — returns error for out-of-range chapter
get_chapter(book_id=BOOK_ID, chapter_number=-1) — returns error for negative chapter
get_section(book_id=BOOK_ID, chapter_number=CH_NUM, section_number=99999) — returns error for invalid section
semantic_search(query="", limit=5) — returns empty results or error, not crash
semantic_search(query="docker", limit=0) — handles zero limit gracefully
semantic_search(query="docker", limit=-1) — handles negative limit gracefully
text_search(query="", limit=5) — returns empty results or error, not crash
hybrid_search(query="a]]][[invalid regex?+*", limit=5) — handles malformed query gracefully
teach_concept(concept="", depth="executive") — returns error for empty concept
generate_brd(goal="") — returns error for empty goal
clear_cache(cache_type="invalid_type") — returns error message about valid types
remove_bookmark(bookmark_id=999999) — returns error for nonexistent bookmark
mark_as_read(book_id="nonexistent", chapter_number=1) — returns error, not crash
```

**Edge Case Totals: 15 tests**

---

***

## Part 4: CLI Smoke Tests *(AUTOMATED — see `tests/test_smoke_mcp.py::test_cli_smoke`)*

Run each command and verify it exits cleanly (exit code 0) with expected output format.

```bash
agentic-pipeline version                              # prints "agentic-pipeline vX.Y.Z"
agentic-pipeline init                                  # prints "initialized successfully"
agentic-pipeline health                                # prints "Pipeline Health" table
agentic-pipeline health --json                         # prints valid JSON
agentic-pipeline library-status                        # prints "Library Status" table
agentic-pipeline library-status --json                 # prints valid JSON
agentic-pipeline pending                               # prints "Pending Approval" or "No books pending"
agentic-pipeline strategies                            # prints "Available Strategies" list
agentic-pipeline stuck                                 # prints stuck list or "No stuck pipelines"
agentic-pipeline validate                              # prints quality table or "All books pass"
agentic-pipeline validate --json                       # prints valid JSON array
agentic-pipeline audit --last 5                        # prints audit table or "No audit entries"
agentic-pipeline autonomy status                       # prints "Autonomy Status" block
agentic-pipeline spot-check --list                     # prints spot-check list or "No spot-checks"
agentic-pipeline backfill --dry-run                    # prints backfill preview or "All tracked"
agentic-pipeline batch-approve                         # prints dry-run preview (no --execute)
agentic-pipeline batch-reject --reason "test preview"  # prints dry-run preview (no --execute)
```

**CLI Totals: 17 tests**

---

***

## Test Run Log

| Date | Tester | Part 1 (Smoke) | Part 2 (Quality) | Part 3 (Edge) | Part 4 (CLI) | Notes |
|------|--------|----------------|-------------------|----------------|--------------|-------|
| 2026-02-11 | Claude | 70 PASS, 1 FAIL, 7 SKIP | 60 PASS, 5 FAIL, 1 PARTIAL, 12 UNTESTED | 15/15 PASS | 17/17 PASS | FTS5 index stale (0 keyword results). generate_summary_embeddings FAIL (missing column). Library is 173 books not 185. |

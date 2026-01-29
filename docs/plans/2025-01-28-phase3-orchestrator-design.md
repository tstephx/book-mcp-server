# Phase 3: Pipeline Orchestrator Design

> **Audience:** Product managers, directors, UX designers, and engineers who need to understand what we're building and why.

---

## Status: 📋 Planned

**Target:** Next implementation phase after Phase 2 (Classifier Agent)

---

## What Is This?

The Pipeline Orchestrator is the "conductor" that moves books through the state machine. It connects the classifier (Phase 2), the existing book-ingestion pipeline, and the approval workflow (Phase 1) into a single automated flow.

**Today:** Each piece exists in isolation. The classifier can classify, the pipeline can process, but nothing connects them automatically.

**With the Orchestrator:** Drop a book in, and it flows through classification → strategy selection → processing → validation → approval → embedding → complete. High-confidence books process fully automatically. Low-confidence books pause for human review.

---

## How It Works

### The Simple Version

```
Book arrives → Hash → Classify → Pick Strategy → Process → Validate → Approve → Embed → Complete
```

### The Detailed Version

```
                              ┌─────────────────────┐
                              │    Entry Points     │
                              ├─────────────────────┤
                              │  CLI    MCP    API  │
                              └─────────┬───────────┘
                                        │
                                        ▼
┌───────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR                                │
│                                                                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐    │
│  │ process_one  │    │  run_worker  │    │  signal_handler  │    │
│  │ (one-shot)   │    │  (queue)     │    │  (graceful stop) │    │
│  └──────┬───────┘    └──────┬───────┘    └──────────────────┘    │
│         │                   │                                      │
│         └─────────┬─────────┘                                      │
│                   ▼                                                │
│         ┌─────────────────┐                                        │
│         │  _process_book  │  ← Core logic, both modes call this   │
│         └────────┬────────┘                                        │
└──────────────────┼─────────────────────────────────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────┐
    │         STATE MACHINE FLOW           │
    │                                       │
    │  DETECTED                            │
    │      │                               │
    │      ▼ (compute hash)                │
    │  HASHING ──────► DUPLICATE (exit)    │
    │      │                               │
    │      ▼ (call ClassifierAgent)        │
    │  CLASSIFYING                         │
    │      │                               │
    │      ▼ (pick strategy)               │
    │  SELECTING_STRATEGY                  │
    │      │                               │
    │      ▼ (subprocess: book-ingestion)  │
    │  PROCESSING ──────► NEEDS_RETRY      │
    │      │                    ▲          │
    │      ▼                    │          │
    │  VALIDATING ──────────────┘          │
    │      │                               │
    │      ▼ (confidence < 0.7?)           │
    │  PENDING_APPROVAL ◄───── yes         │
    │      │        │                      │
    │      │ human  │ auto (≥0.7)          │
    │      ▼        ▼                      │
    │  APPROVED ◄───┘                      │
    │      │                               │
    │      ▼ (subprocess: embeddings)      │
    │  EMBEDDING                           │
    │      │                               │
    │      ▼                               │
    │  COMPLETE                            │
    └──────────────────────────────────────┘
```

### Two Operating Modes

| Mode | Command | Use Case |
|------|---------|----------|
| **One-shot** | `agentic-pipeline process /path/to/book.epub` | Process single book, return result |
| **Queue worker** | `agentic-pipeline worker` | Run continuously, process DETECTED books |

---

## Key Design Decisions

### Decision 1: Why Subprocess Instead of Direct Import?

**What we chose:** Call the existing book-ingestion pipeline via subprocess.

**Why:**

| Approach | Pros | Cons |
|----------|------|------|
| Direct import | Fast, type-safe | Tight coupling, shared failures |
| Subprocess ✓ | Isolated, already tested | Slower (process spawn) |
| HTTP API | Ultimate decoupling | Overkill for local |

The existing CLI works. The orchestrator calls it like a black box:
```bash
python -m src.cli process /path/to/book.epub
```

If processing crashes, the orchestrator survives and marks the book for retry.

**Alternatives considered:**
- **Direct import:** Would require refactoring book-ingestion-python to be import-safe. Not worth it for Phase 3.
- **HTTP API:** Would require running a separate server. Overkill for single-machine deployment.

---

### Decision 2: Why Confidence-Based Routing?

**What we chose:** Books with ≥70% classifier confidence auto-proceed. Below 70% stops for human review.

**Why:**

This balances automation with safety:
- High confidence = classifier is sure = safe to automate
- Low confidence = classifier is uncertain = human should verify

The threshold (0.7) is configurable. We'll tune it based on real-world accuracy.

**Alternatives considered:**
- **Always require approval:** Too slow. Defeats the purpose of automation.
- **Never require approval:** Too risky. Misclassified books would process incorrectly.
- **Stop after classification only:** Delays processing. Better to process and validate together.

---

### Decision 3: Why NEEDS_RETRY Instead of Immediate Retry?

**What we chose:** On failure, mark as NEEDS_RETRY and move to the next book.

**Why:**

Keeps the queue flowing. If a book fails due to a temporary issue (API timeout, disk full), retrying later often succeeds. If we retry immediately:
- Same error probably recurs
- Other books wait unnecessarily
- System can get stuck in retry loops

**Retry strategy:**
- Failed books go to NEEDS_RETRY
- Separate `retry` command processes them later
- After 3 attempts, mark as REJECTED (human intervention needed)

---

### Decision 4: Why Graceful Shutdown?

**What we chose:** On Ctrl+C, finish the current book before stopping.

**Why:**

Interrupting mid-processing leaves books in inconsistent states:
- Partially extracted text
- State says PROCESSING but nothing is running
- Requires manual cleanup

With graceful shutdown:
```
User hits Ctrl+C
→ Orchestrator sets shutdown flag
→ Current book finishes normally
→ Worker exits cleanly
→ No orphaned state
```

---

## State-by-State Logic

### DETECTED → HASHING

```python
# Compute content hash (SHA-256 of file contents)
content_hash = hash_file(book_path)

# Idempotency check: already seen this hash?
existing = repo.find_by_hash(content_hash)
if existing:
    if existing["state"] == "complete":
        return "Already processed"  # Skip
    if existing["state"] not in TERMINAL_STATES:
        return "Already in progress"  # Skip

# Create pipeline record
pipeline_id = repo.create(book_path, content_hash)
transition(pipeline_id, HASHING)
```

### HASHING → CLASSIFYING

```python
# Check for duplicate (same hash, different file)
if is_duplicate(content_hash):
    transition(pipeline_id, DUPLICATE)
    return

transition(pipeline_id, CLASSIFYING)
```

### CLASSIFYING → SELECTING_STRATEGY

```python
# Extract text sample (first ~40K chars)
text_sample = extract_sample(book_path)

# Call ClassifierAgent (Phase 2)
profile = classifier.classify(text_sample, content_hash)

# Store result
repo.update_book_profile(pipeline_id, profile.to_dict())
transition(pipeline_id, SELECTING_STRATEGY)
```

### SELECTING_STRATEGY → PROCESSING

```python
# Pick strategy based on book type + confidence
strategy = strategy_selector.select(profile.to_dict())
repo.update_strategy_config(pipeline_id, strategy)
transition(pipeline_id, PROCESSING)
```

### PROCESSING (subprocess with timeout)

```python
transition(pipeline_id, PROCESSING)

try:
    result = subprocess.run(
        ["python", "-m", "src.cli", "process", book_path],
        cwd=config.book_ingestion_path,
        timeout=config.processing_timeout,  # 10 minutes default
        capture_output=True
    )

    if result.returncode != 0:
        raise ProcessingError(result.stderr)

    transition(pipeline_id, VALIDATING)

except subprocess.TimeoutExpired:
    log.error(f"Processing timed out for {book_path}")
    transition(pipeline_id, NEEDS_RETRY)

except ProcessingError as e:
    log.error(f"Processing failed: {e}")
    transition(pipeline_id, NEEDS_RETRY)
```

### VALIDATING → PENDING_APPROVAL or APPROVED

```python
transition(pipeline_id, VALIDATING)

# Check processing results in shared database
chapters = db.get_chapters_for_book(content_hash)

if not chapters:
    transition(pipeline_id, NEEDS_RETRY)
    return

# Store validation result
repo.update_validation_result(pipeline_id, {
    "chapter_count": len(chapters),
    "total_words": sum(c["word_count"] for c in chapters)
})

# Route based on confidence
confidence = profile.get("confidence", 0)

if confidence >= config.confidence_threshold:
    repo.mark_approved(pipeline_id, approved_by="auto:high_confidence")
    transition(pipeline_id, APPROVED)
else:
    transition(pipeline_id, PENDING_APPROVAL)
```

### APPROVED → EMBEDDING → COMPLETE

```python
transition(pipeline_id, EMBEDDING)

try:
    result = subprocess.run(
        ["python", "-m", "scripts.generate_embeddings", "--book-hash", content_hash],
        cwd=config.book_ingestion_path,
        timeout=config.embedding_timeout,  # 5 minutes default
        capture_output=True
    )

    if result.returncode != 0:
        raise EmbeddingError(result.stderr)

    transition(pipeline_id, COMPLETE)

except (subprocess.TimeoutExpired, EmbeddingError):
    transition(pipeline_id, NEEDS_RETRY)
```

---

## Configuration

### Environment Variables

```bash
# Required
AGENTIC_PIPELINE_DB="/path/to/library.db"
OPENAI_API_KEY="sk-..."

# Optional (with defaults)
ANTHROPIC_API_KEY="sk-ant-..."           # For fallback classifier
BOOK_INGESTION_PATH="/path/to/book-ingestion-python"
PROCESSING_TIMEOUT_SECONDS=600            # 10 minutes
EMBEDDING_TIMEOUT_SECONDS=300             # 5 minutes
CONFIDENCE_THRESHOLD=0.7                  # Auto-approve threshold
WORKER_POLL_INTERVAL_SECONDS=5            # Queue check frequency
MAX_RETRY_ATTEMPTS=3                      # Before marking rejected
```

### Config Class

```python
@dataclass
class OrchestratorConfig:
    db_path: Path
    book_ingestion_path: Path
    processing_timeout: int = 600
    embedding_timeout: int = 300
    confidence_threshold: float = 0.7
    worker_poll_interval: int = 5
    max_retry_attempts: int = 3

    @classmethod
    def from_env(cls) -> "OrchestratorConfig":
        # Load from environment variables with defaults
        ...
```

---

## Logging & Observability

### Structured Logging

All events logged as JSON for easy parsing:

```json
{
  "event": "state_transition",
  "pipeline_id": "abc123",
  "from": "CLASSIFYING",
  "to": "SELECTING_STRATEGY",
  "timestamp": "2025-01-28T12:34:56Z"
}
```

### Event Types

| Event | When |
|-------|------|
| `processing_started` | Book begins processing |
| `state_transition` | State changes |
| `processing_complete` | Book reaches COMPLETE |
| `error` | Any failure |
| `retry_scheduled` | Book moved to NEEDS_RETRY |
| `worker_started` | Queue worker begins |
| `worker_stopped` | Queue worker exits |

### CLI Status Commands

```bash
# View recent activity
agentic-pipeline logs --tail 50

# View specific pipeline
agentic-pipeline status abc123

# Output:
# Pipeline: abc123
# State: PROCESSING
# Book: /path/to/book.epub
# Type: technical_tutorial (90% confidence)
# Duration: 45s (processing...)
# Retries: 0
```

---

## Database Changes

### New Column: retry_count

```sql
ALTER TABLE processing_pipelines ADD COLUMN retry_count INTEGER DEFAULT 0;
```

### New Repository Method

```python
def increment_retry_count(self, pipeline_id: str) -> int:
    """Increment retry count and return new value."""
    ...
```

---

## Testing Approach

### Unit Tests (mocked subprocess)

| Test | Verifies |
|------|----------|
| `test_process_book_success` | Happy path through all states |
| `test_process_book_timeout` | Timeout → NEEDS_RETRY |
| `test_idempotency_already_complete` | Completed books skipped |
| `test_idempotency_in_progress` | In-progress books skipped |
| `test_graceful_shutdown` | Worker finishes current book |
| `test_retry_max_attempts` | Exceeds retries → REJECTED |
| `test_low_confidence_needs_approval` | <70% → PENDING_APPROVAL |
| `test_high_confidence_auto_approves` | ≥70% → APPROVED |

### Integration Test

```python
@pytest.mark.integration
def test_full_pipeline_with_real_book(sample_epub):
    """Test full pipeline with a real book file."""
    result = orchestrator.process_one(sample_epub)

    assert result.state == PipelineState.COMPLETE
    assert result.chapter_count > 0
    assert result.has_embeddings == True
```

---

## File Structure

```
agentic_pipeline/
├── __init__.py
├── cli.py                    # CLI entry points (add: process, worker, retry)
├── config.py                 # NEW: OrchestratorConfig
├── logging.py                # NEW: PipelineLogger
├── mcp_server.py             # MCP tools (add: process_book)
│
├── agents/
│   ├── classifier.py         # Phase 2: ClassifierAgent
│   ├── classifier_types.py   # BookType, BookProfile
│   └── providers/            # OpenAI, Anthropic
│
├── db/
│   ├── config.py
│   ├── migrations.py         # UPDATE: add retry_count migration
│   └── pipelines.py          # UPDATE: add increment_retry_count()
│
├── orchestrator/             # NEW: Phase 3
│   ├── __init__.py           # Exports: Orchestrator
│   ├── orchestrator.py       # Main class: process_one(), run_worker()
│   ├── processors.py         # State handlers: hash, classify, process, embed
│   └── errors.py             # ProcessingError, EmbeddingError, TimeoutError
│
└── pipeline/
    ├── states.py             # PipelineState enum (exists)
    └── strategy.py           # StrategySelector (exists)
```

---

## Entry Points

### CLI Commands

```bash
# Process a single book
agentic-pipeline process /path/to/book.epub

# Run queue worker (continuous)
agentic-pipeline worker

# Retry failed books
agentic-pipeline retry --max-attempts 3

# Check status
agentic-pipeline status <pipeline-id>

# View logs
agentic-pipeline logs --tail 50
```

### MCP Tools

```python
@mcp.tool()
def process_book(path: str) -> dict:
    """Process a book through the pipeline."""
    orchestrator = Orchestrator(get_db_path())
    result = orchestrator.process_one(path)
    return {
        "pipeline_id": result.pipeline_id,
        "state": result.state,
        "book_type": result.book_type,
        "confidence": result.confidence
    }

@mcp.tool()
def get_pipeline_status(pipeline_id: str) -> dict:
    """Get status of a pipeline run."""
    repo = PipelineRepository(get_db_path())
    return repo.get(pipeline_id)
```

---

## What This Enables

With the orchestrator in place, books become searchable for:

| Use Case | Example Query |
|----------|---------------|
| **Teaching** | "Explain recursion using examples from my Python books" |
| **Explaining** | "What does Chapter 5 of Clean Code say about functions?" |
| **Research** | "Find all mentions of microservices across my library" |
| **Design** | "Based on my architecture books, what patterns fit this problem?" |

The flow:
```
Book.epub → Orchestrator → Chapters + Embeddings → SQLite
                                                      ↓
Claude asks question → MCP Server → Semantic Search → Relevant excerpts
                                                      ↓
                                          Claude teaches/explains/designs
```

---

## Cost Projections

| Component | Cost per Book |
|-----------|---------------|
| Classification (OpenAI) | ~$0.001 |
| Processing (local) | $0.00 |
| Embedding (local) | $0.00 |
| **Total** | **~$0.001** |

At 100 books/month: ~$0.10/month for AI costs.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Subprocess hangs indefinitely | Low | Medium | Timeouts (10 min processing, 5 min embedding) |
| Book-ingestion-python path changes | Low | Low | Configurable via environment variable |
| Too many books in NEEDS_RETRY | Medium | Low | Max retry limit, then REJECTED for human review |
| Worker crashes mid-book | Low | Medium | Graceful shutdown; book stays in current state |

---

## Success Metrics

| Metric | Target | How We Measure |
|--------|--------|----------------|
| End-to-end success rate | >90% | Books reaching COMPLETE / total |
| Auto-approval rate | >80% | Books auto-approved / total approved |
| Processing time | <5 min/book | Average time DETECTED → COMPLETE |
| Retry rate | <10% | Books hitting NEEDS_RETRY / total |

---

## Summary

The Pipeline Orchestrator is the glue that connects classification, processing, and approval into an automated flow. It's:

- **Reliable** — Timeouts, retries, graceful shutdown
- **Observable** — Structured logging, status commands
- **Configurable** — Environment variables for all thresholds
- **Testable** — Mocked subprocess for unit tests

It doesn't try to be clever. It moves books through states, handles failures gracefully, and gets out of the way.

---

## Next Steps

1. **Implementation** — Build the orchestrator (~3-4 days)
2. **Integration** — Connect to existing book-ingestion-python
3. **Testing** — Unit tests + real book integration test
4. **Deployment** — CLI commands, MCP tools

Ready for implementation plan? See `docs/plans/2025-01-28-phase3-orchestrator-implementation.md` (to be created).

# Phase 4: Production Hardening Design

> **Audience:** Product managers, directors, UX designers, and engineers who need to understand what we're building and why.

---

## Status: 📋 Planned

**Target:** Next implementation phase after Phase 3 (Pipeline Orchestrator)

**Dependencies:** Phase 3 complete (Orchestrator with process_one, run_worker, retry)

---

## Executive Summary

Phase 4 transforms the pipeline from "works on my machine" to "runs reliably in production." We're adding the operational features that make the difference between a prototype and a tool you can trust to run unattended.

**The Problem:** The Phase 3 orchestrator processes books correctly, but it lacks visibility into what's happening. If a book gets stuck, you won't know until you manually check. If you need to process 500 books urgently, there's no way to prioritize them over routine ingestion.

**The Solution:** Health monitoring, stuck detection, batch operations, priority queues, and a complete audit trail.

---

## What Is This?

Phase 4 adds four production capabilities:

| Capability | What It Does | Why It Matters |
|------------|--------------|----------------|
| **Health Monitoring** | Dashboard showing active/stuck/queued books | Know at a glance if the system is healthy |
| **Stuck Detection** | Automatically find books that stopped progressing | Catch problems before they pile up |
| **Batch Operations** | Process/approve/reject multiple books at once | 10x faster for bulk library imports |
| **Priority Queues** | Process urgent books first | Time-sensitive materials don't wait |
| **Audit Trail** | Immutable log of all decisions | Answer "who approved this and when?" |

---

## How It Works

### The Current State (Phase 3)

```
┌─────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│   process_one() ─────► Book processed ────► Complete    │
│                                                          │
│   run_worker() ─────► Polls queue ────► Processes next  │
│                                                          │
│   ❌ No visibility into stuck books                      │
│   ❌ No batch operations                                 │
│   ❌ No priority control                                 │
│   ❌ Limited audit trail                                 │
└─────────────────────────────────────────────────────────┘
```

### The Phase 4 State

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR + PRODUCTION FEATURES                │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                     HEALTH MONITOR                             │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │  │
│  │  │ Active  │  │  Stuck  │  │ Queued  │  │  Alerts │          │  │
│  │  │    3    │  │    1    │  │   47    │  │    ⚠️   │          │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘          │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                     PRIORITY QUEUE                             │  │
│  │                                                                │  │
│  │  Priority 1 (urgent):     ████░░░░░░  2 books                 │  │
│  │  Priority 5 (normal):     ██████████  45 books                │  │
│  │  Priority 10 (backfill):  ████████░░  8 books                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                     BATCH OPERATIONS                           │  │
│  │                                                                │  │
│  │  batch_process(folder, priority) → Queue all books            │  │
│  │  batch_approve(filter, min_confidence) → Approve matching     │  │
│  │  batch_reject(filter, reason) → Reject with explanation       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                      AUDIT TRAIL                               │  │
│  │                                                                │  │
│  │  2025-01-28 14:32:15  book_abc123  APPROVED  auto:high_conf   │  │
│  │  2025-01-28 14:30:02  book_def456  REJECTED  human:taylor     │  │
│  │  2025-01-28 14:28:44  book_ghi789  APPROVED  human:taylor     │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### Decision 1: Why Real-Time Health Monitoring Instead of Just Logs?

**What we chose:** A dedicated health endpoint that returns current system state.

**The Options We Considered:**

| Approach | Pros | Cons |
|----------|------|------|
| **Grep logs** | No new code | Slow, requires shell access, hard to automate |
| **Database queries** | Already available | Slow for complex aggregations, no real-time view |
| **Health endpoint** ✓ | Fast, automatable, MCP-friendly | Requires new code |
| **External monitoring** | Industry standard | Overkill for single-user system |

**Why we chose the health endpoint:**

1. **MCP Integration** — Claude can call `get_pipeline_health()` and immediately know if something needs attention
2. **Scriptable** — CI/CD or cron can hit the endpoint and alert on anomalies
3. **Fast** — Pre-computed metrics, not ad-hoc queries
4. **Actionable** — Returns stuck pipeline IDs, not just counts

**What this looks like:**

```python
# CLI
$ agentic-pipeline health

Pipeline Health
─────────────────────────────────
  Active:     3 (processing now)
  Stuck:      1 ⚠️ (pipeline abc123, stuck 45 min)
  Queued:     47 (waiting)
  Completed:  892 (last 24h)
  Failed:     3 (needs_retry)
─────────────────────────────────
  Avg time:   3.2 min/book
  Queue wait: ~2.5 hours
```

```python
# MCP
get_pipeline_health()
→ {
    "active": 3,
    "stuck": [{"id": "abc123", "state": "processing", "stuck_minutes": 45}],
    "queued": 47,
    "completed_24h": 892,
    "failed": 3,
    "avg_processing_seconds": 192,
    "estimated_queue_minutes": 150
  }
```

---

### Decision 2: How Do We Define "Stuck"?

**What we chose:** A book is stuck if it's been in a non-terminal state longer than 2x the expected duration for that state.

**The Options We Considered:**

| Approach | Pros | Cons |
|----------|------|------|
| **Fixed timeout** (e.g., 30 min) | Simple | Doesn't account for state differences |
| **Per-state timeout** (e.g., PROCESSING=15min, EMBEDDING=10min) | More accurate | Requires tuning, rigid |
| **Heartbeat-based** (must update every N seconds) | Very accurate | Complex, requires code in worker |
| **Statistical** (2x expected duration) ✓ | Adapts to real performance | Needs history to be accurate |

**Why we chose statistical detection:**

1. **Self-tuning** — As the system learns actual processing times, thresholds adjust automatically
2. **Per-state awareness** — CLASSIFYING (2 seconds) has different expectations than PROCESSING (5 minutes)
3. **No false positives** — A book that takes 4 minutes instead of 3 isn't stuck; one that takes 30 is
4. **Graceful startup** — Reasonable defaults until we have enough data

**How it works:**

```python
# Default timeouts (used until we have enough data)
DEFAULT_STATE_TIMEOUTS = {
    "HASHING": 60,           # 1 minute
    "CLASSIFYING": 120,      # 2 minutes
    "SELECTING_STRATEGY": 30, # 30 seconds
    "PROCESSING": 900,       # 15 minutes
    "VALIDATING": 60,        # 1 minute
    "EMBEDDING": 600,        # 10 minutes
}

# After 100+ books, use: median_duration * 2 for each state
```

**Alternatives we rejected:**

- **Heartbeat system:** Would require modifying the worker to send periodic updates. More accurate but adds complexity and potential failure modes. For a single-user system, the statistical approach is simpler and sufficient.

- **Fixed global timeout:** A 30-minute timeout would miss stuck books in CLASSIFYING (should take seconds) and create false alarms for large books in PROCESSING (legitimately takes 20 minutes).

---

### Decision 3: Why Priority Queues Instead of Simple FIFO?

**What we chose:** Books have a priority field (1-10). Lower numbers process first.

**The Options We Considered:**

| Approach | Pros | Cons |
|----------|------|------|
| **FIFO** (first in, first out) | Simple, fair | Urgent items wait behind routine ones |
| **Priority queue** ✓ | Urgent items processed first | Starvation risk for low-priority items |
| **Multiple queues** (urgent, normal, backfill) | Clear separation | Complex, hard to rebalance |
| **Deadline-based** | Guaranteed completion time | Complex, requires deadline estimates |

**Why we chose priority queues:**

1. **Common scenario:** "I just got these 3 new AI books for a project deadline—process them NOW, not after the 50 backlog items"
2. **Simple mental model:** 1 = urgent, 5 = normal, 10 = whenever
3. **Database-native:** Just an ORDER BY clause, no complex queue infrastructure
4. **Starvation prevention:** We age priorities over time (a book waiting 24 hours at priority 10 becomes priority 5)

**How it works:**

```bash
# Queue a folder with high priority
$ agentic-pipeline batch-queue /path/to/urgent-books --priority 1

# Queue a folder for background processing
$ agentic-pipeline batch-queue /path/to/old-archives --priority 10

# Worker processes: priority ASC, then created_at ASC
# So: priority 1 books first, then 2, then 3... within same priority, oldest first
```

**Starvation prevention:**

```python
# Every hour, boost priority of old items
def age_priorities():
    """Increase priority of items waiting too long."""
    # Items waiting > 24 hours: priority = max(1, priority - 3)
    # Items waiting > 48 hours: priority = 1
```

**Alternatives we rejected:**

- **Multiple named queues:** Creates cognitive overhead ("Is this urgent-urgent or just urgent?"). A simple 1-10 scale is more intuitive.

- **Deadline-based:** Requires knowing how long a book will take to process, which varies wildly. An 800-page textbook takes 10x longer than a 100-page novel.

---

### Decision 4: How Should Batch Operations Work?

**What we chose:** Filter-based batch operations with preview and confirmation.

**The Options We Considered:**

| Approach | Pros | Cons |
|----------|------|------|
| **Explicit list** (batch-approve id1 id2 id3) | Very precise | Tedious for large batches |
| **Filter-based** (batch-approve --min-confidence 0.9) ✓ | Powerful, concise | Could accidentally affect unintended items |
| **Interactive** (show list, select items) | Visual confirmation | Doesn't work well in scripts/MCP |
| **All-or-nothing** (approve-all) | Fast | Dangerous |

**Why we chose filter-based with preview:**

1. **Expressive:** "Approve all technical tutorials with >90% confidence" is one command
2. **Safe by default:** Shows what will be affected before doing it
3. **MCP-friendly:** Filters translate well to tool parameters
4. **Scriptable:** Can be automated once you trust the filters

**How it works:**

```bash
# Preview what would be approved
$ agentic-pipeline batch-approve --min-confidence 0.9 --book-type technical_tutorial --dry-run

Would approve 12 books:
  abc123  "Learning Python" (92% confidence)
  def456  "React in Action" (95% confidence)
  ...

# Actually approve them
$ agentic-pipeline batch-approve --min-confidence 0.9 --book-type technical_tutorial

Approved 12 books.
```

```python
# MCP
batch_approve(
    min_confidence=0.9,
    book_type="technical_tutorial",
    max_count=50  # safety limit
)
→ {"approved": 12, "skipped": 0, "details": [...]}
```

**Safety features:**

- **Dry-run by default in CLI:** Must add `--execute` to actually do it
- **Max count limit:** Won't approve more than N books at once (default 50)
- **Confirmation for MCP:** Returns preview first, requires second call to execute
- **Audit trail:** Every batch operation logged with filters used

---

### Decision 5: What Goes in the Audit Trail?

**What we chose:** Immutable append-only log of all approval decisions and system actions.

**The Options We Considered:**

| Approach | Pros | Cons |
|----------|------|------|
| **No audit** | Simple | Can't answer "who did what when" |
| **Log files** | Easy to implement | Hard to query, can be edited/deleted |
| **Database table** ✓ | Queryable, tamper-resistant | Takes storage |
| **Blockchain/WORM** | Truly immutable | Massive overkill |

**Why we chose a database table:**

1. **Queryable:** "Show me all books approved by the auto-approve system in the last week"
2. **Tamper-resistant:** SQLite in WAL mode with proper permissions is sufficient for personal use
3. **Retention policies:** Can auto-clean old entries while preserving recent history
4. **MCP integration:** Claude can query audit history to understand decisions

**What we capture:**

```sql
CREATE TABLE approval_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id TEXT NOT NULL,          -- Which book
    pipeline_id TEXT,               -- Pipeline run ID
    action TEXT NOT NULL,           -- APPROVED, REJECTED, ROLLED_BACK, etc.
    actor TEXT NOT NULL,            -- "auto:high_confidence", "human:taylor", "batch:filter_xyz"
    reason TEXT,                    -- Why (especially for rejections)
    before_state JSON,              -- State before action
    after_state JSON,               -- State after action
    confidence_at_decision REAL,    -- Classifier confidence when decided
    performed_at TIMESTAMP          -- When
);
```

**What we DON'T capture:**

- File contents (too large, already in book table)
- Embeddings (too large, already in chunks table)
- Intermediate processing steps (covered by state_history table)

**Retention policy:**

```python
AUDIT_RETENTION = {
    "APPROVED": 365,      # Keep 1 year
    "REJECTED": 365,      # Keep 1 year
    "ROLLED_BACK": 730,   # Keep 2 years (important for debugging)
    "BATCH_*": 90,        # Keep 3 months
}
```

---

## Component Details

### 1. Health Monitor

**Purpose:** Real-time visibility into pipeline health.

**Implementation:**

```python
class HealthMonitor:
    """Aggregates pipeline health metrics."""

    def __init__(self, db_path: Path):
        self.repo = PipelineRepository(db_path)
        self.state_thresholds = StateThresholds(db_path)

    def get_health(self) -> HealthReport:
        """Get current system health."""
        return HealthReport(
            active=self._count_active(),
            stuck=self._find_stuck(),
            queued=self._count_queued(),
            completed_24h=self._count_completed(hours=24),
            failed=self._count_needs_retry(),
            avg_processing_seconds=self._avg_processing_time(),
            queue_depth_by_priority=self._queue_by_priority(),
            alerts=self._generate_alerts()
        )

    def _find_stuck(self) -> list[StuckPipeline]:
        """Find pipelines stuck longer than expected."""
        active = self.repo.find_active()
        stuck = []

        for pipeline in active:
            expected_duration = self.state_thresholds.get(pipeline["state"])
            actual_duration = time_since(pipeline["updated_at"])

            if actual_duration > expected_duration * 2:
                stuck.append(StuckPipeline(
                    id=pipeline["id"],
                    state=pipeline["state"],
                    stuck_minutes=actual_duration.minutes,
                    expected_minutes=expected_duration.minutes
                ))

        return stuck
```

**Alerts generated:**

| Alert | Trigger | Severity |
|-------|---------|----------|
| `stuck_pipeline` | Any pipeline stuck >2x expected | Warning |
| `high_failure_rate` | >20% of last 100 books failed | Warning |
| `queue_backup` | >100 books queued | Info |
| `worker_idle` | Worker running but no processing for 1 hour | Info |
| `disk_space_low` | <1GB free in database directory | Critical |

---

### 2. Stuck Detection & Recovery

**Purpose:** Automatically identify and optionally recover stuck pipelines.

**Detection logic:**

```python
def detect_stuck_pipelines() -> list[StuckPipeline]:
    """Find pipelines that have been in the same state too long."""

    # Get expected durations (from history or defaults)
    thresholds = calculate_state_thresholds()

    # Find active (non-terminal) pipelines
    active = repo.find_by_states(NON_TERMINAL_STATES)

    stuck = []
    for pipeline in active:
        threshold = thresholds[pipeline["state"]]
        age = now() - pipeline["updated_at"]

        if age > threshold * STUCK_MULTIPLIER:  # Default: 2x
            stuck.append(StuckPipeline(
                id=pipeline["id"],
                state=pipeline["state"],
                stuck_since=pipeline["updated_at"],
                expected_duration=threshold,
                actual_duration=age
            ))

    return stuck
```

**Recovery options:**

| Action | When to Use | Risk Level |
|--------|-------------|------------|
| **Alert only** | Default; let human decide | None |
| **Reset to DETECTED** | Subprocess likely crashed | Low |
| **Mark as NEEDS_RETRY** | Transient failure suspected | Low |
| **Mark as REJECTED** | Repeated failures | Medium |

**Automatic recovery rules (configurable):**

```python
STUCK_RECOVERY_RULES = {
    "PROCESSING": {
        "after_minutes": 60,
        "action": "reset_to_needs_retry",
        "max_auto_recoveries": 2  # Then alert human
    },
    "EMBEDDING": {
        "after_minutes": 30,
        "action": "reset_to_needs_retry",
        "max_auto_recoveries": 2
    },
    # Other states: alert only, don't auto-recover
}
```

---

### 3. Batch Operations

**Purpose:** Process multiple books efficiently.

**Operations:**

| Operation | Description | Example |
|-----------|-------------|---------|
| `batch_queue` | Add multiple books to queue | Queue entire folder |
| `batch_approve` | Approve matching books | All technical tutorials >90% |
| `batch_reject` | Reject matching books | All newspapers |
| `batch_retry` | Retry failed books | All NEEDS_RETRY |
| `batch_set_priority` | Change priority | Urgent folder → priority 1 |

**Filter parameters (all optional):**

```python
@dataclass
class BatchFilter:
    book_type: str = None           # "technical_tutorial", "textbook", etc.
    min_confidence: float = None    # 0.0 - 1.0
    max_confidence: float = None    # 0.0 - 1.0
    state: str = None               # "pending_approval", "needs_retry"
    created_before: datetime = None
    created_after: datetime = None
    source_path_pattern: str = None # Glob pattern
    max_count: int = 50             # Safety limit
```

**CLI examples:**

```bash
# Queue all EPUBs in a folder with high priority
$ agentic-pipeline batch-queue /path/to/books --pattern "*.epub" --priority 2

# Approve all high-confidence technical books
$ agentic-pipeline batch-approve \
    --book-type technical_tutorial \
    --min-confidence 0.9 \
    --execute

# Reject all newspapers with explanation
$ agentic-pipeline batch-reject \
    --book-type newspaper \
    --reason "Not ingesting periodicals in this library" \
    --execute

# Retry all failed books
$ agentic-pipeline batch-retry --max-attempts 3
```

---

### 4. Priority Queues

**Purpose:** Control processing order.

**Priority scale:**

| Priority | Meaning | Use Case |
|----------|---------|----------|
| 1 | Urgent | Need this for a meeting in an hour |
| 2 | High | Time-sensitive project work |
| 3-4 | Above normal | Important but not urgent |
| 5 | Normal | Default for new books |
| 6-7 | Below normal | Nice to have |
| 8-9 | Low | Background processing |
| 10 | Backfill | Historical archives, process whenever |

**Queue ordering:**

```sql
-- Worker fetches next book to process
SELECT * FROM processing_pipelines
WHERE state = 'DETECTED'
ORDER BY
    priority ASC,           -- Lower priority number = more urgent
    created_at ASC          -- Within same priority, oldest first
LIMIT 1;
```

**Priority aging (anti-starvation):**

```python
def age_queue_priorities():
    """Prevent low-priority items from waiting forever."""

    # Items waiting >24h: boost priority by 2
    repo.execute("""
        UPDATE processing_pipelines
        SET priority = MAX(1, priority - 2)
        WHERE state = 'DETECTED'
        AND created_at < datetime('now', '-24 hours')
        AND priority > 1
    """)

    # Items waiting >48h: set to priority 1
    repo.execute("""
        UPDATE processing_pipelines
        SET priority = 1
        WHERE state = 'DETECTED'
        AND created_at < datetime('now', '-48 hours')
    """)
```

---

### 5. Audit Trail

**Purpose:** Immutable record of all decisions.

**Events captured:**

| Event | Actor Types | Data Captured |
|-------|-------------|---------------|
| APPROVED | auto:high_confidence, human:taylor, batch:filter_abc | confidence, book_type |
| REJECTED | human:taylor, batch:filter_abc | reason, confidence |
| ROLLED_BACK | human:taylor | reason, original_approval |
| PRIORITY_CHANGED | human:taylor, system:aging | old_priority, new_priority |
| BATCH_QUEUED | human:taylor | count, source_path, priority |
| STUCK_RECOVERED | system:health_monitor | recovery_action, stuck_duration |

**Querying the audit trail:**

```bash
# Recent activity
$ agentic-pipeline audit --last 50

# Activity by actor
$ agentic-pipeline audit --actor "auto:high_confidence" --last-days 7

# Activity for a specific book
$ agentic-pipeline audit --book-id abc123
```

```python
# MCP
get_audit_log(
    actor="human:taylor",
    action="APPROVED",
    last_days=7
)
→ [
    {"book_id": "abc123", "action": "APPROVED", "actor": "human:taylor", ...},
    ...
  ]
```

---

## Database Changes

### New Tables

```sql
-- Health metrics cache (refreshed periodically)
CREATE TABLE health_metrics (
    id INTEGER PRIMARY KEY CHECK (id = 1),  -- Singleton
    active_count INTEGER NOT NULL,
    queued_count INTEGER NOT NULL,
    stuck_count INTEGER NOT NULL,
    completed_24h INTEGER NOT NULL,
    failed_count INTEGER NOT NULL,
    avg_processing_seconds REAL,
    queue_by_priority JSON,
    stuck_pipelines JSON,
    alerts JSON,
    updated_at TIMESTAMP NOT NULL
);

-- State duration statistics (for stuck detection)
CREATE TABLE state_duration_stats (
    state TEXT PRIMARY KEY,
    sample_count INTEGER NOT NULL,
    median_seconds REAL NOT NULL,
    p95_seconds REAL NOT NULL,
    max_seconds REAL NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

-- Audit trail (append-only)
CREATE TABLE approval_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id TEXT NOT NULL,
    pipeline_id TEXT,
    action TEXT NOT NULL,
    actor TEXT NOT NULL,
    reason TEXT,
    before_state JSON,
    after_state JSON,
    filter_used JSON,  -- For batch operations
    confidence_at_decision REAL,
    performed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_audit_book_id ON approval_audit(book_id);
CREATE INDEX idx_audit_actor ON approval_audit(actor);
CREATE INDEX idx_audit_performed_at ON approval_audit(performed_at);
```

### Column Additions

```sql
-- Priority support in pipelines table
ALTER TABLE processing_pipelines ADD COLUMN priority INTEGER DEFAULT 5;

-- Recovery tracking
ALTER TABLE processing_pipelines ADD COLUMN auto_recovery_count INTEGER DEFAULT 0;
ALTER TABLE processing_pipelines ADD COLUMN last_heartbeat TIMESTAMP;

-- Index for priority queue
CREATE INDEX idx_pipelines_priority_queue ON processing_pipelines(state, priority, created_at);
```

---

## CLI Commands

### New Commands

```bash
# Health monitoring
agentic-pipeline health                    # Show system health
agentic-pipeline health --watch           # Live updating display
agentic-pipeline health --json            # JSON output for scripts

# Stuck detection
agentic-pipeline stuck                    # List stuck pipelines
agentic-pipeline stuck --recover          # Auto-recover stuck pipelines
agentic-pipeline stuck --reset <id>       # Reset specific pipeline

# Batch operations
agentic-pipeline batch-queue <path> [--priority N] [--pattern GLOB]
agentic-pipeline batch-approve [--filters...] [--execute]
agentic-pipeline batch-reject [--filters...] --reason "..." [--execute]
agentic-pipeline batch-retry [--max-attempts N]
agentic-pipeline batch-priority <priority> [--filters...]

# Audit trail
agentic-pipeline audit [--last N] [--actor X] [--action Y] [--book-id Z]
```

### Enhanced Existing Commands

```bash
# Process with priority
agentic-pipeline process /path/to/book.epub --priority 1

# Worker with health reporting
agentic-pipeline worker --health-interval 60  # Report health every 60s

# Status with stuck warning
agentic-pipeline status <id>  # Shows "⚠️ STUCK" if applicable
```

---

## MCP Tools

### New Tools

```python
@mcp.tool()
def get_pipeline_health() -> dict:
    """Get current system health including active, stuck, queued counts."""

@mcp.tool()
def get_stuck_pipelines() -> list[dict]:
    """Get list of pipelines that appear to be stuck."""

@mcp.tool()
def recover_stuck_pipeline(pipeline_id: str, action: str = "retry") -> dict:
    """Recover a stuck pipeline. Actions: retry, reject, reset."""

@mcp.tool()
def batch_queue_folder(
    path: str,
    priority: int = 5,
    pattern: str = "*"
) -> dict:
    """Queue all matching books in a folder for processing."""

@mcp.tool()
def batch_approve(
    min_confidence: float = None,
    book_type: str = None,
    max_count: int = 50,
    execute: bool = False
) -> dict:
    """Approve books matching filters. Set execute=True to apply."""

@mcp.tool()
def batch_reject(
    book_type: str = None,
    max_confidence: float = None,
    reason: str = "",
    max_count: int = 50,
    execute: bool = False
) -> dict:
    """Reject books matching filters. Set execute=True to apply."""

@mcp.tool()
def set_pipeline_priority(pipeline_id: str, priority: int) -> dict:
    """Change priority of a queued pipeline."""

@mcp.tool()
def get_audit_log(
    book_id: str = None,
    actor: str = None,
    action: str = None,
    last_days: int = 7,
    limit: int = 100
) -> list[dict]:
    """Query the audit trail."""
```

---

## Testing Strategy

### Unit Tests

| Test | Verifies |
|------|----------|
| `test_health_report_generation` | Health metrics are calculated correctly |
| `test_stuck_detection_thresholds` | Stuck detection uses correct thresholds |
| `test_stuck_detection_ignores_terminal` | Completed books aren't flagged as stuck |
| `test_priority_queue_ordering` | Higher priority processed first |
| `test_priority_aging` | Old low-priority items get boosted |
| `test_batch_approve_filters` | Filter logic works correctly |
| `test_batch_approve_dry_run` | Dry run doesn't modify data |
| `test_batch_approve_max_count` | Safety limit is respected |
| `test_audit_trail_immutable` | Audit entries can't be modified |
| `test_audit_trail_captures_all_actions` | All decision types logged |

### Integration Tests

| Test | Verifies |
|------|----------|
| `test_health_during_processing` | Active count reflects actual work |
| `test_stuck_detection_with_timeout` | Timed-out processing detected as stuck |
| `test_batch_queue_folder` | Folder of books queued correctly |
| `test_priority_processing_order` | Worker respects priority order |
| `test_full_audit_trail` | Complete flow captured in audit |

---

## Success Metrics

| Metric | Target | How We Measure |
|--------|--------|----------------|
| Stuck detection accuracy | 100% | All stuck books found within 2x expected duration |
| False positive rate | <5% | Books flagged as stuck that weren't |
| Batch operation speed | <1s per 100 books | Time to queue/approve/reject batches |
| Audit completeness | 100% | Every approval/rejection has audit entry |
| Queue fairness | No item waits >48h | Priority aging prevents starvation |

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Aggressive stuck detection (false positives) | Medium | Low | Conservative defaults (2x expected), easy to tune |
| Batch operation mistakes | Medium | Medium | Dry-run default, max count limits, audit trail |
| Audit table grows too large | Low | Low | Retention policy, auto-cleanup |
| Priority starvation | Low | Medium | Automatic priority aging |
| Health check becomes slow | Low | Low | Cached metrics, async refresh |

---

## What This Enables

With Phase 4, you can:

1. **Trust unattended operation:** "Is everything OK?" → Check health dashboard
2. **Handle bulk imports:** "I have 200 new books" → One batch-queue command
3. **Prioritize urgent work:** "I need this book for tomorrow" → Priority 1
4. **Debug problems:** "Why was this rejected?" → Check audit trail
5. **Recover from issues:** "Something's stuck" → Auto-recovery or manual reset

---

## File Structure

```
agentic_pipeline/
├── health/                     # NEW: Phase 4
│   ├── __init__.py
│   ├── monitor.py              # HealthMonitor class
│   ├── stuck_detector.py       # StuckDetector, StateThresholds
│   └── recovery.py             # StuckRecovery actions
│
├── batch/                      # NEW: Phase 4
│   ├── __init__.py
│   ├── operations.py           # BatchQueue, BatchApprove, BatchReject
│   └── filters.py              # BatchFilter, filter matching
│
├── audit/                      # NEW: Phase 4
│   ├── __init__.py
│   ├── trail.py                # AuditTrail class
│   └── retention.py            # AuditRetention, cleanup
│
├── db/
│   ├── migrations.py           # UPDATE: new tables/columns
│   └── pipelines.py            # UPDATE: priority support
│
├── cli.py                      # UPDATE: new commands
└── mcp_server.py               # UPDATE: new tools
```

---

## Implementation Estimate

| Component | Complexity | Notes |
|-----------|------------|-------|
| Health Monitor | Medium | Aggregation queries, alert logic |
| Stuck Detection | Medium | Threshold calculation, recovery logic |
| Batch Operations | Medium | Filter logic, safety features |
| Priority Queues | Low | Mostly SQL changes |
| Audit Trail | Low | Simple append-only table |
| CLI Commands | Low | Wiring to components |
| MCP Tools | Low | Wiring to components |
| Tests | Medium | Many edge cases to cover |

---

## Summary

Phase 4 takes the orchestrator from "works correctly" to "runs reliably." The key additions are:

| Feature | Why It Matters |
|---------|----------------|
| Health monitoring | Know at a glance if the system is healthy |
| Stuck detection | Catch problems before they pile up |
| Batch operations | Handle bulk imports efficiently |
| Priority queues | Process urgent items first |
| Audit trail | Answer "who did what when?" |

These are the features that separate a prototype from a production system. They're not glamorous, but they're what lets you trust the system to run unattended.

---

## Next Steps

1. **Review this design** — Does it address your operational needs?
2. **Create implementation plan** — Break into tasks with test-first approach
3. **Implement in priority order:**
   - Priority queues (enables batch queue)
   - Batch operations (most immediate value)
   - Health monitoring (visibility)
   - Audit trail (compliance)
   - Stuck detection (automation)

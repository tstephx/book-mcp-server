# Phase 4: Production Hardening - Complete

**Status:** ✅ Complete
**Date:** 2025-01-28
**Tests:** 45 passing

---

## Summary

Phase 4 adds production-ready features for operating the agentic pipeline at scale:
- **Health Monitoring** - Real-time visibility into pipeline status
- **Stuck Detection** - Automatic identification of stalled pipelines
- **Batch Operations** - Bulk approve/reject with filters
- **Priority Queues** - Process important books first
- **Audit Trail** - Complete history of all decisions

---

## Features Implemented

### 1. Health Monitoring

Get real-time pipeline health metrics:

```bash
agentic-pipeline health
```

Output:
```
Pipeline Health
-----------------------------------
  Active:     2 (processing now)
  Stuck:      0
  Queued:     15 (waiting)
  Completed:  47 (last 24h)
  Failed:     3 (needs_retry)
-----------------------------------

Alerts:
  [info] Queue has 150 books waiting (threshold: 100)
```

**JSON output for automation:**
```bash
agentic-pipeline health --json
```

### 2. Stuck Detection

Find pipelines that appear stuck:

```bash
agentic-pipeline stuck
```

Output:
```
Found 2 stuck pipeline(s):

  a1b2c3d4... [processing]
    Stuck for 45 min (expected: 15 min)
    Source: advanced-python.epub

  e5f6g7h8... [embedding]
    Stuck for 22 min (expected: 10 min)
    Source: linux-kernel.pdf
```

**Default timeouts by state:**
| State | Expected Duration |
|-------|------------------|
| HASHING | 1 min |
| CLASSIFYING | 2 min |
| PROCESSING | 15 min |
| EMBEDDING | 10 min |
| PENDING_APPROVAL | No timeout (human) |

### 3. Batch Operations

#### Batch Approve
```bash
# Preview what would be approved
agentic-pipeline batch-approve --min-confidence 0.9 --book-type technical_tutorial

# Actually approve
agentic-pipeline batch-approve --min-confidence 0.9 --execute
```

#### Batch Reject
```bash
# Reject all newspapers
agentic-pipeline batch-reject --book-type newspaper --reason "Not ingesting periodicals" --execute
```

### 4. Priority Queues

Books are processed in priority order (1 = highest, 10 = lowest):

```python
from agentic_pipeline.db.pipelines import PipelineRepository

repo = PipelineRepository(db_path)
repo.create("/important-book.epub", hash, priority=1)  # Process first
repo.create("/normal-book.epub", hash, priority=5)     # Default priority
repo.update_priority(pipeline_id, 2)                   # Bump priority
```

View queue by priority:
```python
repo.get_queue_by_priority()
# {1: 2, 5: 15, 10: 3}  # 2 at priority 1, 15 at priority 5, etc.
```

### 5. Audit Trail

Every decision is logged:

```bash
agentic-pipeline audit --last 20
```

Output:
```
Audit Trail (20 entries)

Time                Action          Actor           Book
2025-01-28 14:32   APPROVED        human:taylor    a1b2c3d4...
2025-01-28 14:30   BATCH_APPROVED  human:cli       batch:5_books
2025-01-28 14:15   REJECTED        auto:low_conf   e5f6g7h8...
```

**Filter by actor or action:**
```bash
agentic-pipeline audit --actor "human:taylor" --action APPROVED
```

---

## MCP Tools

All features available via Claude:

| Tool | Description |
|------|-------------|
| `get_pipeline_health()` | Get health metrics with alerts |
| `get_stuck_pipelines()` | List stuck pipelines |
| `batch_approve_tool()` | Approve matching books |
| `batch_reject_tool()` | Reject matching books |
| `get_audit_log()` | Query audit trail |

**Example Claude conversation:**
```
You: "Check pipeline health"
Claude: "2 active, 15 queued, 3 failed. No stuck pipelines."

You: "Approve all high-confidence technical tutorials"
Claude: "Found 8 matching books. Execute?"

You: "Yes"
Claude: "Approved 8 books. Logged to audit trail."
```

---

## Database Schema

### New Tables

**health_metrics** - Cached health metrics
```sql
CREATE TABLE health_metrics (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    active_count INTEGER,
    queued_count INTEGER,
    stuck_count INTEGER,
    completed_24h INTEGER,
    failed_count INTEGER,
    queue_by_priority JSON,
    alerts JSON,
    updated_at TIMESTAMP
);
```

**state_duration_stats** - For stuck detection
```sql
CREATE TABLE state_duration_stats (
    state TEXT PRIMARY KEY,
    sample_count INTEGER,
    median_seconds REAL,
    p95_seconds REAL,
    max_seconds REAL,
    updated_at TIMESTAMP
);
```

### Schema Updates

**processing_pipelines** - Added columns:
- `priority INTEGER DEFAULT 5`

**approval_audit** - Added columns:
- `filter_used JSON` (for batch operations)

---

## Code Structure

```
agentic_pipeline/
├── audit/
│   ├── __init__.py
│   └── trail.py           # AuditTrail class
├── batch/
│   ├── __init__.py
│   ├── filters.py         # BatchFilter dataclass
│   └── operations.py      # BatchOperations class
├── health/
│   ├── __init__.py
│   ├── monitor.py         # HealthMonitor class
│   └── stuck_detector.py  # StuckDetector class
├── db/
│   ├── migrations.py      # Updated with Phase 4 tables
│   └── pipelines.py       # Added priority methods
├── cli.py                 # Added Phase 4 commands
└── mcp_server.py          # Added Phase 4 tools
```

---

## Tests

All 45 tests passing:

```
tests/test_phase4_migrations.py      4 passed
tests/test_priority_queue.py         5 passed
tests/test_audit_trail.py            6 passed
tests/test_batch_filters.py          5 passed
tests/test_batch_operations.py       4 passed
tests/test_health_monitor.py         5 passed
tests/test_stuck_detection.py        4 passed
tests/test_cli_phase4.py             5 passed
tests/test_mcp_phase4.py             4 passed
tests/test_phase4_integration.py     3 passed
```

Run all Phase 4 tests:
```bash
pytest tests/test_phase4*.py tests/test_priority_queue.py tests/test_audit_trail.py tests/test_batch*.py tests/test_health_monitor.py tests/test_stuck_detection.py tests/test_cli_phase4.py tests/test_mcp_phase4.py -v
```

---

## Configuration

### Alert Thresholds

```python
from agentic_pipeline.health import HealthMonitor

monitor = HealthMonitor(
    db_path,
    alert_queue_threshold=100,    # Alert if queue > 100
    alert_failure_rate=0.20,      # Alert if >20% failure rate
)
```

### Stuck Detection Thresholds

```python
from agentic_pipeline.health import StuckDetector

detector = StuckDetector(
    db_path,
    stuck_multiplier=2.0,  # Flag if 2x expected duration
    custom_thresholds={
        "PROCESSING": 1800,  # Override to 30 min
    },
)
```

---

## Next Steps

After Phase 4:
- [ ] Test with production book volume
- [ ] Tune stuck detection thresholds based on real data
- [ ] Consider Phase 5: Confident Autonomy
  - Auto-approve high-confidence books
  - Calibrated thresholds based on historical accuracy
  - Spot-check sampling

---

## Related Documentation

- [Phase 4 Design](./plans/2025-01-28-phase4-production-hardening-design.md)
- [Phase 4 Implementation Plan](./plans/2025-01-28-phase4-production-hardening-implementation.md)
- [Phase 3 Orchestrator](./plans/2025-01-28-phase3-orchestrator-design.md)

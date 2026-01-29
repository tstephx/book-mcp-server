# Phase 5: Confident Autonomy - Complete

**Status:** ✅ Complete
**Date:** 2025-01-28
**Tests:** 30 passing

---

## Summary

Phase 5 implements a graduated autonomy system that allows the pipeline to auto-approve high-confidence books while maintaining human oversight for edge cases. The system starts cautious and earns trust over time through measured accuracy.

**Key Features:**
- **Autonomy Modes** - Supervised → Partial → Confident progression
- **Calibration Engine** - Measures actual accuracy vs predicted confidence
- **Escape Hatch** - One-command revert to fully supervised mode
- **Spot-Check System** - Ongoing verification of auto-approved books

---

## Autonomy Modes

### Supervised Mode (Default)
- All books require human approval
- System collects accuracy metrics
- Building baseline for calibration

### Partial Autonomy Mode
- Auto-approve: confidence ≥ threshold AND known book type AND zero issues
- Human review for everything else
- Weekly spot-checks of 10% of auto-approved books
- **Requirements:** 100+ books processed, <15% override rate

### Confident Autonomy Mode
- Auto-approve based on per-book-type calibrated thresholds
- Human review only for edge cases (~10% of books)
- Monthly spot-checks of 5% of auto-approved books
- **Requirements:** 500+ books processed, <5% override rate, calibration ±5%

---

## Features Implemented

### 1. Autonomy Configuration

Manage autonomy mode and settings:

```bash
# Check current status
agentic-pipeline autonomy status

# Enable partial autonomy
agentic-pipeline autonomy enable partial

# Enable confident autonomy
agentic-pipeline autonomy enable confident

# Disable autonomy (revert to supervised)
agentic-pipeline autonomy disable
```

Output:
```
Autonomy Status
-----------------------------------
  Mode: partial

Last 30 Days:
  Total processed: 127
  Auto-approved:   89
  Human approved:  35
  Human rejected:  3
```

### 2. Escape Hatch

Immediately revert to supervised mode when something seems off:

```bash
agentic-pipeline escape-hatch "Noticing unusual classification patterns"
```

Output:
```
⚠️  ESCAPE HATCH ACTIVATED

All autonomy disabled. Reverting to supervised mode.
Reason: Noticing unusual classification patterns

To resume: agentic-pipeline autonomy resume
```

Resume after investigation:
```bash
agentic-pipeline autonomy resume
```

### 3. Calibration Engine

The system continuously measures whether predicted confidence matches actual accuracy:

```python
from agentic_pipeline.autonomy import CalibrationEngine

engine = CalibrationEngine(db_path, min_samples=50, target_accuracy=0.95)

# Get calibration for a book type
calibration = engine.calculate_calibration("technical_tutorial")
# {'book_type': 'technical_tutorial', 'sample_count': 127, 'accuracy': 0.94, 'avg_confidence': 0.91}

# Calculate safe threshold for auto-approval
threshold = engine.calculate_threshold("technical_tutorial")
# 0.92 (can auto-approve at 92% confidence with 95% accuracy)

# Update all thresholds
engine.update_thresholds()
```

### 4. Spot-Check System

Verify auto-approved books are actually correct:

```bash
# List books pending spot-check
agentic-pipeline spot-check --list
```

Output:
```
Pending Spot-Checks (8 books)

  a1b2c3d4... [technical_tutorial] 94%
  e5f6g7h8... [technical_tutorial] 92%
  ...
```

```python
from agentic_pipeline.autonomy import SpotCheckManager

manager = SpotCheckManager(db_path, sample_rate=0.10)

# Select books for review (10% sample)
pending = manager.select_for_review()

# Submit review result
manager.submit_result(
    book_id="a1b2c3d4",
    classification_correct=True,
    quality_acceptable=True,
    reviewer="human:taylor"
)

# Check overall accuracy
accuracy = manager.get_accuracy_rate()  # 0.97
```

### 5. Metrics Collection

Track all decisions for analysis:

```python
from agentic_pipeline.autonomy import MetricsCollector

collector = MetricsCollector(db_path)

# Record a decision
collector.record_decision(
    book_id="book123",
    book_type="technical_tutorial",
    confidence=0.92,
    decision="approved",
    actor="auto:high_confidence"
)

# Get aggregated metrics
metrics = collector.get_metrics(days=30)
# {'total_processed': 127, 'auto_approved': 89, 'human_approved': 35, ...}

# Get accuracy for specific book type
accuracy = collector.get_accuracy_by_type("technical_tutorial")
# {'book_type': 'technical_tutorial', 'sample_count': 85, 'accuracy': 0.94}
```

---

## MCP Tools

All features available via Claude:

| Tool | Description |
|------|-------------|
| `get_autonomy_status()` | Get current mode and metrics |
| `set_autonomy_mode(mode)` | Change autonomy mode |
| `activate_escape_hatch_tool(reason)` | Emergency revert to supervised |
| `get_autonomy_readiness()` | Check if ready for next mode |

**Example Claude conversation:**
```
You: "Check autonomy status"
Claude: "Currently in partial mode. 127 books processed in last 30 days.
         89 auto-approved, 35 human approved, 3 rejected."

You: "Are we ready for confident mode?"
Claude: "Not yet. Need 500+ books (currently 127) and <5% override rate
         (currently 2.4%). Recommend staying in partial mode."

You: "Something seems off with the classifications today"
Claude: [activates escape hatch]
        "Escape hatch activated. All books now require human review.
         Reason logged. Run 'autonomy resume' when ready."
```

---

## Database Schema

### New Tables

**autonomy_thresholds** - Per-book-type calibrated thresholds
```sql
CREATE TABLE autonomy_thresholds (
    book_type TEXT PRIMARY KEY,
    auto_approve_threshold REAL,
    sample_count INTEGER NOT NULL DEFAULT 0,
    measured_accuracy REAL,
    last_calculated TIMESTAMP,
    calibration_data JSON,
    manual_override REAL,
    override_reason TEXT
);
```

**spot_checks** - Spot-check tracking
```sql
CREATE TABLE spot_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id TEXT NOT NULL,
    pipeline_id TEXT,
    original_classification TEXT,
    original_confidence REAL,
    auto_approved_at TIMESTAMP,
    classification_correct BOOLEAN,
    quality_acceptable BOOLEAN,
    reviewer TEXT,
    notes TEXT,
    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Existing Tables (Phase 4)

**autonomy_config** - Singleton config row
- `current_mode` - supervised/partial/confident
- `escape_hatch_active` - Boolean flag
- `escape_hatch_reason` - Why activated
- `spot_check_percentage` - Sample rate

**autonomy_feedback** - Decision log for calibration
- Tracks original decision, confidence, human override
- Used to calculate accuracy metrics

---

## Code Structure

```
agentic_pipeline/
├── autonomy/
│   ├── __init__.py          # Package exports
│   ├── config.py            # AutonomyConfig class
│   ├── metrics.py           # MetricsCollector class
│   ├── calibration.py       # CalibrationEngine class
│   └── spot_check.py        # SpotCheckManager class
├── cli.py                   # Added Phase 5 commands
└── mcp_server.py            # Added Phase 5 tools
```

---

## Tests

All 30 tests passing:

```
tests/test_phase5_migrations.py       5 passed
tests/test_autonomy_config.py         6 passed
tests/test_autonomy_metrics.py        3 passed
tests/test_calibration.py             3 passed
tests/test_spot_check.py              3 passed
tests/test_cli_phase5.py              4 passed
tests/test_mcp_phase5.py              4 passed
tests/test_phase5_integration.py      2 passed
```

Run all Phase 5 tests:
```bash
pytest tests/test_phase5*.py tests/test_autonomy*.py tests/test_calibration.py tests/test_spot_check.py tests/test_cli_phase5.py tests/test_mcp_phase5.py -v
```

---

## Configuration

### Calibration Settings

```python
from agentic_pipeline.autonomy import CalibrationEngine

engine = CalibrationEngine(
    db_path,
    min_samples=50,       # Minimum samples before calculating threshold
    target_accuracy=0.95, # Target accuracy for auto-approval
)
```

### Spot-Check Settings

```python
from agentic_pipeline.autonomy import SpotCheckManager

manager = SpotCheckManager(
    db_path,
    sample_rate=0.10,  # 10% of auto-approved books
)
```

---

## Typical Workflow

### 1. Start in Supervised Mode
```bash
agentic-pipeline autonomy status
# Mode: supervised
```

All books go through human review. System collects metrics.

### 2. Check Readiness for Partial Mode
```python
readiness = get_autonomy_readiness()
# {'total_processed': 150, 'override_rate': 0.08, 'ready_for_partial': True}
```

### 3. Enable Partial Autonomy
```bash
agentic-pipeline autonomy enable partial
```

High-confidence books auto-approve. Calibration engine starts measuring.

### 4. Monitor and Spot-Check
```bash
agentic-pipeline spot-check --list
# Review a sample of auto-approved books
```

### 5. If Issues Arise
```bash
agentic-pipeline escape-hatch "Seeing misclassified tutorials"
# Immediately reverts to supervised mode
```

### 6. Resume When Fixed
```bash
agentic-pipeline autonomy resume
```

### 7. Graduate to Confident Mode
After 500+ books with <5% override rate:
```bash
agentic-pipeline autonomy enable confident
```

---

## Safety Guarantees

1. **Conservative Start** - Always begins in supervised mode
2. **Earned Trust** - Must demonstrate accuracy before advancing
3. **Escape Hatch** - One command reverts everything
4. **Spot-Checks** - Continuous verification of auto-approvals
5. **Per-Type Thresholds** - Different book types can have different thresholds
6. **Override Tracking** - Every human correction is logged and analyzed

---

## Related Documentation

- [Phase 5 Design](./plans/2025-01-28-phase5-confident-autonomy-design.md)
- [Phase 5 Implementation Plan](./plans/2025-01-28-phase5-confident-autonomy-implementation.md)
- [Phase 4 Production Hardening](./PHASE4-PRODUCTION-HARDENING-COMPLETE.md)
- [README](../README.md)

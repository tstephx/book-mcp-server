# Agentic Processing Pipeline Design

**Date:** 2025-01-28
**Status:** Approved
**Author:** Taylor Stephens + Claude

## Overview

An agentic AI layer for the book ingestion pipeline that enables adaptive processing, automated quality validation, and a path from supervised to confident autonomy.

### Goals

1. **Adaptive Processing** — Recognize book types (textbook, tutorial, magazine, etc.) and apply optimal processing strategies
2. **Error Recovery** — Automatically retry with adjusted configs when processing fails
3. **Quality Feedback Loop** — Capture human corrections to improve future processing
4. **Autonomous Pipeline** — Progress from supervised approval to confident autonomy as trust builds

### Approach

**Hybrid architecture:** LLM agents for classification and quality decisions, traditional heuristics for heavy processing. MCP-native tools for Claude integration.

**Autonomy model:** Start with supervised autonomy (human approves all), graduate to confident autonomy (auto-approve high-confidence) based on calibration data.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        WATCH FOLDER                                 │
│              /path/to/ebooks/               │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    AGENTIC PROCESSING LAYER                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────┐ │
│  │  Classifier  │→ │  Strategy    │→ │  Processor   │→ │ Quality │ │
│  │    Agent     │  │   Selector   │  │   (existing) │  │ Validator│ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────┘ │
│         ↑                                                    │      │
│         └────────────── Error Recovery Agent ←───────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      APPROVAL GATE (MCP Tool)                       │
│         Claude reviews summary → Approve / Reject / Adjust          │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    EXISTING STORAGE (library.db)                    │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Principles

- State machine controls flow; agents invoked at four checkpoints only
- Existing ingestion code remains untouched — agents configure and invoke it
- All decisions logged for audit trail and autonomy calibration

---

## Components

### 1. Watch Folder & Deduplication

**Trigger modes:**
- File watcher daemon (real-time detection)
- Manual CLI batch processing

**Deduplication:**
- Content hash (first 50KB) prevents reprocessing renamed files
- Subfolder name becomes tag (e.g., `/Agentic-AI/` → tag: "agentic-ai")

**Folder conventions:**
```
/Documents/_ebooks/
├── _inbox/              → Auto-process
├── _rejected/           → Failed books archived
├── _rollback/           → Rolled-back books (restorable)
├── Agentic-AI/          → tag: "agentic-ai"
└── Uncategorized/       → No tag
```

### 2. Classifier Agent

Analyzes book sample (first 5-10 pages) and outputs a book profile.

**Output:**
```json
{
  "book_type": "technical_tutorial",
  "confidence": 0.92,
  "detected_features": {
    "has_code_blocks": true,
    "has_exercises": true,
    "has_numbered_chapters": true,
    "estimated_complexity": "intermediate"
  },
  "suggested_tags": ["ai", "agents", "python"],
  "warnings": []
}
```

**Book type taxonomy:**

| Type | Characteristics | Processing Implications |
|------|-----------------|------------------------|
| `technical_tutorial` | Code samples, step-by-step, exercises | Preserve code blocks, split by chapter |
| `technical_reference` | API docs, tables, dense information | Smaller chunks, preserve structure |
| `textbook` | Sections, summaries, review questions | Detect section hierarchy |
| `narrative_nonfiction` | Flowing prose, few headers | Larger chapters, semantic splitting |
| `research_collection` | Papers, citations, abstracts | Split by paper/article boundaries |
| `newspaper` | Articles, bylines, datelines, short pieces | Split by article, preserve metadata |
| `magazine` | Feature articles, sidebars, mixed content | Split by article, detect feature vs. short |
| `unknown` | Can't confidently classify | Flag for human review |

**Cost:** ~$0.01-0.02 per book (single LLM call)

### 3. Strategy Selector

Deterministic rules engine — no LLM cost. Maps classification to processing config.

**Output:**
```json
{
  "strategy_name": "technical_tutorial_v1",
  "chapter_detection": {
    "method": "toc_with_explicit_fallback",
    "min_words_per_chapter": 1000,
    "max_words_per_chapter": 25000,
    "preserve_code_blocks": true
  },
  "text_cleaning": {
    "remove_headers_footers": true,
    "preserve_formatting": ["code", "lists", "tables"]
  },
  "quality_thresholds": {
    "min_chapters": 3,
    "max_chapters": 50,
    "min_avg_chapter_words": 500
  }
}
```

Strategies stored as versioned JSON files — tunable without code changes.

### 4. Quality Validator Agent

Runs after processing completes. Catches bad results before database commit.

**Output:**
```json
{
  "verdict": "needs_review",
  "confidence": 0.75,
  "issues": [
    {
      "type": "chapter_count_anomaly",
      "severity": "warning",
      "message": "Expected 8-15 chapters, got 47",
      "suggestion": "Likely detecting section headers as chapters"
    }
  ],
  "quality_scores": {
    "chapter_structure": 0.6,
    "content_completeness": 0.95
  },
  "recommended_action": "retry_with_adjustments",
  "adjustments": {
    "chapter_detection.method": "explicit_only"
  }
}
```

**Verdicts:**
- `approved` (>0.9): Ready for commit or supervised review
- `needs_review` (0.7-0.9): Flag for human approval
- `retry_with_adjustments` (<0.7): Send to Error Recovery
- `rejected`: Fundamental issue (corrupt, wrong language)

**Cost:** ~$0.02-0.03 per book

### 5. Error Recovery Agent

Handles failures and quality issues. Max 2 automated retries.

**Recovery strategies:**

| Error Type | Recovery Approach |
|------------|-------------------|
| Chapter over-detection | Tighten detection, raise min_words |
| Chapter under-detection | Try explicit patterns, semantic splitting |
| Code block fragmentation | Strict code preservation |
| Conversion failure | Alternate converter, raw text only |
| Timeout | Increase limits, smaller batches |
| Unknown/repeated | Escalate to human with diagnostics |

**Learning hook:** Every recovery logged with outcome → training data for strategy improvement.

### 6. Approval Gate (MCP Tools)

Complete tool set for supervised autonomy.

**Review tools:**
- `review_pending_books` — Get approval queue with stats
- `preview_chapter` — Spot-check extracted content
- `preview_comparison` — Side-by-side original vs. extracted

**Approval actions:**
- `approve_book` — Commit single book
- `batch_approve` — Bulk approval with confidence threshold
- `reject_book` — Reject with optional retry
- `quick_adjust` — Common fixes (merge chapters, reclassify, etc.)

**Recovery & monitoring:**
- `rollback_book` — Remove approved book, archive for restore
- `get_processing_stats` — System health dashboard

### 7. Pipeline Orchestration

State machine with agent checkpoints.

**States:**
```
DETECTED → HASHING → CLASSIFYING → SELECTING_STRATEGY → PROCESSING
    → VALIDATING → PENDING_APPROVAL → APPROVED → EMBEDDING → COMPLETE
                         ↓
                   NEEDS_RETRY → (back to PROCESSING)
                         ↓
                   REJECTED → ARCHIVED
```

**Pipeline record tracks:** state, book profile, strategy config, validation result, retry count, timing, approval details.

**Health features:**
- Timeout detection per state
- Heartbeat for long operations
- Priority queue support
- Stuck pipeline alerts

---

## Database Schema

### New Tables

```sql
-- Pipeline tracking
CREATE TABLE processing_pipelines (
    id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL,
    book_profile JSON,
    strategy_config JSON,
    validation_result JSON,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 2,
    error_log JSON,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    completed_at TIMESTAMP,
    timeout_at TIMESTAMP,
    last_heartbeat TIMESTAMP,
    priority INTEGER DEFAULT 5,
    approved_by TEXT,
    approval_confidence REAL
);

-- State history (debugging/analytics)
CREATE TABLE pipeline_state_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pipeline_id TEXT NOT NULL,
    from_state TEXT,
    to_state TEXT NOT NULL,
    duration_ms INTEGER,
    agent_output JSON,
    error_details JSON,
    created_at TIMESTAMP
);

-- Versioned strategies
CREATE TABLE processing_strategies (
    name TEXT PRIMARY KEY,
    book_type TEXT NOT NULL,
    config JSON NOT NULL,
    version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE
);

-- System configuration
CREATE TABLE pipeline_config (
    key TEXT PRIMARY KEY,
    value JSON NOT NULL
);

-- Audit trail (immutable)
CREATE TABLE approval_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id TEXT NOT NULL,
    pipeline_id TEXT,
    action TEXT NOT NULL,
    actor TEXT NOT NULL,
    reason TEXT,
    before_state JSON,
    after_state JSON,
    adjustments JSON,
    confidence_at_decision REAL,
    autonomy_mode TEXT,
    session_id TEXT,
    performed_at TIMESTAMP
);

-- Retention policies
CREATE TABLE audit_retention (
    audit_type TEXT PRIMARY KEY,
    retain_days INTEGER NOT NULL,
    last_cleanup TIMESTAMP
);

-- Autonomy calibration metrics
CREATE TABLE autonomy_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    total_processed INTEGER,
    auto_approved INTEGER,
    human_approved INTEGER,
    human_rejected INTEGER,
    human_adjusted INTEGER,
    avg_confidence_auto_approved REAL,
    avg_confidence_human_approved REAL,
    avg_confidence_human_rejected REAL,
    auto_approved_later_rolled_back INTEGER,
    human_approved_later_rolled_back INTEGER,
    metrics_by_type JSON,
    confidence_buckets JSON,
    UNIQUE(period_start, period_end)
);

-- Human correction tracking
CREATE TABLE autonomy_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id TEXT NOT NULL,
    pipeline_id TEXT,
    original_decision TEXT NOT NULL,
    original_confidence REAL,
    original_book_type TEXT,
    human_decision TEXT NOT NULL,
    human_adjustments JSON,
    feedback_category TEXT,
    feedback_notes TEXT,
    created_at TIMESTAMP
);

-- Autonomy configuration (singleton)
CREATE TABLE autonomy_config (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    current_mode TEXT DEFAULT 'supervised',
    auto_approve_threshold REAL DEFAULT 0.95,
    auto_retry_threshold REAL DEFAULT 0.70,
    require_known_book_type BOOLEAN DEFAULT TRUE,
    require_zero_issues BOOLEAN DEFAULT TRUE,
    max_auto_approvals_per_day INTEGER DEFAULT 50,
    spot_check_percentage REAL DEFAULT 0.10,
    escape_hatch_active BOOLEAN DEFAULT FALSE,
    escape_hatch_activated_at TIMESTAMP,
    escape_hatch_reason TEXT
);
```

### Extensions to Existing Tables

```sql
-- books table
ALTER TABLE books ADD COLUMN content_hash TEXT;
ALTER TABLE books ADD COLUMN book_type TEXT;
ALTER TABLE books ADD COLUMN classification_confidence REAL;
ALTER TABLE books ADD COLUMN source_tags JSON;
ALTER TABLE books ADD COLUMN pipeline_id TEXT;
ALTER TABLE books ADD COLUMN approved_at TIMESTAMP;
ALTER TABLE books ADD COLUMN approved_by TEXT;
ALTER TABLE books ADD COLUMN autonomy_mode_at_approval TEXT;

-- chapters table
ALTER TABLE chapters ADD COLUMN quality_score REAL;
ALTER TABLE chapters ADD COLUMN parent_chapter_id TEXT;
ALTER TABLE chapters ADD COLUMN embedding_model TEXT;
ALTER TABLE chapters ADD COLUMN embedding_generated_at TIMESTAMP;
```

---

## Autonomy Transition

### Phases

**Phase 1: SUPERVISED (initial)**
- All books require human approval
- Collect calibration data
- Build trust baseline

**Phase 2: PARTIAL AUTONOMY (after ~100 books, <10% override rate)**
- Auto-approve: confidence ≥0.92 AND zero issues AND known book_type
- Human review: everything else
- Weekly spot-check 10% of auto-approved

**Phase 3: CONFIDENT AUTONOMY (after ~500 books, calibrated thresholds)**
- Auto-approve: confidence ≥ calculated_safe_threshold
- Auto-retry: confidence 0.7-0.9 with adjustments
- Human review: confidence <0.7 OR unknown book_type OR anomaly
- Monthly spot-check 5% of auto-approved

**Escape hatch:** One command reverts to Phase 1 instantly.

### Success Criteria for Confident Autonomy

- 100+ books processed in supervised mode
- Human override rate <10%
- Confidence calibration within ±5%
- Zero auto-approved books rolled back in 30 days

### MCP Tools for Autonomy

```
get_autonomy_readiness()
→ calibration data, override rates, suggested threshold, recommendation

set_autonomy_mode(mode, threshold?)
→ changes current mode

activate_escape_hatch(reason)
→ immediately reverts to supervised
```

---

## Project Structure

```
/path/to/projects/
├── book-ingestion-python/        # Existing (unchanged)
├── book-mcp-server/              # Extended with pipeline_tools.py
└── agentic-pipeline/             # NEW
    ├── config/
    │   ├── strategies/           # Versioned JSON configs
    │   └── prompts/              # Agent prompt templates
    ├── src/agentic_pipeline/
    │   ├── agents/               # classifier, validator, recovery
    │   ├── pipeline/             # states, orchestrator, strategy
    │   ├── triggers/             # watcher, batch
    │   ├── approval/             # queue, actions, comparison
    │   ├── autonomy/             # metrics, feedback, thresholds
    │   ├── db/                   # migrations, pipelines, audit
    │   └── health/               # monitor, alerts
    └── tests/
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
- Package scaffolding
- Database migrations
- Strategy configs (3-4 book types)
- Basic orchestrator (manual classification)
- MCP tools: `review_pending_books`, `approve_book`, `reject_book`

**Milestone:** Manual classify → run ingestion → approve via MCP

### Phase 2: Classifier Agent (Week 3)
- Classifier agent + prompts
- Strategy selector rules engine
- Basic file watcher

**Milestone:** Auto-classify → correct strategy → awaits approval

### Phase 3: Quality & Recovery (Week 4-5)
- Quality Validator agent
- Error Recovery agent
- `preview_comparison`, `quick_adjust`, `rollback_book` tools

**Milestone:** Full supervised autonomy loop

### Phase 4: Production Hardening (Week 6)
- Health monitoring, stuck detection
- Batch operations, priority queues
- Complete audit trail
- Tests

**Milestone:** Reliable daily use

### Phase 5: Confident Autonomy (Week 7+)
- Autonomy metrics collection
- Feedback capture
- Graduated thresholds
- Escape hatch
- Spot-check sampling

**Milestone:** Auto-approve high-confidence, flag edge cases only

---

## Cost Estimates

| Component | Cost per Book | Notes |
|-----------|---------------|-------|
| Classifier Agent | $0.01-0.02 | Single LLM call |
| Quality Validator | $0.02-0.03 | Single LLM call |
| Error Recovery | $0.01-0.02 | Only on failures |
| **Total (success path)** | **~$0.03-0.05** | Classifier + Validator |
| **Total (with retry)** | **~$0.05-0.10** | Adds Recovery agent |

At 100 books/month: ~$3-10/month in LLM costs.

---

## Open Questions

1. **LLM provider:** Claude API vs. OpenAI vs. local models for agents?
2. **Alerting:** Email, Slack, or MCP-only notifications for stuck pipelines?
3. **Multi-user:** Single user for now, or design for shared library access?

---

## Appendix: MCP Tool Reference

### Review & Approval
- `review_pending_books(sort_by?)` — Get approval queue
- `preview_chapter(book_id, chapter_number)` — Read extracted chapter
- `preview_comparison(book_id, chapter_number)` — Side-by-side comparison
- `approve_book(book_id, adjustments?)` — Commit to library
- `batch_approve(book_ids?, min_confidence?, max_issues?)` — Bulk approve
- `reject_book(book_id, reason, retry?)` — Reject with optional retry
- `quick_adjust(book_id, preset)` — Apply common fix preset
- `rollback_book(library_id, reason)` — Remove from library

### Pipeline Management
- `trigger_ingestion(path, recursive?, auto_approve_threshold?)` — Manual batch
- `get_pipeline_health()` — Active, stuck, queue depth, alerts
- `get_processing_stats(days?)` — Volume, success rates, common issues

### Autonomy
- `get_autonomy_readiness()` — Calibration, override rate, recommendation
- `set_autonomy_mode(mode, threshold?)` — Change autonomy level
- `activate_escape_hatch(reason)` — Emergency revert to supervised

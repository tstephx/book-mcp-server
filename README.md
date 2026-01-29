# Agentic Book Processing Pipeline

**An AI-powered system that automatically processes, classifies, and ingests books into a searchable knowledge library.**

---

## What This Is

When you drop a PDF or EPUB into a folder, this system:

1. **Detects** the new file and checks if it's a duplicate
2. **Classifies** the book type (tutorial, textbook, magazine, etc.) using AI
3. **Selects** the optimal processing strategy for that book type
4. **Processes** the book through our existing ingestion pipeline
5. **Validates** the quality of the extracted content
6. **Queues** the book for your approval (or auto-approves if confidence is high)
7. **Commits** approved books to the searchable library

The goal: **Turn a manual, error-prone process into an intelligent, self-improving system.**

---

## Why We Built This

### The Problem

Our book ingestion pipeline works well for most books, but:

- **One size doesn't fit all**: A technical tutorial with code samples needs different processing than a magazine with short articles
- **Quality varies**: Some books get chopped incorrectly, losing semantic meaning
- **Manual intervention**: When processing fails, someone has to diagnose and retry manually
- **No learning**: The system makes the same mistakes repeatedly

### The Solution

An "agentic" layer that adds intelligence to the pipeline:

| Before | After |
|--------|-------|
| Same processing for all books | Adaptive processing based on book type |
| Silent failures | Automated error recovery with retries |
| Manual quality checks | AI-powered validation |
| Fixed behavior | System learns from corrections |

---

## How It Works

### The Pipeline Flow

```
                                    ┌─────────────────┐
                                    │   Watch Folder  │
                                    │   (your ebooks) │
                                    └────────┬────────┘
                                             │
                                             ▼
┌────────────────────────────────────────────────────────────────────┐
│                     AGENTIC PROCESSING LAYER                       │
│                                                                    │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    │
│  │ Classify │ → │ Select   │ → │ Process  │ → │ Validate │    │
│  │ (AI)     │    │ Strategy │    │ (existing│    │ (AI)     │    │
│  └──────────┘    └──────────┘    │ pipeline)│    └──────────┘    │
│       │                          └──────────┘          │          │
│       │                                                │          │
│       └──────────── Error Recovery (AI) ◄──────────────┘          │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │  Approval Gate  │
                                    │  (you review)   │
                                    └────────┬────────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │    Library      │
                                    │  (searchable)   │
                                    └─────────────────┘
```

### Key Components

#### 1. Classifier Agent (AI)
Reads a sample of the book and determines:
- **Book type**: Tutorial, textbook, magazine, narrative, etc.
- **Features**: Has code blocks? Has exercises? Multi-column layout?
- **Confidence**: How sure is the AI about this classification?

#### 2. Strategy Selector (Rules)
Maps the classification to a processing configuration:
- Technical tutorials → preserve code blocks, split by chapter
- Magazines → split by article, detect bylines
- Unknown types → use conservative defaults, flag for review

#### 3. Quality Validator (AI)
After processing, checks:
- Did we get a reasonable number of chapters?
- Are code samples intact or fragmented?
- Does the extracted content match what we expected?

#### 4. Error Recovery (AI)
When something goes wrong:
- Analyzes what failed and why
- Suggests configuration adjustments
- Retries automatically (up to 2 times)
- Escalates to human review if retries fail

#### 5. Approval Gate (You)
Before books enter the library:
- Review pending books with quality scores
- Approve good results with one click
- Reject or adjust problematic ones
- Rollback mistakes even after approval

---

## Autonomy Levels (Phase 5)

The system starts cautious and earns trust over time. This is implemented in [Phase 5: Confident Autonomy](./docs/plans/2025-01-28-phase5-confident-autonomy-design.md).

### Supervised Mode (Current)
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

**Escape hatch**: One command reverts to fully supervised mode instantly.

```bash
agentic-pipeline escape-hatch "Something seems off"
```

---

## Using the System

### For Daily Use

**Check what's pending:**
```bash
agentic-pipeline pending
```

**Approve a book:**
```bash
agentic-pipeline approve <pipeline-id>
```

**Reject with reason:**
```bash
agentic-pipeline reject <pipeline-id> --reason "Poor quality"
```

### Production Monitoring (Phase 4)

**Check pipeline health:**
```bash
agentic-pipeline health
```

**Find stuck pipelines:**
```bash
agentic-pipeline stuck
```

**Batch approve high-confidence books:**
```bash
agentic-pipeline batch-approve --min-confidence 0.9 --execute
```

**Query audit trail:**
```bash
agentic-pipeline audit --last 20
```

### With Claude (MCP Tools)

```
You: "Check for new books to review"

Claude: "3 books pending:
        - AI Agents in Practice (94% confidence) - ready to approve
        - LLM Design Patterns (91% confidence) - ready to approve
        - Magazine XYZ (72% confidence) - needs review"

You: "Approve the first two, show me the magazine"

Claude: [approves books, shows comparison view]
```

---

## Project Structure

```
agentic-pipeline/
├── config/
│   └── strategies/           # Processing configs per book type
│       ├── technical_tutorial_v1.json
│       ├── periodical_v1.json
│       └── conservative_v1.json
│
├── agentic_pipeline/
│   ├── agents/               # AI-powered components
│   ├── pipeline/             # State machine & orchestration
│   ├── approval/             # Approval queue & actions
│   ├── autonomy/             # Trust calibration & thresholds
│   └── db/                   # Database operations
│
└── docs/
    ├── README.md             # This file
    └── DESIGN.md             # Technical decisions & trade-offs
```

---

## Key Metrics

The system tracks:

| Metric | Purpose |
|--------|---------|
| Confidence calibration | Is 90% predicted confidence actually 90% accurate? |
| Human override rate | How often do humans change the AI's decision? |
| Auto-approval success | Do auto-approved books get rolled back later? |
| Processing time | How long does each stage take? |
| Error patterns | What types of books fail most often? |

---

## Getting Started

### Prerequisites
- Python 3.12+
- Existing book-ingestion-python pipeline
- Claude Code (for MCP integration)

### Installation
```bash
cd /path/to/agentic-pipeline
pip install -e .
agentic-pipeline init
```

### Start the file watcher
```bash
agentic-pipeline watch /path/to/ebooks
```

---

## FAQ

**Q: What if the AI classifies a book wrong?**

A: You reject it with the correct classification, and the system learns. Over time, these corrections improve the classifier's accuracy.

**Q: Can I process books without AI?**

A: Yes. Set `--skip-classification` to use conservative defaults, or manually specify the book type.

**Q: What happens during an outage?**

A: The pipeline saves checkpoints at every stage. When service resumes, it picks up where it left off.

**Q: How much does the AI cost?**

A: ~$0.03-0.05 per book (two LLM calls: classification + validation). At 100 books/month, that's $3-5/month.

---

## Related Documentation

- [DESIGN.md](./DESIGN.md) - Technical architecture & decision rationale
- [Phase 1 Plan](./docs/plans/2025-01-28-phase1-foundation.md) - Foundation implementation
- [Phase 2 Classifier](./docs/plans/2025-01-28-phase2-classifier-agent-design.md) - AI classifier design
- [Phase 3 Orchestrator](./docs/plans/2025-01-28-phase3-orchestrator-design.md) - Pipeline orchestration
- [Phase 4 Production Hardening](./docs/PHASE4-PRODUCTION-HARDENING-COMPLETE.md) - Health, batch ops, audit
- [Phase 5 Confident Autonomy](./docs/plans/2025-01-28-phase5-confident-autonomy-design.md) - Graduated autonomy system
- [Design Document](./docs/plans/2025-01-28-agentic-processing-pipeline-design.md) - Full specification

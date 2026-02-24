# Agentic Pipeline: Design Decisions & Rationale

> **Status:** All phases complete (Phase 1–5). Embeddings use OpenAI `text-embedding-3-small` (migrated from sentence-transformers). This document reflects the original design rationale and remains accurate for architecture decisions.

---

This document explains the architectural choices behind the agentic book processing pipeline—what we chose, what we didn't choose, and why.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Core Architecture Decision](#core-architecture-decision)
3. [Why Agentic AI?](#why-agentic-ai)
4. [Hybrid Approach: AI + Heuristics](#hybrid-approach-ai--heuristics)
5. [State Machine Design](#state-machine-design)
6. [Autonomy Model](#autonomy-model)
7. [Approval Workflow](#approval-workflow)
8. [Error Handling Strategy](#error-handling-strategy)
9. [Data & Storage Decisions](#data--storage-decisions)
10. [Cost Considerations](#cost-considerations)
11. [Security & Trust](#security--trust)
12. [What We Explicitly Chose NOT To Build](#what-we-explicitly-chose-not-to-build)
13. [Future Considerations](#future-considerations)

---

## Executive Summary

| Decision | Choice | Key Reason |
|----------|--------|------------|
| Architecture | Hybrid (AI + rules) | Balance cost, capability, predictability |
| AI placement | Decision points only | Minimize cost, maximize impact |
| Autonomy | Graduated (supervised → confident) | Build trust with data |
| Integration | MCP-native | Tight Claude integration |
| Processing | Wrap existing pipeline | Don't reinvent, enhance |

---

## Core Architecture Decision

### What We Chose: State Machine with AI Checkpoints

A deterministic state machine orchestrates the pipeline flow. AI agents are invoked only at specific decision points:

```
State Machine (deterministic)
     │
     ├── CLASSIFYING ────── AI Checkpoint (Classifier Agent)
     │
     ├── SELECTING ──────── Rules Engine (no AI)
     │
     ├── PROCESSING ─────── Existing Pipeline (no AI)
     │
     ├── VALIDATING ─────── AI Checkpoint (Validator Agent)
     │
     └── ERROR ──────────── AI Checkpoint (Recovery Agent)
```

### Alternatives Considered

#### Option A: Fully Autonomous Agent
A single AI agent that handles everything—classification, processing decisions, error recovery—in a continuous loop.

**Why we didn't choose this:**
- **Unpredictable**: Hard to debug when things go wrong
- **Expensive**: Every decision requires an LLM call
- **Overkill**: Most of the pipeline is deterministic (file I/O, text processing)
- **Trust**: Difficult to audit or understand why it made specific choices

#### Option B: No AI (Pure Heuristics)
Enhance the existing pipeline with more sophisticated rules and pattern matching.

**Why we didn't choose this:**
- **Brittle**: Rules break on edge cases
- **Maintenance burden**: Every new book type needs new rules
- **No learning**: Can't improve from feedback
- **Classification accuracy**: Pattern matching can't reliably distinguish "textbook with code" from "magazine with code snippets"

#### Option C: ML Models (Not LLMs)
Train custom classification models on our book corpus.

**Why we didn't choose this:**
- **Cold start**: Need labeled training data we don't have
- **Maintenance**: Models need retraining as book types evolve
- **Limited reasoning**: Can classify but can't explain why or suggest fixes
- **Overkill for volume**: Our ~100 books/month doesn't justify ML infrastructure

### Why This Choice Works

The hybrid approach gives us:

| Benefit | How |
|---------|-----|
| Predictability | State machine ensures consistent flow |
| Intelligence | AI handles genuinely hard decisions |
| Cost control | Only 2-3 LLM calls per book |
| Debuggability | Clear audit trail at each checkpoint |
| Adaptability | AI can handle novel book types |

---

## Why Agentic AI?

### The Core Insight

Books are diverse. A system that treats all books the same will:
- Over-split some (detecting headers as chapters)
- Under-split others (missing implicit chapter breaks)
- Destroy code samples in technical books
- Fragment articles in magazines

**The solution isn't better rules—it's recognizing what kind of book you're processing and adapting accordingly.**

### What "Agentic" Means Here

We use "agentic" to mean AI that:
1. **Perceives**: Analyzes the book to understand its structure
2. **Decides**: Chooses appropriate processing strategy
3. **Acts**: Configures the pipeline accordingly
4. **Learns**: Improves from human feedback over time

This is different from:
- **Chatbots**: Respond to queries but don't take action
- **Copilots**: Suggest but require human execution
- **Automation**: Follow fixed rules without adaptation

### Why Not Just Ask the User?

We could prompt users to classify each book manually. We chose not to because:

1. **Friction**: Adds a step before every ingestion
2. **Expertise required**: Users may not know "is this a textbook or reference?"
3. **Inconsistency**: Different users classify differently
4. **Scalability**: Batch processing becomes manual

The AI classifier handles this automatically with high accuracy, falling back to human judgment only when uncertain.

---

## Hybrid Approach: AI + Heuristics

### Where AI Adds Value

| Task | Why AI? |
|------|---------|
| Book classification | Requires understanding content, not just patterns |
| Quality validation | Needs to judge "does this look right?" holistically |
| Error diagnosis | Must reason about what went wrong and why |

### Where Heuristics Work Better

| Task | Why Heuristics? |
|------|-----------------|
| Strategy selection | Simple mapping: book_type → config |
| Text cleaning | Deterministic transformations |
| Chapter splitting | Algorithmic with configurable parameters |
| File I/O | No intelligence needed |

### The Cost-Capability Trade-off

```
                    High ┌─────────────────────────────────────┐
                         │                                     │
                         │   ┌─────────────────┐               │
                         │   │  Full AI Agent  │               │
              Capability │   │  (expensive)    │               │
                         │   └─────────────────┘               │
                         │                                     │
                         │          ┌─────────────────┐        │
                         │          │  HYBRID ★       │        │
                         │          │  (our choice)   │        │
                         │          └─────────────────┘        │
                         │                                     │
                         │   ┌─────────────────┐               │
                         │   │  Pure Heuristics│               │
                         │   │  (limited)      │               │
                    Low  └───┴─────────────────┴───────────────┘
                         Low                              High
                                       Cost
```

Our hybrid sits in the sweet spot: ~$0.03-0.05/book with near-full-AI capability for the decisions that matter.

---

## State Machine Design

### Why a State Machine?

State machines provide:

1. **Visibility**: Always know exactly where a book is in the pipeline
2. **Resumability**: Can restart from any checkpoint after failures
3. **Auditability**: Complete history of state transitions
4. **Predictability**: Valid transitions are explicitly defined

### State Transitions

```
DETECTED → HASHING → CLASSIFYING → SELECTING → PROCESSING → VALIDATING
                ↓                                    ↓            ↓
           DUPLICATE                            NEEDS_RETRY ←────┘
                                                    ↓
                                               PROCESSING (retry)
                                                    ↓
PENDING_APPROVAL ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
       ↓
    APPROVED → EMBEDDING → COMPLETE
       ↓
    REJECTED → ARCHIVED
```

### Why Not a Simple Queue?

A queue (process in order, done) would be simpler. We chose a state machine because:

| Queue Limitation | State Machine Solution |
|------------------|------------------------|
| Can't retry specific stages | Resume from any state |
| No visibility into progress | Query current state anytime |
| All-or-nothing processing | Partial progress preserved |
| No branching logic | Conditional transitions |

---

## Autonomy Model

### The Trust Problem

Autonomous systems face a chicken-and-egg problem:
- Users don't trust the system until it proves itself
- The system can't prove itself without autonomy

### Our Solution: Graduated Autonomy

```
Phase 1: SUPERVISED
├── All books require human approval
├── System collects accuracy data
└── Duration: ~100 books

Phase 2: PARTIAL AUTONOMY
├── Auto-approve: confidence ≥92%, zero issues
├── Human review: everything else
├── Weekly spot-checks (10%)
└── Duration: ~400 books

Phase 3: CONFIDENT AUTONOMY
├── Auto-approve: confidence ≥ calibrated threshold
├── Human review: low confidence or anomalies
└── Monthly spot-checks (5%)
```

### Alternatives Considered

#### Option A: Full Autonomy from Start
Just trust the AI and let it approve everything.

**Why we didn't choose this:**
- No baseline for "good" performance
- Errors could flood the library before detection
- No way to calibrate confidence thresholds
- Users wouldn't trust the system

#### Option B: Permanent Supervised Mode
Always require human approval.

**Why we didn't choose this:**
- Doesn't scale—human becomes bottleneck
- Wastes human attention on obvious approvals
- No value from AI if humans review everything anyway

### Confidence Calibration

The key insight: **AI confidence scores need calibration.**

When the AI says "90% confident," is it actually right 90% of the time? We track this:

```
Confidence Bucket | Predictions | Actually Correct | Calibration
90-100%          | 45          | 42 (93%)         | ✓ Good
80-90%           | 23          | 18 (78%)         | ⚠ Slightly overconfident
<80%             | 12          | 7 (58%)          | ✓ Correctly uncertain
```

This data drives threshold decisions. If 90%+ confidence is actually 93% accurate, we can safely auto-approve that bucket.

### The Escape Hatch

One command reverts to fully supervised mode:

```bash
agentic-pipeline escape-hatch --reason "Unusual errors detected"
```

This exists because:
- AI systems can fail in unexpected ways
- Business needs may require human oversight temporarily
- Trust can be lost faster than it's built

---

## Approval Workflow

### Why Human-in-the-Loop?

Even with high AI accuracy, human approval serves multiple purposes:

| Purpose | Explanation |
|---------|-------------|
| Quality assurance | Catch AI mistakes before they enter the library |
| Trust building | Users see the AI making good decisions |
| Feedback collection | Corrections improve the system |
| Accountability | Human makes final call on what enters library |

### Approval Interface Design

We chose MCP-native tools over a web UI because:

| MCP Tools | Web UI |
|-----------|--------|
| Integrated with Claude workflow | Separate application to manage |
| Conversational approval | Form-based approval |
| Context-aware (Claude knows what you're working on) | Standalone context |
| No additional infrastructure | Server, auth, hosting needed |

### Batch Operations

For efficiency, we support:
- `batch_approve(min_confidence=0.9)` — Approve all high-confidence books
- Review-by-exception — Only look at flagged items

This prevents the approval queue from becoming a bottleneck.

---

## Error Handling Strategy

### Retry Philosophy

**Retry with intelligence, not stubbornly.**

```
Failure
   ↓
Error Recovery Agent analyzes what went wrong
   ↓
Suggests specific configuration changes
   ↓
Retry with new config (max 2 times)
   ↓
If still failing → Escalate to human
```

### Why Limited Retries?

| Retry Count | Rationale |
|-------------|-----------|
| 0 | Some errors are unrecoverable (corrupt file) |
| 1 | First retry catches transient issues |
| 2 | Second retry with different strategy catches config issues |
| 3+ | Diminishing returns; likely needs human judgment |

### Alternatives Considered

#### Option A: No Automatic Retry
Fail immediately, let humans fix.

**Why we didn't choose this:**
- Many failures are recoverable with different settings
- Wastes human time on fixable issues

#### Option B: Unlimited Retry
Keep trying until it works.

**Why we didn't choose this:**
- Could loop forever on unfixable errors
- Wastes compute on lost causes
- Delays detection of systematic issues

---

## Data & Storage Decisions

### Shared Database

We extend the existing `library.db` rather than creating a separate database.

**Why:**
- Single source of truth
- No synchronization issues
- Existing tools work with new data
- Simpler backup/restore

**Trade-off:**
- Migrations must be backward-compatible
- Schema changes affect both systems

### Audit Trail Design

Every approval action is logged immutably:

```sql
approval_audit:
  - who (actor)
  - what (action)
  - when (timestamp)
  - why (reason)
  - before_state (snapshot)
  - after_state (snapshot)
```

**Why immutable?**
- Compliance: Can prove what happened and when
- Debugging: Reconstruct any past state
- Learning: Analyze patterns in human corrections

### Retention Policies

| Data Type | Retention | Rationale |
|-----------|-----------|-----------|
| Approvals | 1 year | Compliance, auditing |
| Rejections | 90 days | Learn from mistakes, but not forever |
| Rollbacks | Forever | Critical decisions need permanent record |
| Metrics | 2 years | Long-term trend analysis |

---

## Cost Considerations

### LLM Cost Breakdown

| Component | Calls/Book | Estimated Cost |
|-----------|------------|----------------|
| Classifier | 1 | $0.01-0.02 |
| Validator | 1 | $0.02-0.03 |
| Error Recovery | 0-1 | $0.01-0.02 (if needed) |
| **Total** | 2-3 | **$0.03-0.05** |

### Cost Control Mechanisms

1. **AI at checkpoints only**: Don't use AI for things rules can handle
2. **Caching**: Classify once, reuse for similar books
3. **Batch processing**: Amortize overhead across multiple books
4. **Confidence thresholds**: Skip validation for very-high-confidence cases (future)

### Alternatives Considered

#### Option A: Local LLMs
Run open-source models locally (Llama, Mistral).

**Why we didn't choose this (yet):**
- Lower accuracy than Claude for nuanced classification
- Hardware requirements
- Maintenance burden
- May revisit when local models improve

#### Option B: Fine-tuned Models
Train custom models on our book corpus.

**Why we didn't choose this:**
- Cold start problem (need training data)
- Ongoing maintenance as book types evolve
- Overkill for ~100 books/month volume

---

## Security & Trust

### Data Privacy

- Books are processed locally
- Only book samples (first 5-10 pages) sent to LLM for classification
- No book content stored in external services
- LLM calls use API, not persistent storage

### Autonomy Safeguards

| Safeguard | Purpose |
|-----------|---------|
| Max auto-approvals/day | Prevent runaway automation |
| Escape hatch | Instant revert to manual mode |
| Spot-check sampling | Verify auto-approved quality |
| Anomaly detection | Alert on unusual patterns |

### Audit Trail

Every decision is logged with:
- Who made it (human vs AI)
- What information it was based on
- When it happened
- Full before/after state snapshots

This enables complete reconstruction of any decision.

---

## What We Explicitly Chose NOT To Build

### Not Building: Web Dashboard

**Rationale:**
- MCP tools provide equivalent functionality
- No additional infrastructure to maintain
- Conversational interface is more natural for approval workflow
- Can add later if needed

### Not Building: Real-time Processing

**Rationale:**
- Our volume (~100 books/month) doesn't require it
- Batch processing is simpler and more reliable
- File watcher provides "fast enough" detection

### Not Building: Multi-tenant Support

**Rationale:**
- Single-user system for now
- Adding auth/permissions adds complexity
- Can add later if sharing becomes needed

### Not Building: Custom Training

**Rationale:**
- Pre-trained LLMs work well for our use case
- Training requires labeled data we don't have
- Maintenance burden not justified at our scale

---

## Future Considerations

### Phase 2 Enhancements (After Foundation)

| Enhancement | Value | Complexity |
|-------------|-------|------------|
| Learning from corrections | Improve classifier accuracy | Medium |
| Book-to-book similarity | "Process like this similar book" | Medium |
| Content-based deduplication | Catch republished books | Low |

### Phase 3 Enhancements (After Confident Autonomy)

| Enhancement | Value | Complexity |
|-------------|-------|------------|
| Web dashboard | Visual approval interface | High |
| Multi-user support | Shared library access | High |
| Custom fine-tuning | Domain-specific accuracy | Very High |

### What Would Change Our Decisions

| If This Happens | We Might |
|-----------------|----------|
| Volume grows to 1000+ books/month | Add real-time processing, consider local LLMs |
| Multiple users need access | Add web UI, authentication |
| Specific domain focus (e.g., only medical texts) | Fine-tune classification models |
| LLM costs increase significantly | Shift more to local models |
| Local LLM quality improves | Replace cloud API calls |

---

## Summary

This design prioritizes:

1. **Pragmatism over perfection**: Hybrid AI + rules beats pure approaches
2. **Trust over speed**: Graduated autonomy builds confidence
3. **Simplicity over features**: MCP tools, not web dashboards
4. **Adaptability over rigidity**: AI handles novel cases, rules handle known patterns

The result is a system that:
- Costs ~$0.03-0.05 per book
- Handles diverse book types intelligently
- Starts supervised and earns autonomy
- Provides complete audit trails
- Can be extended as needs evolve

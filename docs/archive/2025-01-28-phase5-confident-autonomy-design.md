# Phase 5: Confident Autonomy Design

> **Audience:** Product managers, directors, UX designers, and engineers who need to understand what we're building and why.

---

## Status: 📋 Planned

**Target:** After Phase 4 Production Hardening is battle-tested with real usage

**Dependencies:**
- Phase 4 complete (Health monitoring, batch operations, audit trail)
- 100+ books processed with human review
- Audit data available for analysis

---

## Executive Summary

Phase 5 transforms the pipeline from "human reviews everything" to "human reviews only edge cases." The system earns trust through demonstrated accuracy, then gradually takes on more autonomous decision-making.

**The Problem:** In supervised mode, you review every book. This is appropriate when the system is new, but becomes tedious once you've seen the classifier correctly identify 100 technical tutorials in a row. You're spending time confirming what you already know.

**The Solution:** A graduated autonomy system where the AI earns trust through demonstrated accuracy. High-confidence decisions get auto-approved. Edge cases still come to you. One command reverts everything to supervised mode if something goes wrong.

**Example scenario:** The classifier has processed 200 books. For technical tutorials, it's been right 98% of the time. For textbooks, 95%. For magazines, only 78%. Phase 5 enables: auto-approve technical tutorials at ≥92% confidence, auto-approve textbooks at ≥90%, but always send magazines for human review. If a bad book slips through, activate the escape hatch and the system immediately reverts to full supervision.

---

## Terminology

| Term | Definition |
|------|------------|
| **Confidence** | The classifier's self-reported certainty (0-100%) that its classification is correct |
| **Calibration** | How well confidence matches actual accuracy (e.g., "90% confident" should be right 90% of the time) |
| **Override rate** | How often humans change the AI's decision (lower is better) |
| **Auto-approve threshold** | The minimum confidence required to skip human review |
| **Escape hatch** | One-command revert to fully supervised mode |
| **Spot-check** | Random sampling of auto-approved books to verify quality |

---

## What Is This?

Phase 5 adds calibrated autonomy:

| Capability | What It Does | Why It Matters |
|------------|--------------|----------------|
| **Autonomy Metrics** | Track actual vs. predicted accuracy | Know if confidence scores are trustworthy |
| **Graduated Thresholds** | Different auto-approve levels by book type | Technical tutorials earn trust faster than magazines |
| **Feedback Capture** | Record when humans correct the AI | System learns from mistakes |
| **Spot-Check Sampling** | Random audits of auto-approved books | Catch drift before it becomes a problem |
| **Escape Hatch** | Instant revert to supervised mode | Safety net when something goes wrong |

---

## The Trust Gradient

### Why Not Just Auto-Approve Everything?

Auto-approving books into your knowledge library is high stakes:
- **Bad books waste your time** — You'll search for information and get garbage
- **Mistakes compound** — You might reference incorrect information in your work
- **Trust is hard to rebuild** — One bad experience makes you doubt everything

We need to earn trust gradually, with evidence.

### The Three Autonomy Modes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AUTONOMY PROGRESSION                                  │
│                                                                              │
│   SUPERVISED                PARTIAL                    CONFIDENT             │
│   (Phase 1-4)              AUTONOMY                    AUTONOMY              │
│                                                                              │
│   ┌─────────────┐         ┌─────────────┐            ┌─────────────┐        │
│   │             │         │   Auto ✓    │            │   Auto ✓    │        │
│   │   Human     │   ───►  │   High      │    ───►    │   Most      │        │
│   │   Reviews   │         │   Conf.     │            │   Books     │        │
│   │   100%      │         │             │            │             │        │
│   │             │         │   Human     │            │   Human     │        │
│   │             │         │   Reviews   │            │   Reviews   │        │
│   │             │         │   Rest      │            │   Edge      │        │
│   │             │         │             │            │   Cases     │        │
│   └─────────────┘         └─────────────┘            └─────────────┘        │
│                                                                              │
│   Human workload:          Human workload:           Human workload:        │
│   100%                     ~30-50%                   ~5-15%                  │
│                                                                              │
│   Requirements:            Requirements:             Requirements:           │
│   - None                   - 100+ books              - 500+ books           │
│   - Starting mode          - <15% override rate      - <5% override rate    │
│                            - Calibration ±10%        - Calibration ±5%      │
│                                                      - 0 rollbacks in 30d   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Mode Definitions

**Supervised Mode (Current)**
- Human reviews 100% of books
- System collects accuracy data
- Building baseline for calibration

**Partial Autonomy**
- Auto-approve: confidence ≥ calculated threshold AND known book type AND zero issues
- Human review: everything else
- Weekly spot-check of 10% of auto-approved books

**Confident Autonomy**
- Auto-approve: confidence ≥ calculated threshold (relaxed constraints)
- Auto-retry: confidence 0.7-0.9 with adjustments
- Human review: confidence <0.7 OR unknown book type OR anomaly detected
- Monthly spot-check of 5% of auto-approved books

---

## How It Works

### The Calibration Engine

**What is calibration?**

Calibration measures whether the AI's confidence matches reality:

```
If the AI says "I'm 90% confident this is a technical tutorial"...
...is it actually right 90% of the time?

Good calibration:  90% confident → 90% accurate  ✓
Over-confident:    90% confident → 70% accurate  ✗ (dangerous!)
Under-confident:   90% confident → 98% accurate  ✓ (safe, just inefficient)
```

**Why calibration matters:**

| Calibration | Risk | Effect |
|-------------|------|--------|
| Well-calibrated | Low | Thresholds work as expected |
| Over-confident | HIGH | Bad books get auto-approved |
| Under-confident | Low | More human review than necessary |

We're conservative: if calibration is uncertain, we require higher confidence for auto-approve.

**How we measure calibration:**

```python
def calculate_calibration(book_type: str, confidence_bucket: range) -> float:
    """
    Calculate actual accuracy for books in a confidence range.

    Example:
        calculate_calibration("technical_tutorial", range(85, 95))

        Looks at all technical tutorials where AI confidence was 85-95%.
        Returns the percentage that were actually correct (not overridden by human).
    """
    books = query_books_by_confidence(book_type, confidence_bucket)
    correct = sum(1 for b in books if not b.was_overridden)
    return correct / len(books) if books else None
```

**Calibration buckets:**

| Bucket | Confidence Range | What We Learn |
|--------|------------------|---------------|
| High | 90-100% | How accurate are "certain" predictions? |
| Medium | 80-90% | Where do mistakes start appearing? |
| Low | 70-80% | How unreliable is "uncertain"? |
| Very Low | <70% | Should always go to human review |

### Threshold Calculation

**The formula:**

```python
def calculate_auto_approve_threshold(book_type: str) -> float:
    """
    Calculate the minimum confidence for auto-approval.

    We want: if we auto-approve at this threshold, we'll be right 95%+ of the time.
    """
    # Find the confidence level where actual accuracy ≥ 95%
    for threshold in [0.95, 0.92, 0.90, 0.88, 0.85]:
        accuracy = measure_accuracy_at_threshold(book_type, threshold)
        if accuracy >= 0.95:
            return threshold

    # If we can't find a safe threshold, require human review
    return None  # No auto-approve for this book type
```

**Per-book-type thresholds:**

Different content types earn trust at different rates:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AUTO-APPROVE THRESHOLDS BY BOOK TYPE                      │
│                                                                              │
│   Book Type              Sample   Accuracy   Calculated     Status          │
│                          Size     @ 90%      Threshold                       │
│   ─────────────────────────────────────────────────────────────────────────  │
│   technical_tutorial     127      97%        88%           ✓ Auto-approve   │
│   textbook               89       94%        92%           ✓ Auto-approve   │
│   reference_manual       45       91%        95%           ✓ Auto-approve   │
│   narrative_nonfiction   34       88%        None          ✗ Human review   │
│   magazine               23       76%        None          ✗ Human review   │
│   unknown                12       --         None          ✗ Always human   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why different thresholds?**

- **Technical tutorials** have clear signals (code blocks, chapter structure) — easy to classify accurately
- **Magazines** are diverse (some look like books, some don't) — harder to classify
- **Unknown types** by definition lack training data — always need human review

### The Escape Hatch

**What it is:**

One command that immediately reverts the system to supervised mode:

```bash
$ agentic-pipeline escape-hatch "Classifier seems off today"

⚠️  ESCAPE HATCH ACTIVATED

All autonomy disabled. Reverting to supervised mode.
- All new books require human approval
- Auto-approve thresholds disabled
- Reason logged: "Classifier seems off today"

To resume autonomy: agentic-pipeline autonomy resume
```

**When to use it:**

| Scenario | Action |
|----------|--------|
| You notice an incorrectly classified book | Investigate first, escape hatch if pattern |
| Multiple bad books in one day | Escape hatch immediately |
| LLM provider having issues | Escape hatch until resolved |
| You're going on vacation | Consider escape hatch (or just let it run) |
| Feeling uncertain | Escape hatch — it's free and safe |

**What happens when activated:**

1. `autonomy_config.escape_hatch_active` set to TRUE
2. All books immediately require human review
3. In-progress processing continues (doesn't interrupt)
4. Reason and timestamp logged to audit trail
5. Alert generated for visibility

**Resuming autonomy:**

```bash
$ agentic-pipeline autonomy resume

Escape hatch was active for: 2 days, 4 hours
Reason: "Classifier seems off today"

During escape hatch:
- 34 books processed
- 32 approved by human
- 2 rejected by human

Ready to resume autonomy?
- Current calibration still valid (last updated 3 days ago)
- Threshold for technical_tutorial: 88%
- Threshold for textbook: 92%

Resume? [y/N]
```

---

## Key Design Decisions

### Decision 1: Why Graduated Trust Instead of Binary On/Off?

**What we chose:** Three autonomy modes (supervised, partial, confident) with per-book-type thresholds.

**The Options We Considered:**

| Approach | Pros | Cons |
|----------|------|------|
| **Binary on/off** | Simple | All-or-nothing feels risky |
| **Single global threshold** | Still simple | Ignores that some types are easier than others |
| **Per-book-type thresholds** ✓ | Matches reality | More complex to understand |
| **Per-book thresholds** | Maximum precision | Over-engineering, hard to audit |

**Why we chose graduated, per-book-type:**

1. **Matches user mental model:** "I trust it with tech books, not with magazines" is a natural way to think
2. **Earns trust incrementally:** You can enable autonomy for easy types while keeping hard types supervised
3. **Transparent:** You can see exactly why a book was auto-approved or sent for review
4. **Safe failure mode:** If a book type has insufficient data, it defaults to human review

**What this looks like:**

```bash
$ agentic-pipeline autonomy status

Autonomy Mode: PARTIAL

Per-Type Thresholds:
  technical_tutorial:   88% (auto-approve)  [127 samples, 97% accuracy]
  textbook:            92% (auto-approve)  [89 samples, 94% accuracy]
  reference_manual:    95% (auto-approve)  [45 samples, 91% accuracy]
  narrative_nonfiction: -- (human review)  [34 samples, insufficient accuracy]
  magazine:            -- (human review)  [23 samples, insufficient accuracy]
  unknown:             -- (human review)  [always requires human]

Today: 12 auto-approved, 3 human review, 0 rejected
```

---

### Decision 2: Why 95% Target Accuracy for Auto-Approve?

**What we chose:** Auto-approve threshold set where actual accuracy ≥ 95%.

**The Options We Considered:**

| Target | Risk | User Experience |
|--------|------|-----------------|
| 99% accuracy | Very low risk | Few books auto-approved, lots of human review |
| **95% accuracy** ✓ | Low risk | Good balance of autonomy and safety |
| 90% accuracy | Moderate risk | More autonomy, but 1 in 10 could be wrong |
| 85% accuracy | Higher risk | Significant chance of bad books |

**Why 95%:**

1. **1 in 20 error rate is acceptable:** If 5% of auto-approved books need correction, you'll catch them in spot-checks or when you use them
2. **Matches human error rate:** Humans make mistakes too — 95% AI accuracy is comparable to careful human review
3. **Achievable:** Our classifier actually reaches 95%+ for well-defined book types
4. **Not perfectionist:** Requiring 99% would mean almost everything needs human review, defeating the purpose

**Adjustable per deployment:**

```python
# For a more conservative setup (e.g., professional library)
AUTO_APPROVE_MIN_ACCURACY = 0.98  # 98% accuracy required

# For a more permissive setup (e.g., personal collection)
AUTO_APPROVE_MIN_ACCURACY = 0.90  # 90% accuracy acceptable
```

---

### Decision 3: Why Spot-Check Sampling Instead of Full Audit?

**What we chose:** Random sampling of 5-10% of auto-approved books for human review.

**The Options We Considered:**

| Approach | Pros | Cons |
|----------|------|------|
| **No post-approval audit** | Maximum efficiency | Won't catch drift |
| **Spot-check sampling** ✓ | Catches drift, manageable workload | Statistical, not exhaustive |
| **Delayed full audit** | Eventually reviews everything | Defeats purpose of autonomy |
| **Audit on use** | Reviews when you actually need the book | Problems discovered late |

**Why spot-checking:**

1. **Catches drift:** If the classifier starts making more mistakes, you'll notice in spot-checks before it becomes a big problem
2. **Manageable workload:** 5% of 100 books = 5 reviews per week, not burdensome
3. **Statistical validity:** With 5% sampling, you have 95% confidence of detecting a 10% error rate increase
4. **Maintains trust:** Knowing there's ongoing verification builds confidence in the system

**How spot-checking works:**

```python
def select_spot_check_books(period: str = "week") -> list[Book]:
    """Select random books for spot-check review."""

    auto_approved = get_auto_approved_books(period)
    sample_rate = 0.10 if autonomy_mode == "partial" else 0.05

    # Stratified sampling: proportional to book type
    sample = []
    for book_type in auto_approved.group_by("book_type"):
        type_sample = random.sample(
            book_type.books,
            max(1, int(len(book_type.books) * sample_rate))
        )
        sample.extend(type_sample)

    return sample
```

```bash
$ agentic-pipeline spot-check

Weekly Spot-Check Review
========================

5 books selected for review (10% of 48 auto-approved):

1. "Python Machine Learning" [technical_tutorial, 94% conf]
   → Classification correct? [y/n]
   → Quality acceptable? [y/n]

2. "Database Design Fundamentals" [textbook, 91% conf]
   → Classification correct? [y/n]
   → Quality acceptable? [y/n]

...

Results: 5/5 correct (100%)
Calibration remains valid. No action needed.
```

---

### Decision 4: Why Per-Book-Type Calibration Instead of Global?

**What we chose:** Separate calibration and thresholds for each book type.

**The Options We Considered:**

| Approach | Pros | Cons |
|----------|------|------|
| **Global threshold** | Simple, one number | Ignores that types vary in difficulty |
| **Per-type threshold** ✓ | Matches reality | More complex, need data per type |
| **Per-book threshold** | Maximum precision | Over-fitting, hard to explain |
| **Dynamic per-session** | Adapts in real-time | Too volatile, hard to trust |

**Why per-book-type:**

1. **Reality:** Technical tutorials are genuinely easier to classify than narrative nonfiction. A global threshold would be too conservative for easy types or too permissive for hard types.

2. **Transparency:** "Technical tutorials auto-approve at 88%" is understandable. "This specific book auto-approved because of 47 micro-factors" is not.

3. **Graceful degradation:** New or rare book types simply don't get auto-approved until we have enough data.

**Minimum sample sizes:**

| Autonomy Mode | Minimum Samples Required |
|---------------|-------------------------|
| Partial | 50 books of that type |
| Confident | 100 books of that type |

If we have fewer samples, that book type stays in human review.

---

### Decision 5: Why an Escape Hatch Instead of Gradual Degradation?

**What we chose:** A single command that immediately reverts to full supervision.

**The Options We Considered:**

| Approach | Pros | Cons |
|----------|------|------|
| **Gradual degradation** | Smooth transition | Slow response to problems |
| **Escape hatch** ✓ | Instant safety | Binary, no middle ground |
| **Automatic detection** | No human action needed | False positives could be disruptive |
| **No revert option** | Simplest | Trapped if something goes wrong |

**Why escape hatch:**

1. **Psychology of trust:** Knowing you can instantly revert makes you more willing to try autonomy. It's a safety net.

2. **Fast response:** If something is wrong, you want to stop it NOW, not gradually over days.

3. **Clear state:** Either autonomy is on or it's off. No ambiguous middle states to debug.

4. **Human judgment:** You're better at detecting "something feels wrong" than an algorithm. The escape hatch lets you act on that intuition.

**Why not automatic detection?**

We considered having the system automatically detect problems and revert, but rejected it:

- **False positives are costly:** Auto-reverting when nothing is wrong wastes human time
- **Human context matters:** A spike in "unusual" books might be a new topic you're exploring, not a classifier problem
- **Humans are good at this:** You'll notice if search results seem wrong or a bad book appeared

The system *alerts* you to anomalies. You *decide* whether to activate the escape hatch.

---

### Decision 6: Why Not Learn Directly from Corrections?

**What we chose:** Capture feedback for analysis but don't automatically retrain the classifier.

**The Options We Considered:**

| Approach | Pros | Cons |
|----------|------|------|
| **No learning** | Stable, predictable | Doesn't improve over time |
| **Capture feedback, manual retrain** ✓ | Controlled improvement | Requires periodic human effort |
| **Automatic retraining** | Continuous improvement | Risk of drift, hard to audit |
| **Online learning** | Real-time adaptation | Unstable, could go off the rails |

**Why we capture but don't auto-retrain:**

1. **Stability:** The classifier is a black box (LLM). Auto-retraining could introduce unpredictable changes.

2. **Auditability:** If something goes wrong, you need to know what changed. "The model auto-updated 47 times" is unauditable.

3. **Appropriate feedback loop:** Feedback informs threshold adjustments (safe, reversible) rather than model weights (risky, irreversible).

**How feedback is used:**

```python
# What we capture
feedback = {
    "book_id": "abc123",
    "ai_classification": "technical_tutorial",
    "ai_confidence": 0.89,
    "human_classification": "textbook",  # Human corrected it
    "human_notes": "Has tutorial elements but is primarily a textbook"
}

# What we do with it
# 1. Recalculate calibration (immediately)
recalculate_calibration("technical_tutorial")
recalculate_calibration("textbook")

# 2. Adjust thresholds if needed (immediately)
if calibration_dropped("technical_tutorial"):
    raise_threshold("technical_tutorial")

# 3. Flag for model review (periodically)
# A human reviews accumulated feedback and decides if prompt changes are needed
```

---

## Component Details

### 1. Autonomy Metrics

**Purpose:** Track actual vs. predicted accuracy to determine if auto-approve is safe.

**Data collected:**

```sql
CREATE TABLE autonomy_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,

    -- Volume
    total_processed INTEGER DEFAULT 0,
    auto_approved INTEGER DEFAULT 0,
    human_approved INTEGER DEFAULT 0,
    human_rejected INTEGER DEFAULT 0,
    human_adjusted INTEGER DEFAULT 0,  -- Approved with corrections

    -- Accuracy
    avg_confidence_auto_approved REAL,
    avg_confidence_human_approved REAL,
    avg_confidence_human_rejected REAL,

    -- Quality tracking
    auto_approved_later_rolled_back INTEGER DEFAULT 0,
    human_approved_later_rolled_back INTEGER DEFAULT 0,

    -- Per-type breakdown (JSON)
    metrics_by_type JSON,

    -- Calibration data (JSON)
    confidence_buckets JSON,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(period_start, period_end)
);
```

**Metrics dashboard:**

```bash
$ agentic-pipeline autonomy metrics

Autonomy Metrics (Last 30 Days)
===============================

Volume:
  Total processed:     234
  Auto-approved:       156 (67%)
  Human approved:       71 (30%)
  Human rejected:        7 (3%)

Accuracy:
  Auto-approve accuracy:  98.7% (2 later rolled back)
  Override rate:          4.2% (human changed AI decision)

Calibration:
  90-100% confidence:  Actual 97% accuracy ✓
  80-90% confidence:   Actual 89% accuracy ✓
  70-80% confidence:   Actual 76% accuracy (human review)

Per-Type Performance:
  technical_tutorial:  94% auto-approved, 99% accurate
  textbook:           78% auto-approved, 96% accurate
  magazine:            0% auto-approved (always human)
```

---

### 2. Feedback Capture

**Purpose:** Record when humans correct or adjust the AI's decisions.

**Events captured:**

| Event | What It Means | Impact |
|-------|---------------|--------|
| `CLASSIFICATION_CORRECTED` | Human changed book type | Recalculate calibration |
| `CONFIDENCE_DISAGREEMENT` | Human rejected high-confidence book | Flag for review |
| `APPROVAL_WITH_ADJUSTMENTS` | Approved but with changes | Minor issue, track patterns |
| `ROLLBACK_AUTO_APPROVED` | Auto-approved book removed later | Serious, may trigger escape hatch |

**Feedback schema:**

```sql
CREATE TABLE autonomy_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id TEXT NOT NULL,
    pipeline_id TEXT,

    -- AI's decision
    original_decision TEXT NOT NULL,      -- "auto_approved", "human_review"
    original_confidence REAL,
    original_book_type TEXT,

    -- Human's correction
    human_decision TEXT NOT NULL,         -- "approved", "rejected", "adjusted"
    human_adjustments JSON,               -- What was changed

    -- Categorization
    feedback_category TEXT,               -- "misclassification", "quality_issue", etc.
    feedback_notes TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

### 3. Threshold Calculator

**Purpose:** Determine safe auto-approve thresholds based on historical accuracy.

**Algorithm:**

```python
class ThresholdCalculator:
    """Calculates safe auto-approve thresholds."""

    def __init__(
        self,
        target_accuracy: float = 0.95,
        min_samples: int = 50
    ):
        self.target_accuracy = target_accuracy
        self.min_samples = min_samples

    def calculate_threshold(self, book_type: str) -> Optional[float]:
        """
        Find the minimum confidence level where accuracy >= target.
        Returns None if insufficient data or accuracy not achievable.
        """
        samples = self.get_samples(book_type)

        if len(samples) < self.min_samples:
            return None  # Not enough data

        # Try progressively lower thresholds
        for threshold in [0.95, 0.92, 0.90, 0.88, 0.85, 0.82, 0.80]:
            above_threshold = [s for s in samples if s.confidence >= threshold]

            if len(above_threshold) < 10:
                continue  # Need at least 10 samples in this range

            accuracy = sum(1 for s in above_threshold if s.was_correct) / len(above_threshold)

            if accuracy >= self.target_accuracy:
                return threshold

        return None  # Can't achieve target accuracy
```

**Threshold safety margins:**

```python
# We add a safety margin to calculated thresholds
SAFETY_MARGIN = 0.02  # 2 percentage points

# If calculated threshold is 88%, we use 90%
# This accounts for statistical variance and provides buffer
```

---

### 4. Spot-Check System

**Purpose:** Ongoing verification that auto-approved books are correct.

**Selection algorithm:**

```python
def select_spot_check_batch() -> list[Book]:
    """Select books for periodic spot-check review."""

    # Get recent auto-approved books
    period = "week" if autonomy_mode == "partial" else "month"
    candidates = get_auto_approved_books(period, not_yet_spot_checked=True)

    # Determine sample size
    sample_rate = 0.10 if autonomy_mode == "partial" else 0.05
    target_count = max(5, int(len(candidates) * sample_rate))

    # Stratified sampling by book type (proportional)
    sample = []
    for book_type, books in candidates.group_by("book_type").items():
        type_count = max(1, int(len(books) * sample_rate))
        sample.extend(random.sample(books, min(type_count, len(books))))

    # Also include any that were auto-approved at lower confidence
    # (these are higher risk, always include)
    low_confidence = [b for b in candidates if b.confidence < 0.90]
    sample.extend(low_confidence[:5])  # Up to 5 extra

    return sample[:target_count]
```

**Spot-check workflow:**

```bash
$ agentic-pipeline spot-check start

Starting spot-check review...

Book 1 of 6: "Advanced Python Techniques"
  Classification: technical_tutorial (92% confidence)
  Auto-approved: 3 days ago

  Quick checks:
  1. Is "technical_tutorial" correct? [y/n/unsure]: y
  2. Did processing look correct? [y/n/didn't check]: y

  Result: ✓ Confirmed correct

Book 2 of 6: "Data Science Handbook"
  Classification: textbook (89% confidence)
  Auto-approved: 5 days ago

  Quick checks:
  1. Is "textbook" correct? [y/n/unsure]: n
  2. What should it be? reference_manual

  Result: ✗ Misclassified (feedback recorded)

...

Spot-Check Summary
==================
6 books reviewed
5 correct (83%)
1 misclassified

⚠️  Accuracy below 90% threshold.
    Consider: Raising threshold for textbook type?
    Run: agentic-pipeline autonomy recalibrate
```

---

### 5. Escape Hatch

**Purpose:** Instant revert to supervised mode.

**Implementation:**

```python
def activate_escape_hatch(reason: str) -> None:
    """Immediately disable all autonomy."""

    # Update config
    db.execute("""
        UPDATE autonomy_config SET
            escape_hatch_active = TRUE,
            escape_hatch_activated_at = CURRENT_TIMESTAMP,
            escape_hatch_reason = ?
        WHERE id = 1
    """, [reason])

    # Log to audit trail
    audit.log(
        action="ESCAPE_HATCH_ACTIVATED",
        actor="human:cli",
        reason=reason,
        before_state={"autonomy_mode": current_mode},
        after_state={"autonomy_mode": "supervised"}
    )

    # Generate alert
    alerts.send(
        severity="warning",
        message=f"Escape hatch activated: {reason}",
        action_required="Review and resume when ready"
    )
```

**Escape hatch state:**

```sql
-- Part of autonomy_config table
escape_hatch_active BOOLEAN DEFAULT FALSE,
escape_hatch_activated_at TIMESTAMP,
escape_hatch_reason TEXT
```

**Auto-activation triggers (optional):**

```python
AUTO_ESCAPE_TRIGGERS = {
    # These can auto-activate escape hatch (configurable)
    "rollback_auto_approved": {
        "threshold": 2,      # 2 rollbacks in one day
        "period": "day",
        "enabled": True
    },
    "spot_check_failure_rate": {
        "threshold": 0.20,   # 20% of spot-checks incorrect
        "period": "week",
        "enabled": False     # Disabled by default, human decision
    }
}
```

---

## Database Schema

### New Tables

```sql
-- Autonomy configuration (singleton)
CREATE TABLE autonomy_config (
    id INTEGER PRIMARY KEY CHECK (id = 1),

    -- Current mode
    current_mode TEXT DEFAULT 'supervised',  -- supervised, partial, confident

    -- Global thresholds
    auto_approve_threshold REAL DEFAULT 0.95,
    auto_retry_threshold REAL DEFAULT 0.70,

    -- Safety constraints
    require_known_book_type BOOLEAN DEFAULT TRUE,
    require_zero_issues BOOLEAN DEFAULT TRUE,
    max_auto_approvals_per_day INTEGER DEFAULT 50,

    -- Spot-check settings
    spot_check_percentage REAL DEFAULT 0.10,

    -- Escape hatch
    escape_hatch_active BOOLEAN DEFAULT FALSE,
    escape_hatch_activated_at TIMESTAMP,
    escape_hatch_reason TEXT,

    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Per-type thresholds
CREATE TABLE autonomy_thresholds (
    book_type TEXT PRIMARY KEY,
    auto_approve_threshold REAL,          -- NULL = always human review
    sample_count INTEGER NOT NULL,
    measured_accuracy REAL NOT NULL,
    last_calculated TIMESTAMP NOT NULL,

    -- Calibration data
    calibration_data JSON,                -- Per-bucket accuracy

    -- Manual override
    manual_override REAL,                 -- Human can set threshold
    override_reason TEXT
);

-- Metrics (daily aggregates)
CREATE TABLE autonomy_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,

    total_processed INTEGER DEFAULT 0,
    auto_approved INTEGER DEFAULT 0,
    human_approved INTEGER DEFAULT 0,
    human_rejected INTEGER DEFAULT 0,
    human_adjusted INTEGER DEFAULT 0,

    avg_confidence_auto_approved REAL,
    avg_confidence_human_approved REAL,
    avg_confidence_human_rejected REAL,

    auto_approved_later_rolled_back INTEGER DEFAULT 0,
    human_approved_later_rolled_back INTEGER DEFAULT 0,

    metrics_by_type JSON,
    confidence_buckets JSON,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(period_start, period_end)
);

-- Feedback log
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

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Spot-check tracking
CREATE TABLE spot_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id TEXT NOT NULL,
    pipeline_id TEXT,

    -- What was checked
    original_classification TEXT,
    original_confidence REAL,
    auto_approved_at TIMESTAMP,

    -- Check results
    classification_correct BOOLEAN,
    quality_acceptable BOOLEAN,
    reviewer TEXT,
    notes TEXT,

    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## CLI Commands

### New Commands

```bash
# Autonomy status and control
agentic-pipeline autonomy status          # Show current mode and thresholds
agentic-pipeline autonomy enable partial  # Enable partial autonomy
agentic-pipeline autonomy enable confident # Enable confident autonomy
agentic-pipeline autonomy disable         # Revert to supervised
agentic-pipeline autonomy recalibrate     # Recalculate all thresholds

# Escape hatch
agentic-pipeline escape-hatch "reason"    # Activate escape hatch
agentic-pipeline autonomy resume          # Resume after escape hatch

# Metrics and analysis
agentic-pipeline autonomy metrics         # Show accuracy metrics
agentic-pipeline autonomy calibration     # Show calibration data
agentic-pipeline autonomy readiness       # Check if ready for next mode

# Spot-checks
agentic-pipeline spot-check               # Start spot-check session
agentic-pipeline spot-check status        # Show pending/completed checks

# Thresholds
agentic-pipeline autonomy thresholds      # Show per-type thresholds
agentic-pipeline autonomy set-threshold <type> <value>  # Manual override
```

---

## MCP Tools

### New Tools

```python
@mcp.tool()
def get_autonomy_status() -> dict:
    """Get current autonomy mode, thresholds, and metrics summary."""

@mcp.tool()
def get_autonomy_readiness() -> dict:
    """Check if system is ready to advance to next autonomy mode."""

@mcp.tool()
def set_autonomy_mode(mode: str) -> dict:
    """Change autonomy mode. Modes: supervised, partial, confident."""

@mcp.tool()
def activate_escape_hatch(reason: str) -> dict:
    """Immediately revert to supervised mode."""

@mcp.tool()
def get_calibration_data(book_type: str = None) -> dict:
    """Get calibration metrics, optionally filtered by book type."""

@mcp.tool()
def get_spot_check_queue() -> list[dict]:
    """Get books pending spot-check review."""

@mcp.tool()
def submit_spot_check(
    book_id: str,
    classification_correct: bool,
    quality_acceptable: bool,
    notes: str = None
) -> dict:
    """Submit spot-check review results."""
```

---

## Success Criteria

### For Partial Autonomy

| Criterion | Target | How We Measure |
|-----------|--------|----------------|
| Sample size | 100+ books | Total processed in supervised mode |
| Override rate | <15% | Human corrections / total decisions |
| Calibration | ±10% | |predicted confidence - actual accuracy| |
| Rollback rate | <5% | Books removed after approval |

### For Confident Autonomy

| Criterion | Target | How We Measure |
|-----------|--------|----------------|
| Sample size | 500+ books | Total processed |
| Override rate | <5% | Human corrections / total decisions |
| Calibration | ±5% | |predicted confidence - actual accuracy| |
| Rollback rate | <1% | Books removed after approval |
| Spot-check accuracy | >95% | Correct / total spot-checked |

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Over-confident model | Medium | High | Conservative thresholds, escape hatch |
| Calibration drift | Medium | Medium | Ongoing spot-checks, recalibration |
| User ignores spot-checks | Medium | Medium | Reminders, auto-pause if overdue |
| Escape hatch overused | Low | Low | Track activation history, encourage resumption |
| Bad books in library | Low | High | Rollback capability, audit trail |

---

## What's NOT in Phase 5

Explicitly out of scope:

1. **Automatic model retraining** — Too risky for a single-user system
2. **Multi-user autonomy settings** — Single user only for now
3. **Real-time threshold adjustment** — Thresholds update on schedule, not per-book
4. **Complex anomaly detection** — Simple heuristics only; human judgment for edge cases

---

## Implementation Order

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 5 IMPLEMENTATION ORDER                          │
│                                                                              │
│  1. Database schema ─────────────────────────────────────────────────────┐  │
│     (autonomy_config, thresholds, metrics, feedback tables)              │  │
│                           │                                               │  │
│                           ▼                                               │  │
│  2. Metrics collection ──────────────────────────────────────────────────┤  │
│     (Track approvals, overrides, rollbacks during supervised mode)       │  │
│                           │                                               │  │
│                           ▼                                               │  │
│  3. Calibration engine ──────────────────────────────────────────────────┤  │
│     (Calculate accuracy by confidence bucket, per book type)             │  │
│                           │                                               │  │
│                           ▼                                               │  │
│  4. Threshold calculator ────────────────────────────────────────────────┤  │
│     (Determine safe auto-approve thresholds)                             │  │
│                           │                                               │  │
│                           ▼                                               │  │
│  5. Escape hatch ────────────────────────────────────────────────────────┤  │
│     (Must work before enabling any autonomy)                             │  │
│                           │                                               │  │
│                           ▼                                               │  │
│  6. Partial autonomy mode ───────────────────────────────────────────────┤  │
│     (Auto-approve with strict constraints)                               │  │
│                           │                                               │  │
│                           ▼                                               │  │
│  7. Spot-check system ───────────────────────────────────────────────────┤  │
│     (Ongoing verification)                                               │  │
│                           │                                               │  │
│                           ▼                                               │  │
│  8. Confident autonomy mode ─────────────────────────────────────────────┘  │
│     (Relaxed constraints, lower sample rate)                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

Phase 5 is about **earning trust through evidence**:

| Before Phase 5 | After Phase 5 |
|----------------|---------------|
| Human reviews 100% | Human reviews ~10% |
| Same process for all books | Easy books auto-approved |
| No visibility into accuracy | Calibration metrics dashboard |
| All-or-nothing trust | Graduated trust by book type |
| No safety net | Instant escape hatch |

The key insight: **Autonomy isn't about removing humans from the loop. It's about focusing human attention where it matters most.**

High-confidence technical tutorials don't need your review. Borderline magazines do. Phase 5 makes that distinction automatically, while keeping you in control.

---

## Related Documentation

- [Phase 4 Production Hardening](./2025-01-28-phase4-production-hardening-design.md) — Prerequisites
- [Agentic Pipeline Design](./2025-01-28-agentic-processing-pipeline-design.md) — Overall system architecture
- [Phase 4 Complete](../PHASE4-PRODUCTION-HARDENING-COMPLETE.md) — Current implementation status

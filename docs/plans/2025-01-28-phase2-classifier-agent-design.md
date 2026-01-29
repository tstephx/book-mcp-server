# Phase 2: Classifier Agent Design

> **Audience:** Product managers, directors, UX designers, and engineers who need to understand what we're building and why.

---

## What Is This?

The Classifier Agent is the "brain" that looks at a new book and answers: *What kind of book is this?*

When you drop a book into the system, it needs to figure out whether it's a technical tutorial, a magazine, a biography, or something else. This matters because different book types need different processing strategies—a programming book with code samples needs to preserve formatting differently than a newspaper article.

**Today:** A human would need to manually tag each book's type.

**With the Classifier Agent:** An AI reads a sample of the book and makes this decision automatically, with a confidence score. High confidence books proceed automatically. Low confidence books get flagged for human review.

---

## How It Works

### The Simple Version

```
Book arrives → AI reads a sample → AI returns classification → Pipeline continues
```

### The Detailed Version

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLASSIFIER AGENT                         │
│                                                                  │
│  1. CHECK FOR DUPLICATES                                        │
│     "Have we seen this exact book before?"                      │
│     └─ Yes → Return the previous classification (skip AI call)  │
│     └─ No  → Continue to step 2                                 │
│                                                                  │
│  2. CALL PRIMARY AI (OpenAI GPT-4o-mini)                        │
│     "Read this text and tell me what type of book it is"        │
│     └─ Success → Return classification                          │
│     └─ Failure → Continue to step 3                             │
│                                                                  │
│  3. CALL BACKUP AI (Anthropic Claude)                           │
│     Same request, different provider                            │
│     └─ Success → Return classification                          │
│     └─ Failure → Continue to step 4                             │
│                                                                  │
│  4. SAFE DEFAULT                                                │
│     Return "unknown" with 0% confidence                         │
│     (This triggers human review via existing approval flow)     │
└─────────────────────────────────────────────────────────────────┘
```

### What the AI Returns

For each book, the classifier returns four pieces of information:

| Field | Example | Why It Matters |
|-------|---------|----------------|
| **book_type** | "technical_tutorial" | Determines which processing strategy to use |
| **confidence** | 0.85 (85%) | Decides if human review is needed |
| **suggested_tags** | ["python", "web development"] | Helps with search and organization later |
| **reasoning** | "Contains step-by-step code examples with explanations" | Audit trail—we can see *why* it decided this |

### Book Types We Classify

| Type | Description | Example |
|------|-------------|---------|
| technical_tutorial | Step-by-step guides teaching skills | "Learning Python" |
| technical_reference | Reference manuals, documentation | "PostgreSQL Manual" |
| textbook | Academic material with exercises | "Introduction to Economics" |
| narrative_nonfiction | Stories, biographies, essays | "Steve Jobs" by Isaacson |
| periodical | Magazines, newspapers, journals | "The New Yorker", "NY Times" |
| research_collection | Academic papers, proceedings | "NeurIPS 2024 Papers" |
| unknown | Cannot determine confidently | (Triggers human review) |

---

## Key Design Decisions

### Decision 1: Why Two AI Providers?

**What we chose:** OpenAI as primary, Anthropic Claude as backup.

**Why:**

| Consideration | OpenAI (GPT-4o-mini) | Anthropic (Claude) |
|---------------|---------------------|-------------------|
| Cost per book | ~$0.001 | ~$0.002 - $0.02 |
| Quality | Excellent for classification | Excellent for classification |
| Reliability | 99.9% uptime, occasional hiccups | 99.9% uptime, occasional hiccups |

Using two providers means if OpenAI has an outage (rare, but happens), we automatically switch to Claude. The user never notices—books keep flowing.

**Alternatives we considered:**

- **Single provider (OpenAI only):** Simpler, but a 30-minute OpenAI outage would halt all book processing. Not worth the risk.
- **Three+ providers:** Diminishing returns. Two is enough redundancy for our volume.
- **Local AI models (Ollama):** Lower quality, requires GPU hardware, more maintenance. Not worth it yet—may revisit if costs become an issue at scale.

---

### Decision 2: Why GPT-4o-mini as Default?

**What we chose:** GPT-4o-mini (~$0.001/book) over Claude Sonnet (~$0.02/book).

**Why:**

This is a *classification task*, not creative writing or complex reasoning. We're asking "is this a textbook or a magazine?"—not "write me a novel." Smaller, cheaper models handle classification just as well as expensive ones.

**The math:**
- 100 books/month × $0.001 = $0.10/month with GPT-4o-mini
- 100 books/month × $0.02 = $2.00/month with Claude Sonnet

The cost difference is small at our volume, but there's no reason to pay 20x more for equivalent results.

**Alternatives we considered:**

- **Claude Sonnet as default:** Better for nuanced tasks, but overkill for classification. We'd pay more for no measurable improvement.
- **Free/local models:** Quality isn't there yet for reliable classification. We'd get more "unknown" results, meaning more human review, which defeats the purpose.

---

### Decision 3: Why Pre-Extracted Text (Not Raw Files)?

**What we chose:** The classifier receives plain text that's already been extracted from the book file.

**Why:**

Separation of concerns. The classifier's job is *classification*, not file parsing. By the time text reaches the classifier:
- EPUB structure has been parsed
- PDF text has been extracted
- Encoding issues have been resolved

The classifier just sees clean text and focuses on its one job.

**Alternatives we considered:**

- **Classifier reads files directly:** Would require PDF/EPUB parsing logic inside the classifier. Makes it more complex, harder to test, and duplicates code we already have in the existing pipeline.
- **Classifier receives file path:** Same problem—it would need to know how to open and parse different file formats.

---

### Decision 4: Why Cache in the Pipeline Table?

**What we chose:** Reuse the existing `processing_pipelines` table to store classification results. If we've seen the same book (by content hash), reuse the previous classification.

**Why:**

We already compute a content hash for duplicate detection. We already store the book profile in the pipeline record. Adding a separate cache table would mean:
- Another table to maintain
- Duplicate data
- More code

The simpler approach: check if `find_by_hash()` returns an existing profile. If yes, use it. If no, call the AI.

**Alternatives we considered:**

- **Separate cache table:** Cleaner separation, but adds complexity we don't need. At 100 books/month, the coupling between classifier and pipeline repo is fine.
- **Redis/external cache:** Massive overkill. We're not building Twitter.
- **No caching at all:** Would work—$0.001/book is cheap—but why pay twice to classify the same book? Caching is free.

---

### Decision 5: Why Return "Unknown" on Failure?

**What we chose:** If both AI providers fail, return `book_type: "unknown"` with `confidence: 0.0`.

**Why:**

The existing approval flow already handles low-confidence books by routing them to human review. By returning "unknown" with zero confidence, we trigger that flow automatically. No special error handling needed downstream.

**The alternative approaches:**

| Approach | Problem |
|----------|---------|
| Throw an error | Pipeline halts. Requires special error handling everywhere. |
| Retry indefinitely | Could get stuck forever if there's a real problem. |
| Skip the book | Book disappears silently. Bad user experience. |
| Return "unknown" ✓ | Book proceeds to human review. Human can classify it manually. |

---

### Decision 6: Why This Output Format?

**What we chose:**
```json
{
  "book_type": "technical_tutorial",
  "confidence": 0.85,
  "suggested_tags": ["python", "web development"],
  "reasoning": "Contains step-by-step code examples..."
}
```

**Why each field:**

| Field | Purpose | Who uses it |
|-------|---------|-------------|
| book_type | Determines processing strategy | Pipeline automation |
| confidence | Decides if human review needed | Approval flow |
| suggested_tags | Helps with search/organization | End users, future features |
| reasoning | Explains the decision | Debugging, auditing, human reviewers |

**Alternatives we considered:**

- **Minimal (just book_type + confidence):** Works, but no visibility into *why* the AI decided something. When a human reviewer sees a low-confidence book, they'd have no context.
- **Rich (add detected_features, language, page count):** More data, but we don't have a use for it yet. Can add later if needed. YAGNI.

---

## How Confidence Drives the Flow

The confidence score determines what happens next:

```
Confidence ≥ 0.7    →  Proceed automatically to strategy selection
Confidence < 0.7    →  Flag for human review before continuing
Confidence = 0.0    →  Classification failed, definitely needs human review
```

**Why 0.7 as the threshold?**

It's a starting point. We'll calibrate based on real data:
- If 70%+ confidence books are usually correct → threshold is good
- If 70%+ confidence books are often wrong → raise the threshold
- If 70%+ confidence books are always correct → could lower it to reduce human review

This is Phase 1 of autonomy—we're being conservative. The threshold can change as we build trust in the system.

---

## What This Enables (Future)

With classification working, we unlock:

| Feature | How classification enables it |
|---------|------------------------------|
| **Auto-approval** | High confidence + known book type → no human needed |
| **Smart strategies** | Right processing approach for each book type |
| **Better search** | Suggested tags improve discoverability |
| **Learning from corrections** | When humans fix classifications, we can tune prompts |
| **Quality metrics** | Track accuracy by book type, identify problem areas |

---

## Cost Projections

| Volume | Monthly Cost | Notes |
|--------|--------------|-------|
| 100 books | ~$0.10 | Current expected volume |
| 500 books | ~$0.50 | 5x growth |
| 1,000 books | ~$1.00 | 10x growth |
| 10,000 books | ~$10.00 | Would revisit architecture at this scale |

The classifier is not a significant cost driver. Human review time saved is worth far more than $0.10/month.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| AI returns wrong classification | Medium | Low | Human review catches errors; we learn and tune prompts |
| Both AI providers down simultaneously | Very Low | Medium | Returns "unknown", human reviews manually |
| AI costs increase significantly | Low | Low | Can switch to local models if needed; architecture supports it |
| New book type we haven't seen | Medium | Low | Gets classified as "unknown", human categorizes it, we add new type |

---

## Success Metrics

How we'll know this is working:

| Metric | Target | How we measure |
|--------|--------|----------------|
| Classification accuracy | >90% | Compare AI classification to human corrections |
| Human review rate | <20% | % of books flagged as low confidence |
| Time to classify | <5 seconds | API response time monitoring |
| Cost per book | <$0.01 | Track API spend / books processed |

---

## Summary

The Classifier Agent is a thin, focused component that does one thing well: determine what type of book we're processing. It's:

- **Cost-effective** — $0.001/book with GPT-4o-mini
- **Reliable** — Fallback provider if primary fails
- **Auditable** — Stores reasoning for every decision
- **Conservative** — When uncertain, asks a human

It's not trying to be clever. It's trying to be predictable and get out of the way so books can flow through the pipeline with minimal friction.

---

## Next Steps

1. **Implementation** — Build the classifier agent (~2-3 days)
2. **Integration** — Connect to pipeline orchestrator
3. **Testing** — Verify with sample books
4. **Calibration** — Adjust confidence threshold based on real results

Ready for implementation plan? See `docs/plans/2025-01-28-phase2-classifier-implementation.md` (to be created).

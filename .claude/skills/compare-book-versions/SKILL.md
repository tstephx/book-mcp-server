---
name: compare-book-versions
description: Compare a book's current state against its audit trail to see what changed after reingest/reprocess.
disable-model-invocation: true
---

# Compare Book Versions

Arguments: `<book_id>` — book UUID or partial title

## Steps

### 1. Resolve the book

Query for the book, supporting fuzzy title match:

```sql
-- Try exact ID first
SELECT id, title, author, word_count FROM books WHERE id = '<book_id>';

-- Fallback: fuzzy title match
SELECT id, title, author, word_count FROM books WHERE title LIKE '%<book_id>%';
```

If multiple matches, list them and ask the user to specify.

### 2. Get current chapter state

```sql
SELECT chapter_number, title, word_count
FROM chapters
WHERE book_id = '<resolved_id>'
ORDER BY chapter_number;
```

### 3. Get audit trail

```bash
agentic-pipeline audit --book-id <resolved_id> --last 20
```

Look for entries with action = `reprocess` or `reingest`. These contain `before_state` JSON with the pre-change metrics.

### 4. Query audit directly for before/after data

```sql
SELECT action, reason, before_state, after_state, performed_at
FROM approval_audit
WHERE book_id = '<resolved_id>'
  AND action IN ('reprocess', 'reingest', 'approve')
ORDER BY performed_at DESC
LIMIT 5;
```

The `before_state` column contains JSON with prior chapter counts and metrics.

### 5. Build comparison table

```
Book: <title> by <author>
ID:   <book_id>

                    Before          After           Delta
Chapters:           <n>             <n>             +/-<n>
Total Words:        <n>             <n>             +/-<n>
Avg Ch Words:       <n>             <n>             +/-<n>

Chapter Changes:
  Ch 1: "Old Title" (5000w) -> "New Title" (4800w)  [-200w]
  Ch 4: [NEW] "Added Chapter" (3000w)
  Ch 7: [REMOVED] "Deleted Chapter" (1200w)

Last reprocessed: <date>
Reason: <reason from audit>
```

### 6. Flag concerns

Highlight if:
- Chapter count changed by more than 20%
- Total word count changed by more than 20%
- Any chapter has 0 word count
- Any chapter title is empty or looks like an ISBN
- Recommend running `agentic-pipeline library-issues` if issues found

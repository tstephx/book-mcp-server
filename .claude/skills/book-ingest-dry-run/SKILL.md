---
name: book-ingest-dry-run
description: Preview what the pipeline would do with a book file without processing it. Shows classification, strategy, and duplicate check.
disable-model-invocation: true
---

# Book Ingest Dry Run

Arguments: `<file_path>` — path to an .epub or .pdf file

## Steps

### 1. Validate the file

```bash
# Check file exists and has valid extension
ls -la "<file_path>"
file "<file_path>"
```

Verify extension is `.epub` or `.pdf`. Report file size.

### 2. Compute content hash

```bash
shasum -a 256 "<file_path>" | awk '{print $1}'
```

### 3. Check for duplicates

Query the pipeline DB for existing entries with this hash:

```sql
SELECT pp.id, pp.current_state, pp.created_at, b.title, b.author
FROM processing_pipelines pp
LEFT JOIN books b ON b.id = pp.book_id
WHERE pp.content_hash = '<hash>';
```

If found, report: "This file was already processed on <date> as '<title>' (state: <state>)".

### 4. Run classifier

```bash
agentic-pipeline classify --text "$(head -c 5000 '<file_path>')"
```

If the file is binary (EPUB), extract text first:

```python
# Quick text extraction for classification preview
import ebooklib
from ebooklib import epub
book = epub.read_epub("<file_path>")
# Get first text item for classification
```

Report: predicted book type and confidence.

### 5. Show available strategies

```bash
agentic-pipeline strategies
```

Indicate which strategy would be selected based on the classification.

### 6. Check autonomy mode

```sql
SELECT mode, confidence_threshold FROM autonomy_config ORDER BY updated_at DESC LIMIT 1;
```

Report whether this book would auto-approve or require manual approval based on current autonomy settings and predicted confidence.

### 7. Synthesize report

```
Dry Run Report
==============
File:           <filename> (<size>)
Content Hash:   <hash>
Duplicate:      No / Yes (existing: <title>)
Classification: <type> (confidence: <score>)
Strategy:       <strategy_name>
Autonomy Mode:  <mode>
Auto-approve?:  Yes/No (threshold: <threshold>, predicted: <confidence>)
Next step:      agentic-pipeline process "<file_path>"
```

# Re-Chunking for Retrieval Precision — Design Spec

**Date:** 2026-07-15
**Status:** Approved design, pending spec review
**Prior art:** integrity-doctor spec (2026-07-15) established the backup/manifest and staged-consent conventions reused here.

## 1. Problem and Diagnosis

Chunks run far coarser than the chunker's documented ~500-word target
(library median ~1,373 words), so specific queries miss inside big books
(the "Mount Etna" problem: the answer exists but sits in a 2,500-word
chunk whose embedding averages away the detail).

Empirical diagnosis (2026-07-15, live DB: 289 books, 18,641 chunks):

- **5,780 of ~7,565 chunked chapters (76%) emit exactly ONE chunk**, with
  a median of 2,500 words — chapter passthrough, not chunking.
- Root cause is an **input-format mismatch, not a loop bug or doc drift**:
  `_split_paragraphs()` in `src/utils/chunker.py` splits on `\n\n+`, but
  extracted chapter markdown is wall-of-text (a sampled 6,050-char chapter
  contained 3 double-newlines). With one giant "paragraph," the break
  condition `current_words + para_words > target_words and current_words
  >= min_chunk_words` never sees a boundary to fire on.
- Reproduced: `chunk_chapter()` on that real chapter file returns 1 chunk
  of 2,509 words. The 2,500-word median matches `force_fallback`'s section
  size — fallback-extracted books are the worst-served.

## 2. Goals and Non-Goals

**Goals**

1. Chunk sizes near the 500-word target regardless of input paragraph
   formatting.
2. A before/after retrieval eval so the change is measured, not assumed.
3. Zero production risk: the live `chunks` table is untouched until the
   eval passes and the operator explicitly swaps.

**Non-goals (out of scope)**

- `chapters.embedding` (summary/chapter-level embeddings) — search reads
  only `chunks.embedding` via `load_chunk_embeddings()`; chapter
  embeddings are a separate surface.
- Upstream extraction changes in the `book-ingestion` repo — strictly
  dominated: cross-repo, requires re-extracting 289 books, and doesn't
  fix chapter files already on disk.
- Re-classification, re-summarization, or any pipeline state changes.

## 3. Chunking Strategy (Decision: hierarchical with sentence-window fallback)

Keep the existing paragraph pipeline in `src/utils/chunker.py`; add one
fallback rule.

**New function:** `sentence_windows(text, target_words=500,
overlap_sentences=2) -> list[str]` — splits text to sentences (same
regex family as `_split_at_sentences`), packs sentences greedily into
windows of ~`target_words`, and starts each subsequent window with the
last `overlap_sentences` sentences of the previous one.

**Integration point in `chunk_chapter()`:** after a segment is assembled
(paragraph-packed chunk, or the whole-text passthrough path at
`total_words <= target_words * 1.2`), any segment whose word count
exceeds **`2 * target_words` (1,000 words)** falls through to
`sentence_windows()`. Segments at or under the threshold keep today's
behavior byte-for-byte.

Properties:

- Well-formatted chapters (the ~24% that chunk correctly today) produce
  identical chunks.
- Wall-of-text chapters produce uniform ~500-word windows with
  2-sentence overlap instead of one slab.
- The existing `max_tokens=8000` post-process guard stays as the hard
  token ceiling (windows are far below it; guard retained for safety).
- Docstring updated to describe the two-level strategy truthfully.

Alternatives considered: (A) pure sentence windows everywhere — simpler
but discards paragraph alignment where it exists; (C) fix extraction
upstream — rejected as above. (B) chosen: best fit for a mixed corpus,
builds on existing code.

## 4. Retrieval Eval Harness

**New script:** `scripts/retrieval_eval.py` (project tooling, not a
library module). Two subcommands:

**`build-gold`** — samples ~60 chunks (deterministic seed, stratified
across ~30 books, skipping chunks under 150 words), asks an OpenAI chat
model (default `gpt-4o-mini`, `--model` overridable) to write one
question that only that passage answers, and writes
`eval/gold-queries.json`: a list of `{query, gold_chapter_id,
gold_book_id, source}` records. A second, hand-written file
`eval/gold-queries-manual.json` (same schema, `source: "manual"`) holds
~10 known-miss queries (Etna-class); the runner loads both.

**`run`** — embeds each query via `OpenAIEmbeddingGenerator.generate()`,
scores cosine similarity against a chunk-embedding matrix, aggregates to
**chapter level** (a hit = gold `chapter_id` appears in the top-k
distinct chapters), and reports **hit@5 and MRR** overall and per
source (auto vs manual). `--table chunks` (default, live) or
`--table chunks_staging` selects the matrix, so the identical query set
scores both schemes. Chapter-level scoring makes before/after
apples-to-apples: the gold unit doesn't depend on chunk boundaries.

Gold files are committed to the repo for reproducibility. Estimated
build cost: ~$0.10 of chat calls + pennies of query embeddings per run.

## 5. Staged Rollout — `agentic-pipeline rechunk`

New Click command in `agentic_pipeline/cli.py`, following the house
report-first/consent-flag convention (like `doctor` / `doctor --fix`).

**`rechunk` (default, report + stage):**

1. Create table `chunks_staging` (same schema and
   `UNIQUE(chapter_id, chunk_index)` constraint as `chunks`; dropped and
   recreated on each run).
2. For every chapter with a readable source file, read content via
   `read_chapter_content()`, run the updated `chunk_chapter()`, insert
   rows into `chunks_staging`.
3. Embed staged chunks in batches via
   `OpenAIEmbeddingGenerator.generate_batch()` (respects
   `MAX_BATCH_TOKENS`), writing `chunks_staging.embedding`.
   `--resume` skips staged rows that already have embeddings, so an
   interrupted run doesn't re-spend.
4. Run the eval twice (live matrix, staged matrix) and print a verdict
   table: chunk counts, median words, hit@5 / MRR (auto + manual), and
   PASS/FAIL against the acceptance gate.

Production `chunks` is never written in this mode. Cost and duration are
printed up front with a y/N confirmation before embedding spend.

**`rechunk --swap` (consent to mutate):**

1. Refuses unless a staged run exists and its recorded eval verdict is
   PASS (verdict persisted to `<db-dir>/rechunk/last-verdict.json` by
   step 4 — runtime state lives beside the DB, per the doctor
   convention; only the gold query files live in the repo).
2. Takes a doctor-style timestamped DB backup (reuse
   `create_backup()` from `agentic_pipeline/health/doctor.py`, keep-2
   rotation).
3. In one transaction: `DELETE FROM chunks`, copy all rows from
   `chunks_staging`, drop `chunks_staging`.
4. Post-swap verification: total count matches staging, zero NULL
   embeddings, `doctor` checks pass. Failure inside the transaction
   rolls back; failure after = restore from backup.

**Acceptance gate (binding):** auto-set hit@5 ≥ baseline AND auto-set
MRR ≥ baseline AND manual-set hit@5 strictly greater than baseline.
If staged loses, no swap: the ~$5 spent is the price of the measurement
and production never moved.

**Rollback:** restore the pre-swap backup (documented in the command's
output).

## 6. Cost and Runtime

- Corpus ≈ 26M words ≈ 34M tokens; with 2-sentence overlap on ~76% of
  content, re-embedding lands **~$4–6** at text-embedding-3-large
  pricing (revised up from the earlier $2–3 guess; overlap and finer
  chunks add tokens).
- Expected chunk count roughly doubles (~19k → ~40k). Runtime dominated
  by embedding API calls: on the order of 1–2 hours batched.
- Search memory: `load_chunk_embeddings()` loads the full matrix; ~40k ×
  3072 × 4 bytes ≈ 490 MB transient — acceptable on this machine, noted
  as a future optimization candidate (not in scope).

## 7. Testing (TDD, house rules)

All tests use a temp DB (`monkeypatch.setenv("AGENTIC_PIPELINE_DB", …)`);
never the live library DB.

**Chunker (`tests/test_chunker.py` additions):**
- Wall-of-text fixture (no `\n\n`, ~2,500 words) → N chunks, each within
  [300, 700] words, with verified sentence overlap between consecutive
  windows.
- Well-formatted fixture → output identical to current behavior
  (regression pin).
- Boundary: segment at exactly `2 * target_words` does NOT fall through;
  one word over does.
- Overlap: last `overlap_sentences` sentences of window i open window
  i+1.
- `max_tokens` guard still enforced on pathological input.
- Empty/whitespace text → `[]` (existing behavior preserved).

**Eval harness (`tests/test_retrieval_eval.py`):**
- `build-gold` sampling is deterministic under a fixed seed (mock the
  chat call).
- `run` scoring: hand-built 3-chunk matrix with known cosine ordering →
  expected hit@5/MRR values.
- Chapter-level aggregation: two chunks from the same chapter in top-k
  count as one chapter.

**Rechunk command (`tests/test_rechunk.py`):**
- Staging never mutates `chunks` (row hash of `chunks` identical before
  and after a stage run with mocked embeddings).
- `--swap` refused without a PASS verdict.
- Swap preserves `UNIQUE(chapter_id, chunk_index)`, count parity, zero
  NULL embeddings; simulated mid-transaction failure leaves `chunks`
  untouched.
- `--resume` does not re-embed already-embedded staged rows.

## 8. Risks

- **LLM-generated queries skew "answerable."** Mitigated by the manual
  known-miss set, which is also the strict-improvement arm of the gate.
- **Eval noise on n≈60.** The gate uses ≥ (not strict) on the auto set
  precisely because small-sample jitter shouldn't block a win on the
  manual set; if results are within noise on both, the honest verdict is
  FAIL (don't swap for nothing).
- **Chapters whose source files are gone** (5 known from the doctor
  manifest) can't be re-chunked: their existing live chunks are carried
  into staging unchanged, and the report lists them.

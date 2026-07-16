# Re-Chunking for Retrieval Precision — Design Spec

**Date:** 2026-07-15 (interview revisions same day)
**Status:** Interview complete — 9 decisions locked, pending final user review
**Prior art:** integrity-doctor spec (2026-07-15) established the backup/manifest and staged-consent conventions reused here.

## 0. Interview Decisions (binding)

1. **Eval gate = semantic-only; hybrid reported alongside.** The gate
   isolates the variable changed; hybrid numbers are printed for
   user-facing truth but do not gate.
2. **Worker keeps running; staleness handled by delta re-stage at swap.**
3. **Overlap crowding handled by existing per-chapter aggregation**
   (`best_chunk_per_chapter`); audit every search tool path once, add
   aggregation where missing — no MMR default change.
4. **Manual gold set drafted by Claude from session history, edited by
   the operator before the baseline run.**
5. **Hash-keyed resume is the default; no `--resume` flag; `--fresh`
   forces a clean slate.** No stale-embedding class can exist.
6. **Gold queries generated from chapter passages, not existing chunks**
   — zero old-scheme bias.
7. **Gold generation engine: `claude -p` (subscription billing, $0
   marginal, ~14 min for 60 calls).**
8. **One candidate config; iterate only on gate failure** (hash reuse
   makes reruns cheaper; no sweep machinery up front).
9. **`data_version` stamp + cache self-check** so a swap takes effect in
   running MCP servers immediately — no stale-serve window.

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

Verified environmental facts that shape this design:

- The FTS5 index is **`chapters_fts` — chapter-level**. Re-chunking does
  not touch it and the swap requires no FTS rebuild.
- **Nothing persists `chunks.id`** (bookmarks/progress reference
  chapters); the swap may freely regenerate chunk ids.
- The book-library MCP server caches the chunk-embedding matrix in
  memory (`LibraryCache.get_chunk_embeddings`) with **no DB-change
  detection** — hence decision 9.

## 2. Goals and Non-Goals

**Goals**

1. Chunk sizes near the 500-word target regardless of input paragraph
   formatting.
2. A before/after retrieval eval so the change is measured, not assumed.
3. Zero production risk: the live `chunks` table is untouched until the
   eval passes and the operator explicitly swaps.
4. A swap that takes effect everywhere immediately (decision 9).

**Non-goals (out of scope)**

- `chapters.embedding` (summary/chapter-level embeddings) — search reads
  only `chunks.embedding` via `load_chunk_embeddings()`.
- Upstream extraction changes in the `book-ingestion` repo — cross-repo,
  requires re-extracting 289 books, doesn't fix files already on disk.
- Re-classification, re-summarization, or any pipeline state changes.
- MMR default changes or any ranking-behavior change beyond the chunk
  scheme itself (decision 3): one variable per release.
- Parameter-sweep infrastructure (decision 8).

## 3. Chunking Strategy (hierarchical with sentence-window fallback)

Keep the existing paragraph pipeline in `src/utils/chunker.py`; add one
fallback rule.

**New function:** `sentence_windows(text, target_words=500,
overlap_sentences=2) -> list[str]` — splits text to sentences (same
regex family as `_split_at_sentences`), packs sentences greedily into
windows of ~`target_words`, and starts each subsequent window with the
last `overlap_sentences` sentences of the previous one. A final window
under `min_chunk_words` (100) merges into its predecessor rather than
standing alone.

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
  token ceiling.
- Docstring updated to describe the two-level strategy truthfully.

**Overlap crowding (decision 3):** adjacent windows share 2 sentences,
so two near-identical chunks from one chapter could crowd raw top-k.
Production search already aggregates via `best_chunk_per_chapter`
(`src/utils/chunk_loader.py`); the implementation audits every search
tool path (semantic, hybrid, related-content, etc.) and adds per-chapter
aggregation where any path lacks it. No new ranking machinery.

Alternatives considered: (A) pure sentence windows everywhere — simpler
but discards paragraph alignment where it exists; (C) fix extraction
upstream — rejected as above. (B) chosen: best fit for a mixed corpus,
builds on existing code.

## 4. Retrieval Eval Harness

**New script:** `scripts/retrieval_eval.py` (project tooling, not a
library module). Two subcommands.

**`build-gold`** (decisions 4, 6, 7):

- Samples ~60 **chapters** (deterministic seed, stratified across ~30
  books, skipping chapters under 300 words). For each, reads the chapter
  text via `read_chapter_content()` and takes a deterministic ~400-word
  passage (seeded offset) — the passage, not any chunk, is what the
  generator sees, so neither chunk scheme biases what gets asked.
- Generates one question per passage via **`claude -p`** (subprocess,
  fenced-JSON output parsed with the pattern proven in the Project D
  spike; ~14 min for 60 calls, $0 marginal on subscription billing).
- Writes `eval/gold-queries.json`: `{query, gold_chapter_id,
  gold_book_id, source: "auto"}` records.
- A second file `eval/gold-queries-manual.json` (same schema,
  `source: "manual"`) holds ~10 known-miss queries. Claude drafts these
  from session history (Etna/Rick Steves Sicily-class misses) with gold
  labels verified to exist in the library; the operator edits/strikes
  before the baseline run (decision 4).
- Both gold files are committed to the repo for reproducibility.

**`run`** (decision 1):

- Embeds each query via `OpenAIEmbeddingGenerator.generate()` (queries
  must match the search embedding model regardless of the gold engine).
- Scores two retrieval modes per query set:
  - **semantic** — cosine over the selected chunk matrix → top-k
    distinct chapters. **This is the gating number.**
  - **hybrid** — the production `hybrid_search` path (RRF over
    `chapters_fts` + semantic). **Reported, not gating** — chapter-level
    FTS is unchanged by this project and would dilute attribution.
- Aggregates to **chapter level** (a hit = gold `chapter_id` in top-k
  distinct chapters) and reports **hit@5 and MRR**, split by source
  (auto vs manual) and mode (semantic vs hybrid).
- `--table chunks` (default, live) or `--table chunks_staging` selects
  the semantic matrix, so the identical query set scores both schemes.

## 5. Staged Rollout — `agentic-pipeline rechunk`

New Click command in `agentic_pipeline/cli.py`, following the house
report-first/consent-flag convention (like `doctor` / `doctor --fix`).

**`rechunk` (default: stage + eval, idempotent, hash-keyed — decision 5):**

1. Ensure table `chunks_staging` exists (same schema and
   `UNIQUE(chapter_id, chunk_index)` constraint as `chunks`, plus a
   `content_hash` column). `--fresh` drops it first.
2. For every chapter with a readable source file, read content via
   `read_chapter_content()`, run the updated `chunk_chapter()`, and
   reconcile against existing staging **by `content_hash`** (reuse
   `compute_content_hash` from `src/utils/embedding_sync.py`):
   identical-hash rows keep their embeddings; new/changed rows are
   inserted for embedding; staged rows no longer produced are deleted.
   An interrupted run resumes by simply rerunning `rechunk`; a chunker
   code or param change automatically re-embeds exactly what changed.
   No stale-embedding state is representable.
3. Embed unembedded staged rows in batches via
   `OpenAIEmbeddingGenerator.generate_batch()` (respects
   `MAX_BATCH_TOKENS`). Cost and duration printed up front with a y/N
   confirmation before spend.
4. Record a live-snapshot marker (max `chunks.created_at` + row count)
   for the swap-time staleness check (decision 2).
5. Run the eval (live matrix, staged matrix) and print the verdict
   table: chunk counts, median words, semantic + hybrid hit@5/MRR
   (auto + manual), and PASS/FAIL against the gate. Persist the verdict
   to `<db-dir>/rechunk/last-verdict.json` (runtime state beside the
   DB, per the doctor convention).

Production `chunks` is never written in this mode. Chapters whose source
files are gone (5 known from the doctor manifest) carry their existing
live chunks into staging unchanged and are listed in the report.

**`rechunk --swap` (consent to mutate):**

1. Refuses unless the recorded verdict is PASS.
2. **Staleness check (decision 2):** diff live `chunks` against the
   stage-time marker. If new books landed (the launchd worker keeps
   running throughout), re-chunk + embed just those chapters into
   staging, then proceed. The delta doesn't re-run the eval — both
   schemes contain the new books identically.
3. Takes a doctor-style timestamped DB backup (reuse `create_backup()`
   from `agentic_pipeline/health/doctor.py`, keep-2 rotation).
4. In one transaction: `DELETE FROM chunks`, copy all rows from
   `chunks_staging`, drop `chunks_staging`, **bump `data_version`**
   (decision 9, below).
5. Post-swap verification: count parity with staging, zero NULL
   embeddings, `doctor` checks pass. Failure inside the transaction
   rolls back; failure after = restore from backup (instructions in the
   command output).

**Acceptance gate (binding, semantic mode):** auto-set hit@5 ≥ baseline
AND auto-set MRR ≥ baseline AND manual-set hit@5 strictly greater than
baseline. If staged loses, no swap; iterate on params only then
(decision 8 — hash-keyed staging makes the rerun cheaper).

## 6. Cache Coherence — `data_version` (decision 9)

The book-library MCP server holds the chunk matrix in `LibraryCache`
with no DB-change detection; without this, a swap silently doesn't take
effect in running servers — the "value nobody read" defect class.

- New migration (in `agentic_pipeline/db/migrations.py` `MIGRATIONS`
  list, per house rules): a one-row `library_meta` table with
  `data_version INTEGER` (seeded to 1).
- The swap transaction increments `data_version`.
- `LibraryCache.get_chunk_embeddings()` (and the chapter-embedding
  getter, same fix for `refresh_embeddings`) compares a stored
  `data_version` against one indexed `SELECT` per cache hit
  (microseconds) and self-invalidates on mismatch, reloading from the
  DB on the next call.
- Result: no stale-serve window, no human step to forget, and the same
  mechanism serves all future maintenance operations.

## 7. Cost and Runtime

- Corpus ≈ 26M words ≈ 34M tokens; with 2-sentence overlap on ~76% of
  content, re-embedding lands **~$4–6** at text-embedding-3-large
  pricing. Gold generation: $0 marginal (`claude -p`); query embeddings:
  pennies per eval run.
- Expected chunk count roughly doubles (~19k → ~40k). Runtime dominated
  by embedding API calls: on the order of 1–2 hours batched; gold build
  ~14 min.
- Search memory: full matrix ≈ 40k × 3072 × 4 bytes ≈ 490 MB transient —
  acceptable on this machine; future optimization candidate, not in
  scope.

## 8. Testing (TDD, house rules)

All tests use a temp DB (`monkeypatch.setenv("AGENTIC_PIPELINE_DB", …)`);
never the live library DB. All LLM and embedding calls mocked.

**Chunker (`tests/test_chunker.py` additions):**
- Wall-of-text fixture (no `\n\n`, ~2,500 words) → N chunks, each within
  [300, 700] words, with verified sentence overlap between consecutive
  windows.
- Well-formatted fixture → output identical to current behavior
  (regression pin).
- Boundary: segment at exactly `2 * target_words` does NOT fall through;
  one word over does.
- Overlap: last `overlap_sentences` sentences of window i open window
  i+1; final short window merges into predecessor (no chunk under
  `min_chunk_words`).
- `max_tokens` guard still enforced on pathological input.
- Empty/whitespace text → `[]` (existing behavior preserved).

**Eval harness (`tests/test_retrieval_eval.py`):**
- `build-gold` chapter sampling and passage offsets deterministic under
  a fixed seed (mock the `claude -p` subprocess).
- Malformed generator output (non-JSON, missing fields) skips the sample
  with a logged warning — never a crash or a fabricated record.
- `run` scoring: hand-built 3-chunk matrix with known cosine ordering →
  expected hit@5/MRR values, semantic and hybrid modes.
- Chapter-level aggregation: two chunks from the same chapter in top-k
  count as one chapter.

**Rechunk command (`tests/test_rechunk.py`):**
- Staging never mutates `chunks` (row hash of `chunks` identical before
  and after a stage run with mocked embeddings).
- Hash-keyed reconcile: rerun after interruption embeds only unembedded
  rows; a param change re-embeds exactly the changed chunks; orphaned
  staged rows are deleted; `--fresh` drops everything.
- `--swap` refused without a PASS verdict.
- Staleness delta: a book added between stage and swap gets staged and
  survives the swap.
- Swap preserves `UNIQUE(chapter_id, chunk_index)`, count parity, zero
  NULL embeddings, and bumps `data_version`; simulated mid-transaction
  failure leaves `chunks` AND `data_version` untouched.

**Cache coherence (`tests/test_cache.py` additions):**
- Cache hit with unchanged `data_version` → no reload.
- Bumped `data_version` → self-invalidation and reload on next call.

**Search-path audit:** each search tool path asserted (or made) to
aggregate per chapter before returning results.

## 9. Risks

- **LLM-generated queries skew "answerable."** Mitigated by the manual
  known-miss set, which is also the strict-improvement arm of the gate.
- **Eval noise on n≈60.** The gate uses ≥ (not strict) on the auto set
  precisely because small-sample jitter shouldn't block a win on the
  manual set; if results are within noise on both, the honest verdict is
  FAIL (don't swap for nothing).
- **Hybrid numbers may move less than semantic numbers** (chapter-level
  FTS already catches many gold queries). Expected and acceptable — the
  gate is semantic by design; hybrid is reported so the user-facing
  effect is known, not assumed.
- **`claude -p` flakiness across 60 calls.** Each sample is independent;
  failures skip-and-log, and `build-gold` can be rerun (deterministic
  sampling regenerates the same passages) until the set is full.
- **Chapters whose source files are gone** (5 known) can't be re-chunked:
  their live chunks are carried into staging unchanged and listed in the
  report.

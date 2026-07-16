#!/usr/bin/env python3
"""Retrieval eval CLI — gold-set builder and before/after scorer.

Usage:
    python scripts/retrieval_eval.py build-gold [--n 60] [--seed 42]
    python scripts/retrieval_eval.py run [--table chunks|chunks_staging]

Requires: BOOK_DB_PATH (defaults to the standard library.db location),
OPENAI_API_KEY for `run` (query embeddings), `claude` CLI for build-gold.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DB = Path.home() / "Library/Application Support/book-library/library.db"
EVAL_DIR = REPO_ROOT / "eval"
GOLD_AUTO = EVAL_DIR / "gold-queries.json"
GOLD_MANUAL = EVAL_DIR / "gold-queries-manual.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    bg = sub.add_parser("build-gold", help="Generate the auto gold set via claude -p")
    bg.add_argument("--n", type=int, default=60)
    bg.add_argument("--seed", type=int, default=42)

    run = sub.add_parser("run", help="Score a chunk table against the gold sets")
    run.add_argument("--table", choices=["chunks", "chunks_staging"], default="chunks")

    args = parser.parse_args()
    db_path = Path(os.environ.get("BOOK_DB_PATH", DEFAULT_DB))
    if not db_path.exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        return 1

    from src.utils.retrieval_eval import build_gold, run_eval

    if args.cmd == "build-gold":
        written = build_gold(db_path, GOLD_AUTO, n=args.n, seed=args.seed)
        print(f"{written} gold records -> {GOLD_AUTO}")
        return 0 if written else 1

    from src.utils.openai_embeddings import OpenAIEmbeddingGenerator

    report = run_eval(
        db_path,
        [GOLD_AUTO, GOLD_MANUAL],
        table=args.table,
        embedder=OpenAIEmbeddingGenerator(),
    )
    print(json.dumps({"table": args.table, "report": report}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

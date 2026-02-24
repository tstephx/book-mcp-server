#!/usr/bin/env python
"""Process all books in a directory through the agentic pipeline.

Usage:
    python process_all_books.py /path/to/ebooks
    python process_all_books.py              # uses EBOOKS_DIR env var
"""

import os
import sys
from pathlib import Path

from agentic_pipeline.config import OrchestratorConfig
from agentic_pipeline.orchestrator import Orchestrator


def main():
    if len(sys.argv) > 1:
        ebooks_dir = Path(sys.argv[1])
    elif os.environ.get("EBOOKS_DIR"):
        ebooks_dir = Path(os.environ["EBOOKS_DIR"])
    else:
        print("Usage: python process_all_books.py /path/to/ebooks")
        print("   Or: set EBOOKS_DIR environment variable")
        sys.exit(1)

    if not ebooks_dir.is_dir():
        print(f"Error: {ebooks_dir} is not a directory")
        sys.exit(1)

    epubs = list(ebooks_dir.rglob("*.epub"))
    pdfs = list(ebooks_dir.rglob("*.pdf"))
    books = epubs + pdfs

    if not books:
        print(f"No .epub or .pdf files found in {ebooks_dir}")
        sys.exit(0)

    config = OrchestratorConfig.from_env()
    orchestrator = Orchestrator(config)

    total = len(books)
    completed = 0
    failed = 0
    skipped = 0

    print(f"Processing {total} books from {ebooks_dir}...\n", flush=True)

    for i, book in enumerate(books, 1):
        name = book.name[:45] + "..." if len(book.name) > 45 else book.name
        print(f"[{i}/{total}] {name}", end=" ", flush=True)
        try:
            result = orchestrator.process_one(str(book))
            if result.get("skipped"):
                print("- Skipped", flush=True)
                skipped += 1
            elif result.get("state") == "complete":
                print(f"✓ {result.get('book_type')} ({result.get('confidence', 0):.0%})", flush=True)
                completed += 1
            else:
                print(f"⚠ {result.get('state')}", flush=True)
                failed += 1
        except Exception as e:
            print(f"✗ Error: {str(e)[:40]}", flush=True)
            failed += 1

    print(f"\n{'='*50}", flush=True)
    print(f"DONE! Completed: {completed}, Skipped: {skipped}, Failed: {failed}", flush=True)


if __name__ == "__main__":
    main()

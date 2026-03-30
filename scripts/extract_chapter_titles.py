#!/usr/bin/env python3
"""Extract real chapter titles from EPUB/PDF source files and update the corrections JSON.

Usage:
    python scripts/extract_chapter_titles.py [--dry-run] [--verbose]

Reads chapter-title-corrections.json, looks up each book's source_file in the DB,
extracts the TOC from the EPUB (NCX or NAV), and fills in corrected_title where possible.

Outputs: chapter-title-corrections.json (updated in place)
"""

import argparse
import json
import os
import sqlite3
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

DB_PATH = os.path.expanduser("~/Library/Application Support/book-library/library.db")
JSON_PATH = Path(__file__).parent.parent / "chapter-title-corrections.json"


def get_source_files(db_path: str, book_ids: list[str]) -> dict[str, str]:
    """Return {book_id: source_file} for books that have source files."""
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" for _ in book_ids)
    rows = conn.execute(
        f"SELECT id, source_file FROM books WHERE id IN ({placeholders})",
        book_ids,
    ).fetchall()
    conn.close()
    return {r["id"]: r["source_file"] for r in rows if r["source_file"]}


def extract_epub_toc(epub_path: str, verbose: bool = False) -> list[str]:
    """Extract ordered chapter titles from an EPUB's TOC (NCX or NAV).

    Returns a list of title strings in spine/TOC order.
    """
    try:
        with zipfile.ZipFile(epub_path, "r") as zf:
            # Try NCX first (EPUB 2), then NAV (EPUB 3)
            titles = _try_ncx(zf, verbose) or _try_nav(zf, verbose)
            if not titles:
                # Fallback: extract from OPF spine + HTML headings
                titles = _try_opf_headings(zf, verbose)
            return titles
    except (zipfile.BadZipFile, KeyError, ET.ParseError) as e:
        if verbose:
            print(f"  ERROR reading {epub_path}: {e}")
        return []


def _try_ncx(zf: zipfile.ZipFile, verbose: bool) -> list[str]:
    """Parse NCX table of contents."""
    ncx_files = [n for n in zf.namelist() if n.endswith(".ncx")]
    if not ncx_files:
        return []

    ncx_path = ncx_files[0]
    if verbose:
        print(f"  Found NCX: {ncx_path}")

    tree = ET.parse(zf.open(ncx_path))
    root = tree.getroot()

    # Handle namespace
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    titles = []
    nav_map = root.find(f"{ns}navMap")
    if nav_map is None:
        return []

    for nav_point in nav_map.findall(f"{ns}navPoint"):
        text_elem = nav_point.find(f"{ns}navLabel/{ns}text")
        if text_elem is not None and text_elem.text:
            title = text_elem.text.strip()
            if title:
                titles.append(title)

        # Also get nested navPoints (sub-chapters)
        for sub_point in nav_point.findall(f"{ns}navPoint"):
            sub_text = sub_point.find(f"{ns}navLabel/{ns}text")
            if sub_text is not None and sub_text.text:
                sub_title = sub_text.text.strip()
                if sub_title:
                    titles.append(sub_title)

    return titles


def _try_nav(zf: zipfile.ZipFile, verbose: bool) -> list[str]:
    """Parse EPUB 3 NAV document."""
    # Find NAV document from OPF
    opf_files = [n for n in zf.namelist() if n.endswith(".opf")]
    if not opf_files:
        return []

    tree = ET.parse(zf.open(opf_files[0]))
    root = tree.getroot()
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    # Find nav item in manifest
    manifest = root.find(f"{ns}manifest")
    if manifest is None:
        return []

    nav_href = None
    for item in manifest.findall(f"{ns}item"):
        props = item.get("properties", "")
        if "nav" in props:
            nav_href = item.get("href")
            break

    if not nav_href:
        return []

    # Resolve path relative to OPF
    opf_dir = os.path.dirname(opf_files[0])
    nav_path = os.path.join(opf_dir, nav_href) if opf_dir else nav_href
    # Normalize path separators
    nav_path = nav_path.replace("\\", "/")

    if nav_path not in zf.namelist():
        # Try without directory prefix
        for name in zf.namelist():
            if name.endswith(nav_href):
                nav_path = name
                break
        else:
            return []

    if verbose:
        print(f"  Found NAV: {nav_path}")

    content = zf.read(nav_path).decode("utf-8", errors="replace")

    # Parse as HTML/XHTML - handle namespace issues
    # Strip XHTML namespace for easier parsing
    content = content.replace('xmlns="http://www.w3.org/1999/xhtml"', "")
    content = content.replace("xmlns:epub=", "data-epub=")

    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return []

    titles = []
    # Find all <a> tags inside <nav> with epub:type="toc" or any <ol>/<li> structure
    for nav_elem in root.iter("nav"):
        for a_tag in nav_elem.iter("a"):
            text = "".join(a_tag.itertext()).strip()
            if text:
                titles.append(text)
        if titles:
            break

    # If no nav found, try all <a> in <ol> structures
    if not titles:
        for ol in root.iter("ol"):
            for a_tag in ol.iter("a"):
                text = "".join(a_tag.itertext()).strip()
                if text:
                    titles.append(text)

    return titles


def _try_opf_headings(zf: zipfile.ZipFile, verbose: bool) -> list[str]:
    """Last resort: read spine order from OPF and extract <h1>/<h2> from each HTML."""
    opf_files = [n for n in zf.namelist() if n.endswith(".opf")]
    if not opf_files:
        return []

    tree = ET.parse(zf.open(opf_files[0]))
    root = tree.getroot()
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    # Build id->href map from manifest
    manifest = root.find(f"{ns}manifest")
    if manifest is None:
        return []

    id_to_href = {}
    opf_dir = os.path.dirname(opf_files[0])
    for item in manifest.findall(f"{ns}item"):
        item_id = item.get("id")
        href = item.get("href")
        if item_id and href:
            full = os.path.join(opf_dir, href) if opf_dir else href
            id_to_href[item_id] = full.replace("\\", "/")

    # Get spine order
    spine = root.find(f"{ns}spine")
    if spine is None:
        return []

    titles = []
    for itemref in spine.findall(f"{ns}itemref"):
        idref = itemref.get("idref")
        href = id_to_href.get(idref)
        if not href or href not in zf.namelist():
            continue

        try:
            html = zf.read(href).decode("utf-8", errors="replace")
            # Quick regex-free heading extraction
            title = _extract_heading(html)
            if title:
                titles.append(title)
        except Exception:
            continue

    return titles


def _extract_heading(html: str) -> str:
    """Extract first h1 or h2 text from HTML string."""
    # Strip namespaces for simpler parsing
    html = html.replace('xmlns="http://www.w3.org/1999/xhtml"', "")
    try:
        root = ET.fromstring(html)
    except ET.ParseError:
        return ""

    for tag in ["h1", "h2", "h3"]:
        for elem in root.iter(tag):
            text = "".join(elem.itertext()).strip()
            if text and len(text) > 1:
                return text
    return ""


def extract_pdf_toc(pdf_path: str, verbose: bool = False) -> list[str]:
    """Extract TOC from PDF bookmarks using pikepdf if available."""
    try:
        import pikepdf
    except ImportError:
        if verbose:
            print(f"  SKIP PDF (pikepdf not installed): {pdf_path}")
        return []

    try:
        with pikepdf.open(pdf_path) as pdf:
            with pdf.open_outline() as outline:
                titles = []
                for item in outline.root:
                    if hasattr(item, "title") and item.title:
                        titles.append(item.title.strip())
                    # One level of children
                    if hasattr(item, "children"):
                        for child in item.children:
                            if hasattr(child, "title") and child.title:
                                titles.append(child.title.strip())
                return titles
    except Exception as e:
        if verbose:
            print(f"  ERROR reading PDF {pdf_path}: {e}")
        return []


def is_generic_title(title: str) -> bool:
    """Check if a title is generic like 'Chapter 1' or 'Section 5'."""
    import re

    return bool(re.match(r"^(Chapter|Section)\s+\d+", title, re.IGNORECASE))


def match_titles_to_chapters(toc_titles: list[str], chapters: list[dict]) -> list[dict]:
    """Match extracted TOC titles to existing chapter entries.

    Strategy:
    - Filter out generic TOC entries from the extracted list too
    - If count matches chapter count, map 1:1
    - If TOC has more entries (sub-sections), try to match by position
    """
    # Filter extracted titles - remove very short or generic ones
    real_titles = [t for t in toc_titles if len(t) > 1 and not is_generic_title(t)]

    if not real_titles:
        return chapters

    num_chapters = len(chapters)

    if len(real_titles) == num_chapters:
        # Perfect 1:1 match
        for ch, title in zip(chapters, real_titles):
            ch["corrected_title"] = title
    elif len(real_titles) >= num_chapters:
        # More TOC entries than chapters - take first N
        # (common when TOC includes sub-sections that were merged)
        for ch, title in zip(chapters, real_titles[:num_chapters]):
            ch["corrected_title"] = title
    else:
        # Fewer TOC entries than chapters - map what we can
        for i, title in enumerate(real_titles):
            if i < num_chapters:
                chapters[i]["corrected_title"] = title

    return chapters


def main():
    parser = argparse.ArgumentParser(description="Extract chapter titles from EPUBs/PDFs")
    parser.add_argument("--dry-run", action="store_true", help="Don't write output file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed progress")
    args = parser.parse_args()

    # Load corrections JSON
    with open(JSON_PATH) as f:
        data = json.load(f)

    books = data["books"]
    book_ids = [b["book_id"] for b in books]

    # Get source files from DB
    source_files = get_source_files(DB_PATH, book_ids)

    stats = {
        "total": len(books),
        "source_found": 0,
        "source_missing": 0,
        "epub_extracted": 0,
        "pdf_extracted": 0,
        "no_toc": 0,
        "chapters_fixed": 0,
    }

    for book in books:
        bid = book["book_id"]
        source = source_files.get(bid)

        if not source or not os.path.isfile(source):
            stats["source_missing"] += 1
            if args.verbose:
                print(f"SKIP (no source): {book['book_title'][:60]}")
            continue

        stats["source_found"] += 1
        ext = Path(source).suffix.lower()

        if args.verbose:
            print(f"\nProcessing: {book['book_title'][:60]}")
            print(f"  Source: {source}")

        if ext == ".epub":
            toc_titles = extract_epub_toc(source, args.verbose)
        elif ext == ".pdf":
            toc_titles = extract_pdf_toc(source, args.verbose)
        else:
            if args.verbose:
                print(f"  SKIP (unsupported format): {ext}")
            continue

        if not toc_titles:
            stats["no_toc"] += 1
            if args.verbose:
                print("  No TOC titles found")
            continue

        if ext == ".epub":
            stats["epub_extracted"] += 1
        else:
            stats["pdf_extracted"] += 1

        if args.verbose:
            print(f"  Found {len(toc_titles)} TOC entries for {len(book['chapters'])} chapters")
            for i, t in enumerate(toc_titles[:5]):
                print(f"    {i + 1}. {t}")
            if len(toc_titles) > 5:
                print(f"    ... and {len(toc_titles) - 5} more")

        # Match titles to chapters
        before_count = sum(1 for ch in book["chapters"] if ch["corrected_title"])
        book["chapters"] = match_titles_to_chapters(toc_titles, book["chapters"])
        after_count = sum(1 for ch in book["chapters"] if ch["corrected_title"])
        stats["chapters_fixed"] += after_count - before_count

    # Update stats in output
    data["instructions"]["extraction_stats"] = stats
    data["instructions"]["note"] = (
        "corrected_title was auto-filled from EPUB/PDF TOC where possible. "
        "Review and adjust as needed. Empty corrected_title = no extraction available."
    )

    if not args.dry_run:
        with open(JSON_PATH, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nUpdated: {JSON_PATH}")
    else:
        print("\nDRY RUN — no file written")

    print("\n--- Results ---")
    print(f"Total books:       {stats['total']}")
    print(f"Source found:      {stats['source_found']}")
    print(f"Source missing:    {stats['source_missing']}")
    print(f"EPUB TOC extracted:{stats['epub_extracted']}")
    print(f"PDF TOC extracted: {stats['pdf_extracted']}")
    print(f"No TOC found:      {stats['no_toc']}")
    print(f"Chapters fixed:    {stats['chapters_fixed']}")


if __name__ == "__main__":
    main()

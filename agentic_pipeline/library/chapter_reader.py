"""Shared utility for reading chapter content from disk."""

from pathlib import Path


def read_chapter_content(file_path: str, books_dir: Path) -> str:
    """Read chapter content from file, handling split chapters.

    Resolves relative paths against books_dir and handles split chapters
    stored as directories of numbered .md files.

    Args:
        file_path: Path to the chapter file (absolute or relative)
        books_dir: Base directory for book chapter files

    Returns:
        Chapter text content

    Raises:
        FileNotFoundError: If the chapter file cannot be found
    """
    path = Path(file_path)

    if not path.is_absolute():
        try:
            rel = path.relative_to("data/books")
            path = books_dir / rel
        except ValueError:
            path = books_dir / path

    if path.is_file():
        return path.read_text(encoding="utf-8")

    # Handle split chapters (directory with numbered .md parts)
    dir_path = path if path.is_dir() else path.with_suffix("")
    if dir_path.is_dir():
        parts = sorted(
            p for p in dir_path.glob("[0-9]*.md")
            if not p.name.startswith("_")
        )
        if not parts:
            parts = sorted(
                p for p in dir_path.glob("*.md")
                if not p.name.startswith("_")
            )
        if parts:
            return "\n\n".join(p.read_text(encoding="utf-8") for p in parts)

    raise FileNotFoundError(f"Chapter not found: {file_path}")

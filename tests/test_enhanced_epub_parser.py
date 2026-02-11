"""Tests for enhanced EPUB parser."""

import pytest
import tempfile
import zipfile
from pathlib import Path

from agentic_pipeline.converters.enhanced_epub_parser import (
    EnhancedEPUBParser,
    EPUBStructure,
    SplitPoint,
    SpineItem,
    parse_epub,
)


def create_minimal_epub(path: Path, with_anchors: bool = False) -> None:
    """Create a minimal valid EPUB for testing."""
    with zipfile.ZipFile(path, 'w') as zf:
        # mimetype (must be first, uncompressed)
        zf.writestr('mimetype', 'application/epub+zip')

        # container.xml
        zf.writestr('META-INF/container.xml', '''<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>''')

        # content.opf
        zf.writestr('OEBPS/content.opf', '''<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test Book</dc:title>
    <dc:creator>Test Author</dc:creator>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
    <item id="chapter1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
    <item id="chapter2" href="chapter2.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine toc="ncx">
    <itemref idref="chapter1"/>
    <itemref idref="chapter2"/>
  </spine>
</package>''')

        # toc.ncx with or without anchors
        if with_anchors:
            ncx_content = '''<?xml version="1.0"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <navMap>
    <navPoint id="np1" playOrder="1">
      <navLabel><text>Chapter 1: Introduction</text></navLabel>
      <content src="chapter1.xhtml"/>
    </navPoint>
    <navPoint id="np2" playOrder="2">
      <navLabel><text>Section 1.1: Getting Started</text></navLabel>
      <content src="chapter1.xhtml#section-1-1"/>
    </navPoint>
    <navPoint id="np3" playOrder="3">
      <navLabel><text>Section 1.2: Setup</text></navLabel>
      <content src="chapter1.xhtml#section-1-2"/>
    </navPoint>
    <navPoint id="np4" playOrder="4">
      <navLabel><text>Chapter 2: Advanced Topics</text></navLabel>
      <content src="chapter2.xhtml"/>
    </navPoint>
  </navMap>
</ncx>'''
        else:
            ncx_content = '''<?xml version="1.0"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <navMap>
    <navPoint id="np1" playOrder="1">
      <navLabel><text>Chapter 1: Introduction</text></navLabel>
      <content src="chapter1.xhtml"/>
    </navPoint>
    <navPoint id="np2" playOrder="2">
      <navLabel><text>Chapter 2: Advanced Topics</text></navLabel>
      <content src="chapter2.xhtml"/>
    </navPoint>
  </navMap>
</ncx>'''

        zf.writestr('OEBPS/toc.ncx', ncx_content)

        # chapter1.xhtml
        zf.writestr('OEBPS/chapter1.xhtml', '''<?xml version="1.0"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Chapter 1</title></head>
<body>
  <h1>Chapter 1: Introduction</h1>
  <p>This is the introduction to the book. It contains important information.</p>
  <h2 id="section-1-1">Section 1.1: Getting Started</h2>
  <p>This section covers getting started with the topic. We will explore various concepts.</p>
  <h2 id="section-1-2">Section 1.2: Setup</h2>
  <p>This section covers the setup process. Follow these steps carefully.</p>
</body>
</html>''')

        # chapter2.xhtml
        zf.writestr('OEBPS/chapter2.xhtml', '''<?xml version="1.0"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Chapter 2</title></head>
<body>
  <h1>Chapter 2: Advanced Topics</h1>
  <p>This chapter covers advanced topics in depth. We will discuss complex patterns.</p>
  <p>Additional content here to make the chapter longer and more substantial.</p>
</body>
</html>''')


@pytest.fixture
def minimal_epub(tmp_path):
    """Create a minimal EPUB without anchors."""
    epub_path = tmp_path / "test.epub"
    create_minimal_epub(epub_path, with_anchors=False)
    return epub_path


@pytest.fixture
def epub_with_anchors(tmp_path):
    """Create an EPUB with anchor-level TOC entries."""
    epub_path = tmp_path / "test_anchors.epub"
    create_minimal_epub(epub_path, with_anchors=True)
    return epub_path


class TestEnhancedEPUBParser:
    """Test the enhanced EPUB parser."""

    def test_parse_basic_structure(self, minimal_epub):
        """Test parsing basic EPUB structure."""
        structure = parse_epub(minimal_epub)

        assert structure.title == "Test Book"
        assert "Test Author" in structure.authors
        assert structure.spine_count == 2
        assert structure.has_ncx is True

    def test_spine_order(self, minimal_epub):
        """Test that spine items are in reading order."""
        structure = parse_epub(minimal_epub)

        assert len(structure.spine) == 2
        assert "chapter1" in structure.spine[0].href
        assert "chapter2" in structure.spine[1].href

    def test_spine_content(self, minimal_epub):
        """Test that spine items contain text content."""
        structure = parse_epub(minimal_epub)

        assert "Introduction" in structure.spine[0].text
        assert "Advanced Topics" in structure.spine[1].text

    def test_split_points_without_anchors(self, minimal_epub):
        """Test split points from file-level TOC entries."""
        structure = parse_epub(minimal_epub)

        # Should have 2 split points (one per chapter)
        assert len(structure.split_points) == 2

        titles = [sp.title for sp in structure.split_points]
        assert "Chapter 1: Introduction" in titles
        assert "Chapter 2: Advanced Topics" in titles

        # None should have anchors
        for sp in structure.split_points:
            assert sp.anchor is None
            assert sp.is_anchor_split is False

    def test_split_points_with_anchors(self, epub_with_anchors):
        """Test split points including anchor-level entries."""
        structure = parse_epub(epub_with_anchors)

        # Should have 4 split points (2 chapters + 2 sections with anchors)
        assert len(structure.split_points) == 4
        assert structure.anchor_count == 2

        # Check anchor entries
        anchor_splits = [sp for sp in structure.split_points if sp.is_anchor_split]
        assert len(anchor_splits) == 2

        anchor_titles = [sp.title for sp in anchor_splits]
        assert "Section 1.1: Getting Started" in anchor_titles
        assert "Section 1.2: Setup" in anchor_titles

    def test_split_point_full_href(self, epub_with_anchors):
        """Test full_href property includes anchor."""
        structure = parse_epub(epub_with_anchors)

        anchor_split = next(sp for sp in structure.split_points if sp.anchor == "section-1-1")
        assert "#section-1-1" in anchor_split.full_href

    def test_toc_titles_compatibility(self, minimal_epub):
        """Test flat toc_titles list for compatibility."""
        structure = parse_epub(minimal_epub)

        assert len(structure.toc_titles) == 2
        assert "Chapter 1: Introduction" in structure.toc_titles
        assert "Chapter 2: Advanced Topics" in structure.toc_titles

    def test_full_text_combined(self, minimal_epub):
        """Test that full_text contains all content."""
        structure = parse_epub(minimal_epub)

        assert "Introduction" in structure.full_text
        assert "Advanced Topics" in structure.full_text
        assert "Getting Started" in structure.full_text

    def test_skip_front_matter(self, tmp_path):
        """Test that front matter titles are skipped."""
        epub_path = tmp_path / "frontmatter.epub"

        with zipfile.ZipFile(epub_path, 'w') as zf:
            zf.writestr('mimetype', 'application/epub+zip')
            zf.writestr('META-INF/container.xml', '''<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>''')

            zf.writestr('content.opf', '''<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test</dc:title>
  </metadata>
  <manifest>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
    <item id="cover" href="cover.xhtml" media-type="application/xhtml+xml"/>
    <item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine toc="ncx">
    <itemref idref="cover"/>
    <itemref idref="ch1"/>
  </spine>
</package>''')

            zf.writestr('toc.ncx', '''<?xml version="1.0"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <navMap>
    <navPoint id="np1" playOrder="1">
      <navLabel><text>Cover</text></navLabel>
      <content src="cover.xhtml"/>
    </navPoint>
    <navPoint id="np2" playOrder="2">
      <navLabel><text>Table of Contents</text></navLabel>
      <content src="cover.xhtml"/>
    </navPoint>
    <navPoint id="np3" playOrder="3">
      <navLabel><text>Chapter 1</text></navLabel>
      <content src="ch1.xhtml"/>
    </navPoint>
  </navMap>
</ncx>''')

            zf.writestr('cover.xhtml', '<html><body>Cover page</body></html>')
            zf.writestr('ch1.xhtml', '<html><body><h1>Chapter 1</h1><p>Content here.</p></body></html>')

        structure = parse_epub(epub_path)

        # "Cover" and "Table of Contents" should be skipped
        assert "Cover" not in structure.toc_titles
        assert "Table of Contents" not in structure.toc_titles
        assert "Chapter 1" in structure.toc_titles

    def test_word_count_estimation(self, minimal_epub):
        """Test that split points have word count estimates."""
        structure = parse_epub(minimal_epub)

        for sp in structure.split_points:
            assert sp.word_count > 0

    def test_content_preview(self, minimal_epub):
        """Test that split points have content previews."""
        structure = parse_epub(minimal_epub)

        for sp in structure.split_points:
            assert len(sp.content_preview) > 0
            assert len(sp.content_preview) <= 500


class TestSplitPointDataclass:
    """Test SplitPoint dataclass."""

    def test_is_anchor_split_true(self):
        """Test is_anchor_split when anchor is present."""
        sp = SplitPoint(
            href="chapter.xhtml",
            anchor="section-1",
            title="Section 1",
            spine_index=0
        )
        assert sp.is_anchor_split is True

    def test_is_anchor_split_false(self):
        """Test is_anchor_split when no anchor."""
        sp = SplitPoint(
            href="chapter.xhtml",
            anchor=None,
            title="Chapter 1",
            spine_index=0
        )
        assert sp.is_anchor_split is False

    def test_full_href_with_anchor(self):
        """Test full_href includes anchor."""
        sp = SplitPoint(
            href="chapter.xhtml",
            anchor="section-1",
            title="Section 1",
            spine_index=0
        )
        assert sp.full_href == "chapter.xhtml#section-1"

    def test_full_href_without_anchor(self):
        """Test full_href without anchor."""
        sp = SplitPoint(
            href="chapter.xhtml",
            anchor=None,
            title="Chapter 1",
            spine_index=0
        )
        assert sp.full_href == "chapter.xhtml"

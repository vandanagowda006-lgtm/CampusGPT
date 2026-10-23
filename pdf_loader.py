
"""PDF loader that handles text PDFs, scanned PDFs,
and repeated headers/footers.

load_pdf(path) -> [{"page": 1, "text": "..."}]
Page numbers start from 1.
"""

import os
import re
from collections import Counter

from text_chunker import parse_heading

MIN_TEXT_CHARS = 40


def _clean(text):
    """Clean extracted PDF text."""
    text = (text or "").replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _ocr_page(page):
    """OCR one PyMuPDF page. Return an empty string if unavailable."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""

    cmd = os.getenv("TESSERACT_CMD")

    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd

    try:
        pix = page.get_pixmap(dpi=200, alpha=False)

        image = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples
        )

        return pytesseract.image_to_string(
            image,
            lang=os.getenv("OCR_LANG", "eng")
        )

    except Exception as exc:
        print(f"OCR failed on a page: {exc}")
        return ""


def _norm(line):
    """Normalize a line to help detect repeated headers and footers."""
    return re.sub(r"\d+", "#", line.strip().lower())


def _remove_repeated_headers_footers(pages):
    """Remove repeated edge lines without removing section headings."""

    if len(pages) < 4:
        return pages

    def edge_lines(text):
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        return lines[:2] + lines[-2:]

    counts = Counter()

    for page in pages:
        counts.update(
            _norm(line)
            for line in edge_lines(page["text"])
            if len(line) < 80
        )

    limit = max(3, int(len(pages) * 0.6))

    repeated = {
        key
        for key, count in counts.items()
        if count >= limit and key
    }

    if not repeated:
        return pages

    for page in pages:
        edge = set(edge_lines(page["text"]))
        kept = []

        for line in page["text"].splitlines():
            stripped = line.strip()

            if (
                stripped in edge
                and _norm(stripped) in repeated
                and not parse_heading(stripped)
            ):
                continue

            kept.append(line)

        page["text"] = "\n".join(kept).strip()

    return pages


def _load_with_pymupdf(path):
    """Extract text using PyMuPDF, with OCR as a fallback."""

    import fitz

    pages = []

    with fitz.open(path) as doc:
        for number, page in enumerate(doc, start=1):
            text = page.get_text("text", sort=True)

            if len(text.strip()) < MIN_TEXT_CHARS:
                ocr_text = _ocr_page(page)

                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text

            pages.append({
                "page": number,
                "text": _clean(text)
            })

    return pages


def _load_with_pypdf(path):
    """Extract text using pypdf when PyMuPDF is unavailable."""

    from pypdf import PdfReader

    reader = PdfReader(path)

    return [
        {
            "page": number,
            "text": _clean(page.extract_text() or "")
        }
        for number, page in enumerate(reader.pages, start=1)
    ]


def load_pdf(path):
    """Load a PDF and return cleaned text for every page."""

    try:
        pages = _load_with_pymupdf(path)

    except ImportError:
        print(
            "PyMuPDF not installed; "
            "falling back to pypdf (no OCR)."
        )

        pages = _load_with_pypdf(path)

    pages = _remove_repeated_headers_footers(pages)

    return pages


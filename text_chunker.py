"""Section-aware text chunker for CampusGPT."""

import re

_KINDS = (
    "day|week|module|unit|chapter|section|lesson|"
    "lecture|topic|session|part"
)

_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}

_ROMAN = {
    "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5,
    "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10,
}

_NUMBER = r"(?:\d{1,3}|[ivxlcdm]+|" + "|".join(_WORDS) + r")"

MAX_HEADING_LEN = 180
MAX_TITLE_LEN = 150

EMPTY_NOTE = "(No content found under this heading in the PDF.)"


# Recognizes headings such as:
# MODULE I
# Module - 1
# Module 1: Questions
# 1. MODULE 1
# Section 2
# Section 2: Establishment
_HEADING_PATTERNS = [
    re.compile(
        rf"^\s*(?:[•●▪*\-]+\s*)?(?:\d+\s*[.)]\s*)?"
        rf"(?P<kind>{_KINDS})\s*[-–—:#.]?\s*"
        rf"(?P<number>{_NUMBER})\b"
        rf"(?P<tail>\s*(?:(?:[:#–—-])\s*[^\n]{{0,150}}"
        rf"|\([^()\n]{{0,80}}\))?)\s*$",
        re.IGNORECASE,
    ),
]

# Recognizes numbered subsection headings such as:
# 2.1 Establishment of the Information Imparting Club for Women
# 2.2 Objectives
# 2.3 Activities Conducted
# 3.1 Impact
# 2.1: Establishment of the Club
_NUMERIC_SUBSECTION_PATTERN = re.compile(
    r"^\s*(?:[•●▪*\-]+\s*)?"
    r"(?P<major>\d{1,3})\s*\.\s*(?P<minor>\d{1,3})"
    r"(?:\s*[.)])?\s+"
    r"(?P<title>.+?)\s*$",
    re.IGNORECASE,
)

# Recognizes references in questions such as:
# Explain Section 2
# Explain Section 2.1
# Explain Module 1
_QUERY_SUBSECTION_RE = re.compile(
    r"\b(?P<kind>section|chapter|unit)\s*"
    r"(?P<major>\d{1,3})\s*\.\s*(?P<minor>\d{1,3})\b",
    re.IGNORECASE,
)

_QUERY_RE = re.compile(
    rf"\b(?P<kind>{_KINDS})\s*[-–—:#.]?\s*"
    rf"(?P<number>{_NUMBER})\b",
    re.IGNORECASE,
)


def _value(token):
    """Convert a numeric, Roman numeral, or number word to an integer."""

    token = token.lower()

    if token.isdigit():
        return int(token)

    if token in _ROMAN:
        return _ROMAN[token]

    return _WORDS.get(token)


def _normalize_heading_kind(kind):
    return kind.lower()


def parse_heading(line):
    """Return (display label, normalized key) for a recognized heading."""

    line = (line or "").strip()

    if not line or len(line) > MAX_HEADING_LEN:
        return None

    # First, recognize numbered subsection headings like 2.1, 2.2, 3.1.
    subsection_match = _NUMERIC_SUBSECTION_PATTERN.match(line)

    if subsection_match:
        major = int(subsection_match.group("major"))
        minor = int(subsection_match.group("minor"))
        title = subsection_match.group("title").strip()

        # Avoid treating an excessively long or empty title as a heading.
        if title and len(title) <= MAX_TITLE_LEN:
            # Numbered subsection headings are stored under the
            # normalized "section X.Y" key.
            key = f"section {major}.{minor}"
            return line, key

    # Next, recognize named headings like Module 1 or Section 2.
    for pattern in _HEADING_PATTERNS:
        match = pattern.match(line)

        if not match:
            continue

        kind = match.group("kind")
        number = match.group("number")
        tail = match.group("tail").strip()

        value = _value(number)

        if value is None:
            continue

        if len(tail) > MAX_TITLE_LEN:
            continue

        key = f"{_normalize_heading_kind(kind)} {value}"

        return line, key

    return None


def find_sections_in_query(question):
    """Find normalized section keys in a user's question."""

    question = question or ""
    keys = []

    # Detect subsection references first, so Section 2.1 does not
    # accidentally become Section 2.
    for match in _QUERY_SUBSECTION_RE.finditer(question):
        kind = match.group("kind").lower()
        major = int(match.group("major"))
        minor = int(match.group("minor"))

        key = f"{kind} {major}.{minor}"

        if key not in keys:
            keys.append(key)

    # Detect regular references such as Module 1 or Section 2.
    for match in _QUERY_RE.finditer(question):
        kind = match.group("kind").lower()
        number = match.group("number")
        value = _value(number)

        if value is None:
            continue

        # Avoid adding a parent section when the query already
        # identifies a numbered subsection, such as Section 2.1.
        if kind in ("section", "chapter", "unit"):
            after_number = question[match.end():]

            if re.match(r"\s*\.\s*\d{1,3}\b", after_number):
                continue

        key = f"{kind} {value}"

        if key not in keys:
            keys.append(key)

    return keys


def _split_long(text, chunk_size, step):
    """Split long text into overlapping chunks."""

    pieces = []
    start = 0

    while start < len(text):
        pieces.append(text[start:start + chunk_size])

        if start + chunk_size >= len(text):
            break

        start += step

    return pieces


def create_chunks(pages_data, chunk_size=1000, overlap=150):
    """Create chunks while preserving source and section metadata."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    step = chunk_size - overlap
    chunks = []

    state = {
        "source": None,
        "label": None,
        "key": None,
        "has_body": False,
        "page": None,
    }

    def add_chunk(source, page, body):
        label = state["label"]
        key = state["key"]

        chunk_text = f"[{label}]\n{body}" if label else body

        chunks.append({
            "source": source,
            "page": page,
            "text": chunk_text,
            "section": label,
            "section_key": key,
        })

    def close_section():
        if state["label"] and not state["has_body"]:
            add_chunk(
                state["source"],
                state["page"],
                EMPTY_NOTE,
            )

    for page_data in pages_data:
        source = page_data.get("source", "Unknown PDF")
        page = page_data["page"]
        text = (page_data.get("text") or "").strip()

        if source != state["source"]:
            close_section()

            state.update({
                "source": source,
                "label": None,
                "key": None,
                "has_body": False,
                "page": None,
            })

        if not text:
            continue

        buffer = ""

        def flush():
            nonlocal buffer

            if buffer:
                add_chunk(source, page, buffer)
                state["has_body"] = True
                buffer = ""

        raw_lines = text.splitlines()
        line_index = 0

        while line_index < len(raw_lines):
            line = raw_lines[line_index].strip()
            line_index += 1

            if not line:
                continue

            # Handle headings split across lines during PDF extraction,
            # such as "MODULE" followed by "1".
            if re.fullmatch(
                rf"(?:{_KINDS})\s*[-–—:#.]?",
                line,
                re.IGNORECASE,
            ):
                if line_index < len(raw_lines):
                    next_line = raw_lines[line_index].strip()

                    if re.fullmatch(
                        _NUMBER,
                        next_line,
                        re.IGNORECASE,
                    ):
                        line = f"{line} {next_line}"
                        line_index += 1

            heading = parse_heading(line)

            if heading:
                flush()
                close_section()

                state.update({
                    "label": heading[0],
                    "key": heading[1],
                    "has_body": False,
                    "page": page,
                })

                print(
                    f"Detected heading: {heading[0]!r} "
                    f"-> {heading[1]!r} "
                    f"({source}, page {page})"
                )

                continue

            if len(line) > chunk_size:
                flush()

                for piece in _split_long(
                    line,
                    chunk_size,
                    step,
                ):
                    add_chunk(source, page, piece)
                    state["has_body"] = True

                continue

            if not buffer:
                buffer = line

            elif len(buffer) + len(line) + 1 <= chunk_size:
                buffer += "\n" + line

            else:
                flush()
                buffer = line

        flush()

    close_section()

    print(f"Created {len(chunks)} chunks.")

    return chunks
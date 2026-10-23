"""
CampusGPT RAG pipeline.

Supports:
- General PDF question answering
- Hybrid retrieval using FAISS + BM25
- Section-aware questions (Module, Unit, Chapter, etc.)
- Follow-up questions using conversation history
- Source citations
- Safe handling of missing information

Public API:
    build_rag_pipeline(pdf_paths, original_names=None) -> (index, chunks)
    ask_question(index, chunks, question, history=None) -> str
"""

import os
import random
import re
import time
from pathlib import Path

from dotenv import load_dotenv

from embeddings import create_embeddings, create_embedding
from pdf_loader import load_pdf
from text_chunker import create_chunks, find_sections_in_query
from vector_store import create_vector_store, search_vector_store

load_dotenv()

ANSWER_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

NOT_FOUND = "I couldn't find that in the uploaded PDFs."

RETRYABLE_ERRORS = (
    "503",
    "UNAVAILABLE",
    "500",
    "INTERNAL",
    "429",
    "RESOURCE_EXHAUSTED",
    "502",
    "BAD_GATEWAY",
    "504",
    "DEADLINE_EXCEEDED",
)

_client = None


# ============================================================
# GEMINI CLIENT
# ============================================================

def _get_client():
    global _client

    if _client is None:
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY was not found. "
                "Please check your .env file."
            )

        _client = genai.Client(api_key=api_key)

    return _client


def _generate(prompt, temperature=0.1, retries=4):
    """Generate a response with retries for temporary API errors."""

    from google.genai import types

    for attempt in range(retries):
        try:
            response = _get_client().models.generate_content(
                model=ANSWER_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature
                ),
            )

            return (response.text or "").strip()

        except Exception as exc:
            error_text = str(exc).upper()

            retryable = any(
                error in error_text
                for error in RETRYABLE_ERRORS
            )

            if not retryable or attempt == retries - 1:
                if retryable:
                    raise RuntimeError(
                        "Gemini is temporarily busy or has reached "
                        "a request limit. Please try again later."
                    ) from exc

                raise

            wait_seconds = (
                2 ** (attempt + 1)
                + random.uniform(0, 0.5)
            )

            print(
                f"Gemini retry {attempt + 1}/{retries - 1}; "
                f"waiting {wait_seconds:.1f} seconds."
            )

            time.sleep(wait_seconds)


# ============================================================
# BUILD THE PDF SEARCH INDEX
# ============================================================

def build_rag_pipeline(pdf_paths=None, original_names=None):
    """
    Load PDF text, split it into chunks, generate embeddings,
    and create the hybrid search index.
    """

    if pdf_paths is None:
        raise ValueError("Please provide at least one PDF.")

    if isinstance(pdf_paths, (str, Path)):
        pdf_paths = [pdf_paths]
    else:
        pdf_paths = list(pdf_paths)

    if not pdf_paths:
        raise ValueError("Please upload at least one PDF.")

    if original_names is not None:
        original_names = list(original_names)

        if len(original_names) != len(pdf_paths):
            raise ValueError(
                "The number of original filenames must match "
                "the number of PDF paths."
            )

        display_names = {
            str(Path(path).resolve()): name
            for path, name in zip(pdf_paths, original_names)
        }
    else:
        display_names = {}

    all_pages = []

    for pdf_path in pdf_paths:
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {pdf_path}"
            )
        filename = display_names.get(
            str(pdf_path.resolve()),
            pdf_path.name,
        )

        pages = load_pdf(str(pdf_path))

        for page_data in pages:
            page_text = (
                page_data.get("text") or ""
            ).strip()

            if page_text:
                all_pages.append({
                    "source": filename,
                    "page": int(page_data["page"]),
                    "text": page_text,
                })

    if not all_pages:
        raise ValueError(
            "No readable text was found in the uploaded PDFs. "
            "For scanned PDFs, check that Tesseract OCR is installed."
        )

    chunks = create_chunks(all_pages)
    _annotate_paper_context(chunks)

    if not chunks:
        raise ValueError(
            "No text chunks were created from the uploaded PDFs."
        )

    print("\n========== CAMPUSGPT INDEX ==========")
    print("PDF pages with text:", len(all_pages))
    print("Total chunks:", len(chunks))

    sections = sorted({
        chunk.get("section_key")
        for chunk in chunks
        if chunk.get("section_key")
    })

    print("Detected sections:", sections)

    vectors = create_embeddings(
        [chunk["text"] for chunk in chunks],
        task_type="RETRIEVAL_DOCUMENT",
    )

    index, stored_chunks = create_vector_store(
        chunks,
        vectors,
    )

    print("Hybrid search index created successfully.")
    print("=====================================\n")

    return index, stored_chunks


# ============================================================
# CONVERSATION HISTORY
# ============================================================

def _history_text(history, limit=6):
    """Convert recent conversation messages into readable text."""

    lines = []

    for message in (history or [])[-limit:]:
        role = (
            "User"
            if message.get("role") == "user"
            else "Assistant"
        )

        content = str(
            message.get("content", "")
        )[:500]

        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def _recent_user_messages(history, limit=8):
    """Return recent user messages in chronological order."""
    messages = []
    for item in history or []:
        if item.get("role") == "user":
            text = str(item.get("content", "")).strip()
            if text:
                messages.append(text)
    return messages[-limit:]


def _standalone_question(question, history):
    """Resolve common follow-ups deterministically before asking Gemini."""
    question = question.strip()
    user_messages = _recent_user_messages(history)
    if not user_messages:
        return question

    latest_prior = user_messages[-1]
    previous_prior = user_messages[-2] if len(user_messages) >= 2 else ""
    lower = question.lower().strip()
    prior_lower = latest_prior.lower()

    # Outcome/sequence follow-ups often get rewritten into a request for the
    # previous explanation, which causes the chatbot to repeat itself. Preserve
    # the user's actual question and attach the latest topic only as context.
    asks_what_happened_next = bool(re.search(
        r"\b(?:what happened after|what happened next|what happened eventually|"
        r"what was the final outcome|what was the outcome|what was the result|"
        r"did (?:he|she|they|it) change|did anything change|was there any change|"
        r"what followed)\b",
        lower,
    ))
    if asks_what_happened_next:
        return (
            f"{question}\nContext from the immediately preceding user question: "
            f"{latest_prior}. Answer the latest outcome/sequence question directly; "
            "do not repeat the previous answer. If the PDF does not state what "
            "happened next, say that the outcome is not specified."
        )

    # Resolve follow-ups that refer to the previous answer's structure.
    # A request for a specific number of points must change the format, not
    # merely repeat the previous paragraph.
    count_request = re.search(
        r"\b(?:give|list|show|provide|write)\b.*?\b(one|two|three|four|five|\d+)\b.*?\b(points?|reasons?|advantages?|disadvantages?|examples?)\b|"
        r"\b(one|two|three|four|five|\d+)\s+(points?|reasons?|advantages?|disadvantages?|examples?)\b",
        lower,
    )
    if count_request and not re.search(r"\b(difference|compare|differentiate)\b", prior_lower):
        count = next((g for g in count_request.groups() if g and g in {"one", "two", "three", "four", "five"} or (g and g.isdigit())), None)
        topic = latest_prior.strip().rstrip("?.! ")
        return (
            f"Rewrite the answer about {topic} as exactly {count or 'the requested number of'} "
            "numbered points. Keep only information supported by the uploaded PDFs. "
            "Do not repeat the answer as a paragraph."
        )

    # “The second point” means the second item from the immediately preceding
    # answer, even if the previous user message was only a formatting request.
    if re.search(r"\b(second|2nd)\s+point\b", lower):
        topic = latest_prior.strip().rstrip("?.! ")
        if len(latest_prior.split()) <= 7 and previous_prior:
            topic = previous_prior.strip().rstrip("?.! ")
        style = "in simple English" if re.search(r"\b(simple english|simple words|easy language)\b", lower) else "clearly"
        return f"Explain only the second point from the answer about {topic} {style}. Do not explain other points."

    # A short follow-up asking for a number of points after a comparison
    # should continue the comparison, not jump back to an older topic.
    comparison = re.search(
        r"(?:difference|differentiate|compare)\s+between\s+(.+?)\s+and\s+(.+?)(?:[?.!,;]|\s+answer|\s+in\s+a\s+table|$)",
        latest_prior,
        flags=re.IGNORECASE,
    )
    if not comparison and previous_prior:
        comparison = re.search(
            r"(?:difference|differentiate|compare)\s+between\s+(.+?)\s+and\s+(.+?)(?:[?.!,;]|\s+answer|\s+in\s+a\s+table|$)",
            previous_prior,
            flags=re.IGNORECASE,
        )

    count_match = re.search(r"\b(one|two|three|four|five|\d+)\b", lower)
    asks_for_points = bool(re.search(r"\b(points?|differences?|reasons?|examples?)\b", lower))
    if comparison and count_match and asks_for_points:
        count = count_match.group(1)
        left = comparison.group(1).strip(" .?!,;:")
        right = comparison.group(2).strip(" .?!,;:")
        return (
            f"Give {count} differences between {left} and {right} from the uploaded PDFs "
            "as a numbered list, not a table."
        )

    # Style-only follow-ups should retain the most recent task's subject.
    # A style request is a follow-up only when it refers back to an earlier
    # answer ("it", "this", "same answer", etc.). A new topic such as
    # "Explain embeddings in simple English" must still be answered from the PDF.
    style_only = bool(re.search(
        r"\b(?:explain|rewrite|make|put|say)\b.*\b(?:it|this|that|same answer|previous answer|last answer)\b.*"
        r"\b(?:simple words?|simple english|easier words?|simplify|shorter|easy language)\b|"
        r"\b(?:same answer|it|this|that|previous answer|last answer)\b.*"
        r"\b(?:simple words?|simple english|easier words?|simplify|shorter|easy language)\b|"
        r"\b(?:simple words?|simple english|easier words?|simplify|shorter|easy language)\b.*"
        r"\b(?:it|this|that|previous answer|last answer)\b",
        lower,
    )) and not bool(re.search(
        r"\b(?:what|who|when|where|why|how many|define|difference between|compare|importance of|advantages of|disadvantages of|embeddings|machine learning|artificial intelligence|rag)\b",
        lower,
    ))

    if style_only:
        # If the previous user message was itself a short follow-up, retain
        # the preceding substantive topic instead of treating it as a topic.
        topic_message = latest_prior
        prior_is_short_followup = bool(re.search(
            r"\b(give|any|two|three|four|five|points?|same|it|this|that)\b",
            prior_lower,
        )) and len(latest_prior.split()) <= 7
        if prior_is_short_followup and previous_prior:
            topic_message = previous_prior
        topic = topic_message.strip().rstrip("?.! ")
        if re.search(r"\b(simple english|easier words?|simplify|make it simpler|in easy language)\b", lower):
            style = "in simple English"
        elif re.search(r"\b(shorter)\b", lower):
            style = "in a shorter form"
        else:
            style = "in simple words"
        # If the immediately preceding request asked for points, retain that
        # list format during simplification; do not inherit an older table.
        prior_was_points = bool(re.search(
            r"\b(points?|numbered list|bullet points?)\b",
            prior_lower,
        ))
        if prior_was_points:
            return (
                f"Rewrite the same answer about {topic} {style} "
                "as a numbered list, not a table."
            )
        return f"Explain the same answer about {topic} {style}."

    # Pronoun-only follow-ups should refer to the immediately preceding task.
    if re.fullmatch(r"(?:give )?(?:its|their|the|this|that|it)?\s*(?:importance|advantages|disadvantages|examples|meaning|definition)\s*[.!?]*", lower):
        topic = latest_prior.strip().rstrip("?.! ")
        detail = lower.strip(" .!?")
        return f"Explain the {detail} of the topic in: {topic}."

    # Use Gemini for less predictable follow-ups, with explicit priority given
    # to the most recent exchange so older topics do not override it.
    conversation = _history_text(history, limit=10)
    prompt = f"""
You are CampusGPT's follow-up question interpreter. Rewrite the latest user
message as one standalone search query for the uploaded PDFs. Do not answer it.

CONVERSATION HISTORY (oldest to newest):
{conversation}

LATEST USER MESSAGE:
{question}

Rules:
- Resolve pronouns and incomplete follow-ups using the MOST RECENT relevant task.
- A request for points immediately after a comparison means points about that
  comparison, not a topic from further back in the conversation.
- A request for the same answer in simple English changes the language only;
  preserve the topic and content of the immediately preceding answer.
- Preserve the latest requested number, format, and style.
- Preserve module, unit, chapter, paper, year, and filename identifiers.
- Do not invent details. Return one standalone search query only.
"""
    try:
        rewritten = _generate(prompt, temperature=0.0).strip().strip('"').strip("'")
        rewritten = re.sub(r"\s+", " ", rewritten).strip()
        if rewritten and len(rewritten) < 1000:
            return rewritten
    except Exception as exc:
        print("Question rewriting skipped:", exc)
    return question


def _requested_table_columns(question, history):
    """Get the table schema requested now; do not let an older table override it."""
    current = str(question or "").strip()
    lower = current.lower()

    # Explicit schemas in the current request always win over conversation history.
    # Handle the common wording used in demonstrations:
    # "exactly two columns: Concept and Explanation".
    explicit = re.search(
        r"(?is)(?:exactly\s+)?(?:two|2|three|3|four|4|five|5|six|6)\s+columns?"
        r"\s*(?:named|called)?\s*[:\-]\s*([^\n.!?]+)",
        current,
    )
    if explicit:
        raw = explicit.group(1).strip().strip("`'\"")
        names = re.split(r"\s*,\s*|\s+and\s+|\s*\|\s*", raw)
        names = [name.strip(" `\".,;:-") for name in names if name.strip(" `\".,;:-")]
        if 2 <= len(names) <= 6 and all(len(name) <= 35 for name in names):
            return names

    # Explicit named headers without a numeric column count.
    named = re.search(
        r"(?is)columns?\s*(?:named|called)\s*[:\-]\s*([^\n.!?]+)",
        current,
    )
    if named:
        names = re.split(r"\s*,\s*|\s+and\s+|\s*\|\s*", named.group(1).strip())
        names = [name.strip(" `\".,;:-") for name in names if name.strip(" `\".,;:-")]
        if 2 <= len(names) <= 6 and all(len(name) <= 35 for name in names):
            return names

    # Retain the specifically requested two-column schema when simplifying a
    # table, but only if the latest request refers to the previous table and
    # doesn't itself specify a new schema.
    if re.search(r"\b(previous|last|same)\s+table\b", lower) or re.search(
        r"\b(simplify|simpler|simple english|easier)\b", lower
    ):
        for message in reversed(history or []):
            if message.get("role") != "assistant":
                continue
            content = str(message.get("content", "")).strip()
            parsed = _parse_markdown_table(content)
            if parsed:
                return parsed[0]

    # Only apply this legacy schema when the CURRENT request explicitly names it.
    if re.search(r"(?is)(?:exactly\s+three\s+columns|three\s+columns).{0,180}"
                 r"aspect.{0,100}advantages.{0,100}limitations|"
                 r"aspect\s*[,/|and]+\s*advantages\s*[,/|and]+\s*limitations", lower):
        return ["Aspect", "Advantages", "Limitations"]
    return None


def _markdown_table_header_and_width(text):
    """Return (header cells, width) for the first Markdown table, if any."""
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if cells and not all(re.fullmatch(r"\s*:?-{3,}:?\s*", cell) for cell in cells):
                return cells, len(cells)
    return None, 0


def _has_exact_table_schema(text, columns):
    """Check exact Markdown headers and consistent cell counts in every row."""
    lines = [line.strip() for line in (text or "").splitlines()
             if line.strip().startswith("|") and line.strip().endswith("|")]
    if len(lines) < 2:
        return False
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    if len(header) != len(columns) or [x.casefold() for x in header] != [x.casefold() for x in columns]:
        return False
    separator = [cell.strip() for cell in lines[1].strip("|").split("|")]
    if len(separator) != len(columns) or not all(re.fullmatch(r":?-{3,}:?", cell) for cell in separator):
        return False
    return all(len(line.strip("|").split("|")) == len(columns) for line in lines[2:])


def _find_prior_table_with_schema(history, columns):
    """Return the newest previous assistant table that already has this schema."""
    for message in reversed(history or []):
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content", "")).strip()
        if _has_exact_table_schema(content, columns):
            return content
    return None


def _fallback_table_from_answer(answer):
    """Create a valid, conservative two-column table if Gemini fails formatting."""
    lines = [re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", line).strip()
             for line in (answer or "").splitlines()]
    items = [line for line in lines if line and not line.startswith("**Sources:")
             and not line.startswith("|") and not re.fullmatch(r"[-*_`# ]+", line)]
    if not items:
        items = ["Not enough information to format a table."]
    rows = ["| Topic | Details |", "| --- | --- |"]
    for item in items[:20]:
        safe = item.replace("|", "\\|")
        rows.append(f"| Information | {safe} |")
    return "\n".join(rows)


def _parse_markdown_table(text):
    """Parse a simple Markdown table into headers and rows; return None if invalid."""
    lines = [line.strip() for line in (text or "").splitlines()
             if line.strip().startswith("|") and line.strip().endswith("|")]
    if len(lines) < 3:
        return None

    def cells(line):
        return [part.strip().replace("\\|", "|") for part in line.strip().strip("|").split("|")]

    headers = cells(lines[0])
    separator = cells(lines[1])
    if not headers or len(separator) != len(headers):
        return None
    if not all(re.fullmatch(r":?-{3,}:?", item.replace(" ", "")) for item in separator):
        return None
    rows = []
    for line in lines[2:]:
        row = cells(line)
        if len(row) == len(headers):
            rows.append(row)
    return (headers, rows) if rows else None


def _table_to_bullets(text, requested_count=None):
    """Convert a Markdown table into meaningful bullets, keeping row concepts attached to their details."""
    parsed = _parse_markdown_table(text)
    if not parsed:
        return None

    headers, rows = parsed
    empty_values = {"not mentioned in the pdf", "n/a", "—", "-", "none", ""}

    usable_rows = []
    for row in rows:
        cells = [
            (headers[i], value.strip())
            for i, value in enumerate(row)
            if i < len(headers) and value.strip().lower() not in empty_values
        ]
        if not cells:
            continue
        # Ignore repeated header rows sometimes returned inside a table.
        if len(cells) == len(headers) and all(
            cells[i][1].casefold() == headers[i].casefold()
            for i in range(len(headers))
        ):
            continue
        usable_rows.append(cells)

    if not usable_rows:
        return None

    # Comparison tables commonly have columns such as Aspect | AI | ML.
    # Convert their definition row into one complete point per concept, then
    # combine relationship cells into one point when that meets the requested
    # count. This avoids detached or unlabeled ML definitions.
    first_header = headers[0].strip().casefold()
    comparison_headers = len(headers) >= 3 and first_header in {
        "aspect", "category", "feature", "criteria", "basis"
    }
    if comparison_headers and requested_count and requested_count > 0:
        comparison_facts = []
        relationship_facts = []
        for cells in usable_rows:
            aspect = cells[0][1].strip()
            aspect_lower = aspect.casefold()
            if any(word in aspect_lower for word in ("relationship", "similarity", "difference")):
                details = [f"**{header}:** {value}" for header, value in cells[1:] if value]
                if details:
                    relationship_facts.append(
                        f"**{aspect}:** " + "; ".join(details)
                    )
            else:
                for header, value in cells[1:]:
                    if value:
                        comparison_facts.append(f"**{header} — {aspect}:** {value}")
        comparison_facts.extend(relationship_facts)

        # Preserve explanatory notes that appear immediately before or after
        # a Markdown table (for example, "Note: All ML systems are part of AI...").
        # These notes are often where the source puts the overall relationship,
        # so dropping them loses a distinct fact during table-to-bullets conversion.
        table_lines = [
            line.strip() for line in (text or "").splitlines()
            if line.strip().startswith("|") and line.strip().endswith("|")
        ]
        note_facts = []
        for line in (text or "").splitlines():
            stripped = line.strip()
            if not stripped or (stripped.startswith("|") and stripped.endswith("|")):
                continue
            if re.match(r"(?i)^\*{0,2}note\s*:", stripped):
                note = re.sub(r"(?i)^\*{0,2}note\s*:\s*", "", stripped)
                note = re.sub(r"\*{1,2}$", "", note).strip()
                if note:
                    note_facts.append(f"**Relationship:** {note}")

        # Avoid repeating a note if the table already contains the same
        # relationship fact. Otherwise keep the note as its own available fact.
        for note in note_facts:
            note_text = re.sub(r"\W+", " ", note).casefold()
            if not any(note_text in re.sub(r"\W+", " ", fact).casefold()
                       or re.sub(r"\W+", " ", fact).casefold() in note_text
                       for fact in comparison_facts):
                comparison_facts.append(note)

        if comparison_facts:
            if len(comparison_facts) > requested_count:
                comparison_facts = comparison_facts[:requested_count - 1] + [
                    "; ".join(comparison_facts[requested_count - 1:])
                ]
            # If the source has fewer facts than requested, do not invent new ones.
            return "\n".join(f"- {fact}" for fact in comparison_facts)

    def row_as_bullet(cells):
        if len(cells) == 1:
            header, value = cells[0]
            return f"**{header}:** {value}"
        subject = cells[0][1]
        details = [f"**{header}:** {value}" for header, value in cells[1:]]
        return f"**{subject}** — " + "; ".join(details)

    facts = [row_as_bullet(cells) for cells in usable_rows]

    if requested_count and requested_count > 0:
        if len(facts) < requested_count:
            # Split only existing sentence content when the table has too few rows.
            # Keep the row's concept attached to its first sentence.
            expanded = []
            for cells in usable_rows:
                if len(cells) == 1:
                    expanded.append(row_as_bullet(cells))
                    continue
                subject = cells[0][1]
                details = [f"{header}: {value}" for header, value in cells[1:]]
                detail_text = " ".join(details)
                sentences = [
                    part.strip()
                    for part in re.split(r"(?<=[.!?])\s+", detail_text)
                    if part.strip()
                ]
                if len(sentences) > 1:
                    expanded.append(f"**{subject}:** {sentences[0]}")
                    expanded.extend(sentences[1:])
                else:
                    expanded.append(f"**{subject}** — " + "; ".join(details))
            if len(expanded) >= requested_count:
                facts = expanded

        if len(facts) > requested_count:
            # Preserve the requested count while retaining all available information.
            facts = facts[:requested_count - 1] + ["; ".join(facts[requested_count - 1:])]

    return "\n".join(f"- {fact}" for fact in facts)


def _requested_format_and_count(question):
    """Identify the desired OUTPUT format, rather than the format being converted."""
    text = (question or "").strip().lower()
    format_pattern = r"bullet points?|bullets?|numbered list|numbered points?|paragraph|table"
    # Prefer the format named after conversion words (e.g. 'table into bullets').
    target = re.search(
        rf"\b(?:into|as|to|in the form of)\s+(?:exactly\s+)?(?:\d+\s+|one\s+|two\s+|three\s+|four\s+|five\s+)?({format_pattern})\b",
        text,
    )
    if not target:
        # Keep the format in capture group 1 in both branches. Without this
        # wrapper, fallback matches like "table" have no group 1 and raise
        # IndexError("no such group") during follow-up formatting.
        target = re.search(rf"\b({format_pattern})\b", text)
    requested_format = target.group(1).lower() if target else "simple"

    number_words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    count_match = re.search(
        r"\b(?:exactly\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:bullet\s+points?|bullets?|numbered\s+points?|points?|items?)\b",
        text,
    )
    count = None
    if count_match:
        token = count_match.group(1)
        count = int(token) if token.isdigit() else number_words.get(token)
    return requested_format, count


def _format_only_followup(question, history):
    """Reformat the most recent assistant answer without retrieving a new topic."""
    lower = question.strip().lower()
    # Check bullet requests before table requests. A sentence like
    # "Convert that table into 3 bullet points" mentions both formats, but
    # the requested OUTPUT format is bullets.
    requested_format, requested_count = _requested_format_and_count(lower)
    format_request = re.search(
        r"\b(bullet points?|bullets?|numbered list|numbered points?|paragraph|table)\b",
        requested_format,
    ) if requested_format != "simple" else None
    simplify_previous = bool(re.search(
        r"\b(explain|rewrite|make|put|convert|turn)\b.*\b(previous|last|above|that|those|these|it|them|answer|response|bullet points?)\b.*\b(simple english|simple words|simpler|easier|shorter|easy to understand)\b|"
        r"\b(explain|rewrite|make)\b.*\b(simple english|simple words|simpler|easier|shorter)\b.*\b(previous|last|above|answer|response|it|them|those|these)\b|"
        r"\b(make|explain|rewrite)\b.*\b(previous answer|last answer|it|them|those|these|those bullet points|these bullet points)\b.*\b(simple|simpler|easier|shorter)\b",
        lower,
    ))
    is_format_only = (
        (bool(format_request) and bool(re.search(
            r"\b(now|make|put|convert|change|turn|rewrite|format|present|write)\b",
            lower,
        )) and not bool(re.search(
            r"\b(what|who|when|where|why|how|advantages of|disadvantages of|define|explain the)\b",
            lower,
        )))
        or simplify_previous
        or bool(re.search(r"\b(make|rewrite|explain)\b.*\b(those|these|the)\s+bullet points?\b.*\b(simpler|simple|easier|shorter)\b", lower))
    )
    if not is_format_only:
        return None

    last_answer = None
    for message in reversed(history or []):
        if message.get("role") == "assistant":
            content = str(message.get("content", "")).strip()
            if content:
                last_answer = content
                break
    if not last_answer:
        return None

    # Keep the source list intact and transform only the answer body.
    source_match = re.search(
        r"(?is)\n+\*\*Sources?:\*\*.*$",
        last_answer,
    )
    source_block = source_match.group(0).strip() if source_match else ""
    answer_body = last_answer[:source_match.start()].strip() if source_match else last_answer
    if not answer_body:
        return None

    requested_format = format_request.group(1).lower() if format_request else "simple"

    # A missing-information answer must remain a missing-information answer
    # during style/format changes; never turn it into unrelated PDF content.
    missing_answer = bool(re.fullmatch(
        r"(?is)(?:not mentioned in the pdf|i couldn't find (?:that|the information|sufficiently relevant text) in the uploaded pdfs?\.?|the (?:provided )?document (?:does not contain|lacks) this information\.?)[.!\s]*",
        answer_body.strip(),
    ))
    if missing_answer:
        missing_text = "Not mentioned in the PDF."
        if requested_format.startswith("bullet"):
            return "- Not mentioned in the PDF." + ("\n\n" + source_block if source_block else "")
        if requested_format.startswith("table"):
            columns = _requested_table_columns(question, history) or ["Aspect", "Advantages", "Limitations"]
            header = "| " + " | ".join(columns) + " |"
            separator = "| " + " | ".join(["---"] * len(columns)) + " |"
            row = ["Information"] + [missing_text] * (len(columns) - 1)
            table = header + "\n" + separator + "\n| " + " | ".join(row) + " |"
            return table + ("\n\n" + source_block if source_block else "")
        if requested_format == "simple":
            return "The PDF does not contain this information." + ("\n\n" + source_block if source_block else "")

    if requested_format == "simple":
        prompt = f"""Rewrite the previous answer in simple, clear English. Preserve its exact meaning and every important fact. Do not answer a new question, do not search the PDF for a different answer, and do not add facts. If the previous answer says information is missing, keep that meaning. Return only the rewritten answer.\n\nPREVIOUS ANSWER:\n{answer_body}"""
        try:
            simple_answer = _generate(prompt, temperature=0.0).strip()
        except Exception:
            simple_answer = answer_body
        return (simple_answer or answer_body) + ("\n\n" + source_block if source_block else "")

    # A table-to-bullets conversion is a structural operation, not a new
    # generation task. Convert deterministically so Gemini cannot return the
    # original table unchanged or invent replacement wording.
    if requested_format.startswith("bullet"):
        converted = _table_to_bullets(answer_body, requested_count=requested_count)
        if not converted:
            # If the prior answer has a malformed Markdown table, salvage its
            # non-separator rows into meaningful bullets instead of asking the
            # model to invent placeholders such as "Point one".
            raw_rows = []
            for line in answer_body.splitlines():
                stripped = line.strip()
                if not (stripped.startswith("|") and stripped.endswith("|")):
                    continue
                cells = [cell.strip() for cell in stripped.strip("|").split("|")]
                if cells and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
                    continue
                if cells and not any(re.search(r"\b(concept|aspect|topic)\b", c, re.I) for c in cells):
                    raw_rows.append(" — ".join(c for c in cells if c))
            if raw_rows:
                if requested_count and len(raw_rows) > requested_count:
                    raw_rows = raw_rows[:requested_count - 1] + ["; ".join(raw_rows[requested_count - 1:])]
                converted = "\n".join(f"- {row}" for row in raw_rows)
        if converted:
            converted = _enforce_bullet_count(converted, requested_count)
            return converted + ("\n\n" + source_block if source_block else "")
    wants_simple_english = bool(re.search(
        r"\b(simple english|simpler words|easy english|simple words|easier to understand|make it easier|simplify|simpler language)\b",
        lower,
    ))

    table_columns = _requested_table_columns(question, history)

    # If the last answer is already a valid table with the requested schema,
    # don't call the model and risk changing headers or adding duplicate rows.
    if (requested_format.startswith("table") and table_columns
            and _has_exact_table_schema(answer_body, table_columns)
            and not wants_simple_english):
        return answer_body + ("\n\n" + source_block if source_block else "")

    if requested_format.startswith("table"):
        if table_columns:
            column_list = " | ".join(table_columns)
            format_instruction = f"""
Convert the immediately preceding answer into a Markdown table with EXACTLY these columns, in this order:
{column_list}

STRICT RULES:
- Header must be exactly: | {column_list} |
- Every row must have exactly {len(table_columns)} cells.
- Never rename, remove, reorder, or add columns.
- Use only information in the immediately preceding answer. Do not add facts or examples.
- Use simple English, concise cells, and remove duplicate ideas.
- If the request asks to make the table easier to understand, rewrite each cell in simpler words; do not return the original wording unchanged.
- Use real <br> tags for line breaks inside cells, not escaped <br> text.
- Do not add an extra row just to say information is missing. Use 'Not mentioned in the PDF' only for a requested cell whose information is genuinely absent from the provided answer/source.
- Return only the Markdown table. Do not add a source section here; existing source labels are restored by the application.
"""
        else:
            format_instruction = (
                "Convert the immediately preceding answer into a real Markdown table. "
                "Use a header row, a separator row, and at least one data row. "
                "If the recent user history explicitly requested column names, preserve "
                "those exact names and order. Otherwise choose concise, useful headers "
                "based on the answer. Use only information already in the immediately "
                "preceding answer; do not add facts or examples. Use simple English and "
                "short, non-repetitive cells. Do not add an artificial 'Additional Aspects' "
                "row. Return only the table."
            )
    elif requested_format.startswith("paragraph"):
        format_instruction = (
            "Rewrite the answer as one paragraph, preserving only the information "
            "in the answer. Do not add facts."
        )
    elif requested_format.startswith("bullet"):
        format_instruction = (
            "Rewrite the answer as bullet points. If the user requests a specific "
            "number of bullets, produce exactly that number. When the source answer "
            "contains only one idea, split that same idea into distinct, non-redundant "
            "aspects without inventing new facts. Do not repeat the same point just "
            "to reach the requested count. Preserve the meaning and do not add facts."
        )
    else:
        format_instruction = (
            "Rewrite the answer as a numbered list, preserving only the information "
            "in the answer. Do not add facts."
        )

    if requested_count is not None:
        format_instruction += (
            f" The output must contain exactly {requested_count} bullet/list points. "
            "Count the points before responding."
        )
    if wants_simple_english:
        format_instruction += (
            " Use simple English and short, clear wording. For a table, actively rewrite "
            "each cell in easier words instead of copying the original wording unchanged. "
            "Preserve every distinct fact, keep the exact requested headers, and do not add "
            "new information. In Markdown table cells, use <br> for line breaks without "
            "escaping the angle brackets."
        )

    prompt = f"""You are reformatting the user's most recent answer in CampusGPT.
Do not answer a new question and do not use older conversation turns.
{format_instruction}
Return only the reformatted answer, without citations or a source section.

MOST RECENT ANSWER:
{answer_body}
"""
    try:
        reformatted = _generate(prompt, temperature=0.0).strip()
    except Exception as exc:
        print("Format-only follow-up skipped:", exc)
        return None

    if not reformatted:
        return None

    if requested_format.startswith("bullet") and requested_count:
        reformatted = _enforce_bullet_count(reformatted, requested_count)
        # If Gemini returned a table despite a bullet request, convert the table
        # rather than showing the wrong output format.
        if _parse_markdown_table(reformatted):
            converted = _table_to_bullets(reformatted, requested_count=requested_count)
            if converted:
                reformatted = _enforce_bullet_count(converted, requested_count)

    # Validate table output deterministically. Gemini may return bullets or prose
    # even when a table was requested, so repair once and use a safe fallback.
    if requested_format.startswith("table") and not table_columns:
        header, width = _markdown_table_header_and_width(reformatted)
        has_separator = any(
            line.strip().startswith("|") and line.strip().endswith("|")
            and re.search(r"\|\s*:?-{3,}:?\s*\|", line)
            for line in reformatted.splitlines()
        )
        if not header or width < 2 or not has_separator:
            strict_prompt = f"""Convert the answer below into a valid Markdown table.
Return only a table with a header row, a separator row, and data rows.
Use concise headers appropriate to the content. Keep all facts grounded in the
answer; do not invent information. Do not add an artificial 'Additional Aspects' row.

ANSWER:
{answer_body}
"""
            try:
                repaired = _generate(strict_prompt, temperature=0.0).strip()
                repaired_header, repaired_width = _markdown_table_header_and_width(repaired)
                repaired_separator = any(
                    line.strip().startswith("|") and line.strip().endswith("|")
                    and re.search(r"\|\s*:?-{3,}:?\s*\|", line)
                    for line in repaired.splitlines()
                )
                if repaired_header and repaired_width >= 2 and repaired_separator:
                    reformatted = repaired
                else:
                    reformatted = _fallback_table_from_answer(answer_body)
            except Exception as exc:
                print("Generic table repair skipped:", exc)
                reformatted = _fallback_table_from_answer(answer_body)

    # Validate exact requested table headers after generation. Gemini can ignore
    # schema instructions, so retry once with a strict repair prompt.
    if requested_format.startswith("table") and table_columns:
        if not _has_exact_table_schema(reformatted, table_columns):
            strict_prompt = f"""Repair the Markdown table below.
This is a formatting repair, not a new question.
The table MUST have exactly {len(table_columns)} columns with these exact headers
in this exact order: {" | ".join(table_columns)}.
The header row MUST be:
| {" | ".join(table_columns)} |
Every body row must have exactly {len(table_columns)} cells.
Do not rename or add headers. Preserve supported facts, avoid repetition, and
write 'Not mentioned in the PDF' where the source answer does not provide the
required information. Return only the corrected Markdown table.

TABLE TO REPAIR:
{reformatted}
"""
            try:
                repaired = _generate(strict_prompt, temperature=0.0).strip()
                if repaired and _has_exact_table_schema(repaired, table_columns):
                    reformatted = repaired
                else:
                    # If the model fails twice, preserve a prior answer that
                    # already has the exact requested schema rather than show
                    # a malformed table with changed headers.
                    previous_table = _find_prior_table_with_schema(history, table_columns)
                    if previous_table:
                        reformatted = previous_table
                    else:
                        cells = " | ".join(table_columns)
                        separator = " | ".join(["---"] * len(table_columns))
                        missing = " | ".join(["Not mentioned in the PDF"] * len(table_columns))
                        reformatted = f"| {cells} |\n| {separator} |\n| {missing} |"
            except Exception as exc:
                print("Strict table repair skipped:", exc)
                previous_table = _find_prior_table_with_schema(history, table_columns)
                if previous_table:
                    reformatted = previous_table
                else:
                    cells = " | ".join(table_columns)
                    separator = " | ".join(["---"] * len(table_columns))
                    missing = " | ".join(["Not mentioned in the PDF"] * len(table_columns))
                    reformatted = f"| {cells} |\n| {separator} |\n| {missing} |"

    # Gemini sometimes escapes HTML line breaks inside Markdown cells, making
    # them display as literal text. Normalize only this common formatting error.
    reformatted = re.sub(r"\\<br\s*/?\s*>", "<br>", reformatted, flags=re.IGNORECASE)
    return reformatted + ("\n\n" + source_block if source_block else "")


# ============================================================
# RETRIEVAL HELPERS
# ============================================================

def _expand_neighbours(chunks, chunk_ids, top_n=3):
    """
    Include adjacent chunks from the same PDF when useful.
    This helps when information spans chunk boundaries.
    """

    selected = set(chunk_ids)

    for chunk_id in chunk_ids[:top_n]:
        current = chunks[chunk_id]
        current_source = current.get("source")

        for neighbour_id in (
            chunk_id - 1,
            chunk_id + 1,
        ):
            if not 0 <= neighbour_id < len(chunks):
                continue

            neighbour = chunks[neighbour_id]

            if neighbour.get("source") == current_source:
                selected.add(neighbour_id)

    return sorted(selected)


# ============================================================
# PAPER-AWARE RETRIEVAL
# ============================================================

_PAPER_HEADING_RE = re.compile(
    r"(?i)\b(?:model\s+)?question\s+paper\b[^\n]{0,140}"
)


def _normalize_paper_title(value):
    """Normalize paper titles so punctuation and Roman numerals do not matter."""
    text = (value or "").lower()
    text = re.sub(r"\bpaper\s*[-:#.]?\s*i\b", "paper 1", text)
    text = re.sub(r"\bpaper\s*[-:#.]?\s*ii\b", "paper 2", text)
    text = re.sub(r"\bpaper\s*[-:#.]?\s*iii\b", "paper 3", text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _extract_requested_paper_title(question):
    """Extract an explicit question-paper title from the user's request."""
    text = str(question or "").strip()
    match = _PAPER_HEADING_RE.search(text)
    if not match:
        return None

    title = match.group(0).strip(" \t:;,.!?-–—")
    # Do not accidentally include the requested module after the paper title.
    title = re.split(
        r"(?i)\b(?:in|for|within|under)\s+"
        r"(?:module|unit|chapter|section|part|topic)\b",
        title,
        maxsplit=1,
    )[0].strip(" \t:;,.!?-–—")

    # Require a paper number/Roman numeral or a year/effectivity identifier.
    # This avoids treating generic requests such as "explain question paper
    # format" as requests for a specific named paper.
    has_identifier = bool(
        re.search(r"(?i)\bpaper\s*[-:#.]?\s*(?:[ivxlcdm]+|\d+)\b", title)
        or re.search(r"(?i)\b(?:20\d{2}|effect\s+from|effective\s+from)\b", title)
    )
    if not has_identifier:
        return None
    return title or None


def _annotate_paper_context(chunks):
    """
    Track question-paper headings through each PDF.
    Each chunk receives the most recently detected paper title for its source.
    """
    active_source = None
    active_title = None

    for chunk in chunks:
        source = chunk.get("source", "Unknown PDF")
        if source != active_source:
            active_source = source
            active_title = None

        text = chunk.get("text") or ""
        lines = text.splitlines()
        detected_title = None

        for line_index, line in enumerate(lines):
            # Ignore the synthetic section label inserted by text_chunker.
            line = re.sub(r"^\s*\[[^\]]+\]\s*$", "", line).strip()
            if not line:
                continue

            match = _PAPER_HEADING_RE.search(line)
            if not match:
                continue

            detected_title = match.group(0).strip(" \t:;,.!?-–—")
            # Many PDFs put the year/effectivity note on the next line.
            if line_index + 1 < len(lines):
                next_line = lines[line_index + 1].strip()
                if re.search(r"(?i)\b(effect\s+from|effective\s+from|20\d{2})\b", next_line):
                    detected_title += " " + next_line
            break

        if detected_title:
            active_title = detected_title

        chunk["paper_title"] = active_title


def _paper_title_matches(candidate_title, requested_title):
    """Return True only when the requested identifying title matches a chunk title."""
    if not candidate_title or not requested_title:
        return False

    candidate = _normalize_paper_title(candidate_title)
    requested = _normalize_paper_title(requested_title)
    if not candidate or not requested:
        return False

    # If the request includes an effectivity phrase or year, require those
    # details to be present too; a similarly numbered paper from another year
    # must not be treated as the requested paper.
    requested_years = re.findall(r"20\d{2}", requested)
    if requested_years and not all(year in candidate for year in requested_years):
        return False
    if "effect from" in requested and "effect from" not in candidate:
        return False
    if "effective from" in requested and "effective from" not in candidate:
        return False

    # Full normalized title match is safest. Allow the candidate to contain
    # extra descriptors, but do not accept a partial title when the request
    # specifies a year/effectivity.
    if requested in candidate:
        return True
    return not requested_years and "effect from" not in requested and "effective from" not in requested and candidate in requested


def _paper_matching_chunk_ids(chunks, requested_title):
    """Return chunk IDs belonging to the requested paper, never all papers by default."""
    _annotate_paper_context(chunks)
    return {
        i for i, chunk in enumerate(chunks)
        if _paper_title_matches(chunk.get("paper_title"), requested_title)
    }


def _is_extraction_request(question):
    """
    Identify requests to list or extract multiple items.

    This does not assume the uploaded document is a model paper.
    """

    text = question.lower()

    extraction_phrases = (
        "all questions",
        "all the questions",
        "give questions",
        "list questions",
        "extract questions",
        "all topics",
        "all points",
        "list all",
        "give all",
        "every question",
        "complete list",
        "entire list",
        "all steps",
        "questions from",
        "questions in",
        "questions of",
        "multiple-choice questions",
        "mcqs",
        "short-answer questions",
        "short answer questions",
        "practice questions",
        "quiz questions",
    )

    has_all = bool(re.search(
        r"\b(all|every|complete|entire|full|each)\b",
        text,
    ))
    has_items = bool(re.search(
        r"\b(questions?|qns?|subquestions?|topics?|points?|steps?|definitions?|examples?|problems?|exercises?)\b",
        text,
    ))
    generation_request = bool(re.search(
        r"\b(create|generate|make|prepare)\b.{0,50}\b(questions?|mcqs?|quiz|practice)\b", text
    ))
    return any(phrase in text for phrase in extraction_phrases) or (has_all and has_items) or generation_request


def _build_context(chunks, chunk_ids):
    """Build source-labelled context for Gemini."""

    context_parts = []
    source_map = {}

    for source_id, chunk_id in enumerate(
        chunk_ids,
        start=1,
    ):
        chunk = chunks[chunk_id]

        source_map[source_id] = (
            chunk.get("source", "Unknown PDF"),
            chunk.get("page", "?"),
        )

        context_parts.append(
            f"SOURCE_ID: {source_id}\n"
            f"File: {chunk.get('source', 'Unknown PDF')}\n"
            f"Page: {chunk.get('page', '?')}\n"
            f"Paper: {chunk.get('paper_title') or 'Not identified'}\n"
            f"Section: {chunk.get('section') or 'Not specified'}\n"
            f"Content:\n{chunk['text']}"
        )

    return "\n\n".join(context_parts), source_map


# ============================================================
# GENERIC SECTION RESOLUTION
# ============================================================

def _canonical_section_key(value):
    """Normalize section labels from different chunker formats."""
    text = str(value or "").strip().lower()
    pattern = r"\b(module|mod|unit|chapter|chap|ch|week|section|part|lesson|lecture|topic|session|day)\s*[-_:. ]*\s*(\d+|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten)\b"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    kind = match.group(1).lower()
    kind = {"mod": "module", "chap": "chapter", "ch": "chapter"}.get(kind, kind)
    token = match.group(2).lower()
    words = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}
    romans = {"i":1,"ii":2,"iii":3,"iv":4,"v":5,"vi":6,"vii":7,"viii":8,"ix":9,"x":10}
    number = int(token) if token.isdigit() else words.get(token, romans.get(token))
    return f"{kind}_{number}" if number is not None else f"{kind}_{token}"


def _section_key_parts(value):
    match = re.fullmatch(r"([a-z]+)_(\d+)", _canonical_section_key(value))
    return (match.group(1), int(match.group(2))) if match else None


def _find_section_chunks(chunks, requested_keys):
    """Find exact section labels, then infer missing labels from numbered boundaries per PDF."""
    requested = [_canonical_section_key(key) for key in requested_keys]
    selected = set()
    sources = list(dict.fromkeys(c.get("source", "Unknown PDF") for c in chunks))

    # Exact metadata match works with keys like module_1 or Module 1.
    for i, chunk in enumerate(chunks):
        value = chunk.get("section_key") or chunk.get("section")
        if _canonical_section_key(value) in requested:
            selected.add(i)

    for requested_key in requested:
        requested_parts = _section_key_parts(requested_key)
        if not requested_parts:
            continue
        kind, number = requested_parts

        for source in sources:
            source_ids = [i for i,c in enumerate(chunks) if c.get("source", "Unknown PDF") == source]
            if not source_ids:
                continue
            # If this PDF has a direct label, use it rather than infer a duplicate span.
            if any(_canonical_section_key(chunks[i].get("section_key") or chunks[i].get("section")) == requested_key for i in source_ids):
                continue

            headings = []
            for i in source_ids:
                chunk = chunks[i]
                label = chunk.get("section_key") or chunk.get("section")
                parts = _section_key_parts(label)
                if parts and parts[0] == kind:
                    headings.append((i, parts[1]))
                    continue
                # Recover headings from actual extracted chunk text if metadata missed them.
                for line in (chunk.get("text") or "").splitlines()[:8]:
                    line = re.sub(r"^\[[^\]]+\]\s*", "", line).strip()
                    parts = _section_key_parts(line)
                    if parts and parts[0] == kind:
                        headings.append((i, parts[1]))
                        break

            # Infer only when a later numbered boundary proves where this span ends.
            later = [(i,n) for i,n in headings if n > number]
            if not later:
                continue
            end_id, _ = min(later, key=lambda pair: pair[0])
            earlier = [(i,n) for i,n in headings if n < number and i < end_id]
            if earlier:
                previous_id, _ = max(earlier, key=lambda pair: pair[0])
                start_pos = source_ids.index(previous_id) + 1
            else:
                start_pos = 0
            end_pos = source_ids.index(end_id)
            inferred = source_ids[start_pos:end_pos]
            if inferred and any((chunks[i].get("text") or "").strip() for i in inferred):
                selected.update(inferred)
                print(f"Section boundary inference: {requested_key}; {len(inferred)} chunks from {source!r} before the next {kind} heading.")

    return sorted(selected)


# ============================================================
# SOURCE-FAITHFUL SECTION EXTRACTION
# ============================================================

def _format_section_extraction(chunks, matching_ids, requested_keys):
    """Return exact indexed section text without requiring Gemini."""
    grouped = {}
    for chunk_id in matching_ids:
        chunk = chunks[chunk_id]
        source = chunk.get("source", "Unknown PDF")
        grouped.setdefault(source, []).append(chunk)

    parts = [
        "Here is the text found under " + ", ".join(requested_keys) +
        " in your uploaded PDFs. The text below is extracted from the documents."
    ]
    for source, source_chunks in grouped.items():
        parts.append(f"\n### {source}")
        seen = set()
        for chunk in source_chunks:
            text = (chunk.get("text") or "").strip()
            # Remove the synthetic label added by the chunker; retain actual PDF text.
            text = re.sub(r"^\[[^\]]+\]\s*\n", "", text, count=1)
            if not text or text == "(No content found under this heading in the PDF.)":
                continue
            # Avoid exact duplicate chunks while preserving document order.
            if text in seen:
                continue
            seen.add(text)
            parts.append(f"\n**Page {chunk.get('page', '?')}**\n\n{text}")

    if len(parts) == 1:
        return NOT_FOUND
    return "\n".join(parts)


# ============================================================
# ANSWER GENERATION
# ============================================================

def _requested_bullet_count(question):
    """Return a requested bullet count, including common informal phrasings."""
    text = (question or "").lower().strip()
    number = r"(one|two|three|four|five|six|seven|eight|nine|ten|\d+)"
    patterns = (
        rf"\bexactly\s+{number}\s+(?:bullet\s+points?|bullets?|points?|items?)\b",
        rf"\b{number}\s+(?:bullet\s+points?|bullets?|numbered\s+points?|points?|items?)\b",
        rf"\b(?:make|give|write|convert|turn|put|format)\b.{{0,45}}?\b{number}\s+(?:bullet\s+points?|bullets?|numbered\s+points?|points?|items?)\b",
    )
    match = next((m for pattern in patterns if (m := re.search(pattern, text))), None)
    if not match:
        return None
    # The number is always the last capture group for each pattern, but patterns
    # may contain an earlier capture; choose the rightmost non-empty group.
    token = next((g for g in reversed(match.groups()) if g), None)
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    return words.get(token, int(token) if token and token.isdigit() else None)

def _enforce_bullet_count(answer, count):
    """Keep exactly N bullet lines by merging surplus items without inventing facts."""
    if not count or count < 1:
        return answer
    lines = (answer or "").splitlines()
    bullet_indexes = [i for i, line in enumerate(lines) if re.match(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", line)]
    if len(bullet_indexes) <= count:
        return answer
    bullets = [re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", lines[i]).strip() for i in bullet_indexes]
    # Preserve every point: merge overflow into the final allowed bullet.
    kept = bullets[:count - 1] + ["; ".join(x.rstrip(".; ") for x in bullets[count - 1:])]
    output = []
    first_bullet = bullet_indexes[0]
    output.extend(lines[:first_bullet])
    output.extend(f"- {item}" for item in kept)
    last_bullet = bullet_indexes[-1]
    output.extend(lines[last_bullet + 1:])
    return "\n".join(output).strip()

def ask_question(index, chunks, question, history=None):
    """
    Retrieve relevant PDF chunks and answer using their contents.

    Supports general questions, section-based queries,
    extraction requests, and follow-up questions.
    """

    
        
        
    if not question or not question.strip():
        return "Please enter a question."

    # Handle greetings and casual messages without searching PDFs.
    casual_message = question.strip().lower()
    casual_message = re.sub(r"[^\w\s']", "", casual_message)
    casual_message = re.sub(r"\s+", " ", casual_message).strip()

    greetings = {
        "hi", "hello", "hey", "hii", "hiii",
        "good morning", "good afternoon", "good evening",
        "good night", "howdy", "sup",
        "thank you", "thanks", "thx",
        "bye", "goodbye",
    }

    if casual_message in greetings:
        responses = {
            "hi": "Hi! 👋 Welcome to CampusGPT. What would you like to study today?",
            "hello": "Hello! 😊 I'm CampusGPT. Ask me anything about your uploaded study materials!",
            "hey": "Hey! 👋 Ready to study? Ask me a question about your PDFs.",
            "hii": "Hii! 💜 What would you like to learn today?",
            "hiii": "Hiii! 💜 What can I help you study?",
            "good morning": "Good morning! ☀️ What are we studying today?",
            "good afternoon": "Good afternoon! 😊 How can I help with your studies?",
            "good evening": "Good evening! 🌟 What would you like to study?",
            "good night": "Good night! 🌙 Rest well, and good luck with your studies!",
            "howdy": "Hey there! 👋 What can I help you learn?",
            "sup": "Hey! 😄 What are we working on today?",
            "thank you": "You're welcome! 💜 Happy to help.",
            "thanks": "You're welcome! 💜",
            "thx": "Anytime! 😊",
            "bye": "Bye! 👋 Good luck with your studies!",
            "goodbye": "Goodbye! 👋 See you next time!",
        }
        return responses[casual_message]

    if not chunks:
        return (
            "No PDF content is available. "
            "Please upload a PDF first."
        )

    try:
        original_question = question.strip()

        # Formatting-only follow-ups should transform the immediately
        # preceding assistant answer instead of retrieving an older version
        # of the topic from the PDF. Preserve its source labels unchanged.
        reformatted_answer = _format_only_followup(
            original_question,
            history,
        )
        if reformatted_answer is not None:
            return reformatted_answer

        standalone = _standalone_question(
            original_question,
            history,
        )

        # Detect explicit section and paper references in the current request.
        section_keys = (
            find_sections_in_query(standalone)
            or find_sections_in_query(original_question)
        )
        requested_paper_title = (
            _extract_requested_paper_title(original_question)
            or _extract_requested_paper_title(standalone)
        )
        requested_paper_chunk_ids = None

        if requested_paper_title:
            requested_paper_chunk_ids = _paper_matching_chunk_ids(
                chunks,
                requested_paper_title,
            )
            if not requested_paper_chunk_ids:
                return (
                    f"I couldn't verify the requested paper title "
                    f"'{requested_paper_title}' in the uploaded PDFs. "
                    "I did not substitute content from another paper. "
                    "Check whether the paper title and year/effectivity text "
                    "are readable in the uploaded PDF."
                )
            print(
                "Paper-aware retrieval:",
                requested_paper_title,
                "| Matching chunks:",
                len(requested_paper_chunk_ids),
            )

        wants_many = _is_extraction_request(standalone) or bool(
            re.search(r"\b(all|every|entire|complete|full|list)\b", standalone.lower())
        )
        # Question generation is different from verbatim extraction. Even if
        # the request contains words such as "all" or "MCQs", generate the
        # requested questions from the selected section instead of returning
        # the raw section text.
        question_generation_request = bool(re.search(
            r"\b(create|generate|make|prepare)\b.{0,60}\b(questions?|mcqs?|quiz|practice)\b",
            original_question.lower(),
        ))

        # For explicit section requests, trust exact section metadata first.
        # This avoids an unnecessary query-embedding API call and prevents
        # unrelated documents from being substituted for a missing section.
        exact_section_ids = []
        if section_keys:
            exact_section_ids = _find_section_chunks(chunks, section_keys)

            # If a paper title was requested, require BOTH the paper and the
            # section to match. Never merge Module 1 from a different paper.
            if requested_paper_chunk_ids is not None:
                exact_section_ids = [
                    chunk_id for chunk_id in exact_section_ids
                    if chunk_id in requested_paper_chunk_ids
                ]

            print("Resolved section chunks:", len(exact_section_ids))

            if not exact_section_ids:
                if requested_paper_title:
                    return (
                        f"I found chunks tagged for the requested paper, but couldn't safely "
                        f"identify {', '.join(section_keys)} within "
                        f"'{requested_paper_title}'. No Module/Unit from another paper was used."
                    )
                detected = sorted({str(c.get("section_key")) for c in chunks if c.get("section_key")})
                detected_text = ", ".join(detected) if detected else "none detected"
                return (
                    f"I couldn't safely identify {', '.join(section_keys)} in the uploaded PDFs.\n\n"
                    f"Detected section headings: {detected_text}.\n\n"
                    "No unrelated section was substituted."
                )

            exact_section_ids = sorted(exact_section_ids)
            if wants_many and not question_generation_request:
                print("Returning complete section text without Gemini generation.")
                return _format_section_extraction(chunks, exact_section_ids, section_keys)

            hit_ids = exact_section_ids
            hits = [
                {"chunk_id": chunk_id, "chunk": chunks[chunk_id], "score": 0.0}
                for chunk_id in exact_section_ids
            ]
            ids = exact_section_ids
            context, source_map = _build_context(chunks, ids)

        if not section_keys:
            print("\n========== CAMPUSGPT SEARCH ==========")
            print("Question:", original_question)
            print("Search query:", standalone)
            print("Section keys:", section_keys)
            print("Total chunks:", len(chunks))

            # Create a query embedding for semantic retrieval. This must be
            # generated from the current question, not reused from indexing.
            query_vector = create_embedding(
                standalone,
                task_type="RETRIEVAL_QUERY",
            )
            broad_summary_request = bool(re.search(
                r"\b(summary|summari[sz]e|overview|main concepts?|important points?|key points?|topics covered)\b",
                standalone.lower(),
            ))
            # Summaries and question-generation need broad coverage, not only the
            # top few chunks. For small PDFs, retrieve every chunk.
            if question_generation_request:
                desired_top_k = min(20, len(chunks))
            else:
                desired_top_k = 8 if broad_summary_request else 4
            top_k = min(desired_top_k, max(1, len(chunks)))

            # When a paper title is explicit, search the full ranked list first,
            # then keep only chunks tagged with that paper. Searching only the
            # global top-k could otherwise miss the requested paper entirely.
            search_top_k = len(chunks) if requested_paper_chunk_ids is not None else top_k
            hits = search_vector_store(
                index,
                chunks,
                standalone,
                query_vector,
                top_k=search_top_k,
                section_keys=None,
            )

            if requested_paper_chunk_ids is not None:
                hits = [
                    hit for hit in hits
                    if hit["chunk_id"] in requested_paper_chunk_ids
                ]
                if not question_generation_request:
                    hits = hits[:top_k]

            # Question generation needs coverage across the selected material,
            # not only the highest-scoring chunks. Respect the requested paper
            # boundary when one was supplied.
            if question_generation_request and len(chunks) <= 40:
                candidate_ids = (
                    sorted(requested_paper_chunk_ids)
                    if requested_paper_chunk_ids is not None
                    else list(range(len(chunks)))
                )
                hits = [
                    {"chunk_id": chunk_id, "chunk": chunks[chunk_id], "score": 0.0}
                    for chunk_id in candidate_ids
                ]
                print("Question generation: using all chunks for coverage.")

            print("Retrieved chunks:", len(hits))
            if not hits:
                return (
                    "I couldn't find sufficiently relevant text in the uploaded PDFs. "
                    "Please try a more specific phrase from the document."
                )
            hit_ids = [hit["chunk_id"] for hit in hits]
            ids = _expand_neighbours(chunks, hit_ids)
            if requested_paper_chunk_ids is not None:
                ids = [
                    chunk_id for chunk_id in ids
                    if chunk_id in requested_paper_chunk_ids
                ]
            ids = sorted(set(ids))
            context, source_map = _build_context(chunks, ids)

            for hit in hits[:5]:
                chunk = hit["chunk"]
                print(
                    f"File: {chunk.get('source')} | Page: {chunk.get('page')} | "
                    f"Section: {chunk.get('section_key')} | Score: {hit.get('score', 0):.5f}"
                )
                print("Preview:", chunk.get("text", "")[:200])
            print("======================================\n")

        print("Context chunks:", len(ids))

        for hit in hits[:5]:
            chunk = hit["chunk"]

            print(
                f"File: {chunk.get('source')} | "
                f"Page: {chunk.get('page')} | "
                f"Section: {chunk.get('section_key')} | "
                f"Score: {hit.get('score', 0):.5f}"
            )

            print(
                "Preview:",
                chunk.get("text", "")[:200],
            )

        print("======================================\n")

        extraction_note = ""

        if wants_many:
            extraction_note = """
The user wants multiple items or an exhaustive list.
Include all relevant items present in the supplied context.
Do not invent missing items or claim the entire document
has been covered unless the supplied context supports that.
"""

        if re.search(r"\b(create|generate|make|prepare)\b.{0,60}\b(multiple-choice questions|mcqs?|short-answer questions|short answer questions|practice questions|quiz questions|questions)\b", original_question.lower()):
            extraction_note += """
QUESTION-GENERATION MODE:
Generate the requested number and type of questions from the supplied PDF context.
For MCQs, include four options (A-D) and mark the correct answer after each question.
For short-answer questions, provide a concise answer supported by the context.
Do not claim the PDF contains no information when the supplied context contains relevant material.
Do not use facts outside the supplied PDF context. If the requested count exceeds the available distinct facts, generate fewer rather than inventing content.
"""

        summary_note = ""
        if re.search(r"\b(summary|summari[sz]e|overview|main concepts?|important points?|key points?|topics covered)\b", standalone.lower()):
            summary_note = """
This is a broad summary/overview request. Cover the distinct major topics present
in the supplied context instead of summarizing only the first few chunks. Group
related details where appropriate, avoid repeated ideas, and do not claim to cover
the entire PDF if the supplied context is incomplete. Do not invent unsupported topics.
"""

        section_note = ""

        if section_keys:
            section_note = f"""
The question explicitly mentions these sections:
{", ".join(section_keys)}.

Prefer information belonging to the requested section.
Do not mix information from unrelated sections when answering
a section-specific request.

If the context does not establish which document or section
matches a requested paper title or year, clearly state that
the exact match could not be verified.
"""

        missing_phrase_rule = ""
        if re.search(r"(?is)(?:write|say|state|use)\s+(?:exactly\s+)?[\"']?not mentioned in the pdf", original_question):
            missing_phrase_rule = "If a requested fact or cell is not explicitly supported by the PDF context, use exactly: Not mentioned in the PDF. Do not replace it with a paraphrase."

        prompt = f"""
You are CampusGPT, a careful assistant that answers questions
about user-uploaded PDF documents.

Use the supplied PDF context as your source of factual information.

RULES:
1. Answer the user's latest question using the supplied PDF context.
2. Interpret follow-up questions using the resolved standalone question and
   the original wording shown below.
3. Answer the intended question, not merely the literal words of an incomplete
   follow-up.
4. Follow the latest formatting instruction. The latest request takes
   priority over any older format in the conversation:
   - Use a Markdown table ONLY when the latest user message explicitly asks
     for a table.
   - If the latest message asks for points, or the resolved search question
     says "numbered list, not a table", use a numbered list.
   - If the user asks for bullet points, use bullet points.
   - If the user asks for a paragraph, use paragraph form.
   - Never keep an old table format just because an earlier answer was a table.
5. If the user specifies a number of points, output exactly that many
   clearly numbered items when the PDF context supports them. A request such
   as “Give only two points” must not return a paragraph or repeat the prior
   answer unchanged. Do not invent facts just to reach the requested count.
   If the user asks to explain “the second point”, explain only item 2 from
   the immediately preceding answer, not the whole topic or a different item.
   For a follow-up like "Give any two points" after a comparison, give two
   numbered differences, not a two-row comparison table.
6. If the user asks for simple English, explain the same content with familiar
   words and short sentences, avoiding unnecessary technical terms.
7. If the user asks for more detail, examples, advantages, disadvantages, or
   importance, answer that specific request about the resolved topic.
8. For "same answer in simple English" after a request for points, keep
   the numbered-list format and simplify the wording. Do not convert that
   answer into a table. Preserve a prior format only when it was explicitly
   requested for the same answer and no later instruction implies a different
   format.
9. Use only factual information supported by the supplied PDF context. Do not
   invent facts, quotations, filenames, page numbers, years, or section contents.
10. If the answer is not supported by the context, clearly say you could not
    find the information in the uploaded PDFs. If the user explicitly requests
    the exact phrase "Not mentioned in the PDF", use that exact phrase for the
    missing fact/cell, without adding an unsupported alternative.
11. Do not say information is absent merely because the user used different
    wording from the PDF.
12. For exact questions, quotations, definitions, lists, or steps, preserve the
    document's wording as closely as practical.
13. If multiple PDFs contain similar content, distinguish them by filename and
    page number.
14. Preserve requested paper names, years, module numbers, unit numbers,
    chapter numbers, and other identifiers. Do not silently substitute another.
15. Do not mix unrelated modules, papers, or sections in a section-specific answer.
15A. VISUAL OUTPUT: If the user explicitly asks for a flowchart, process diagram,
     decision tree, sequence diagram, or concept map, provide a valid Mermaid
     diagram inside a fenced ```mermaid code block, followed by a short explanation.
     Use only facts supported by the supplied PDF context. If the PDF lacks enough
     information to build a faithful diagram, state what is missing instead of guessing.
15B. If a diagram or image is already embedded in the PDF and the user asks what it
     shows, use only related captions and surrounding text in the supplied context.
     Do not claim to have visually inspected an embedded image unless image content
     was actually supplied to the model. Refer the user to the PDF page preview in
     the app when visual inspection is needed.
15C. If the user asks to convert existing content into a diagram, preserve the same
     concepts, order, labels, and relationships. Do not add unsupported steps.
16. Cite only source IDs from the supplied context that actually support the
    answer. Never invent citations or page numbers. Do not insert inline [1],
    [2], etc. markers; use the required SOURCE_IDS trailer only.
17. Answer in the same language as the user's question.

{section_note}

{missing_phrase_rule}

{summary_note}

{extraction_note}

PDF CONTEXT:
{context}

RESOLVED SEARCH QUESTION:
{standalone}

ORIGINAL LATEST USER MESSAGE (follow its requested format and style):
{original_question}

CONVERSATION-HISTORY SAFETY:
The resolved search question above already contains the topic needed for this
answer. Do NOT use previous assistant answers as factual evidence and do NOT
repeat them just because they appear in conversation history. Answer the latest
question directly from the PDF context. If the latest question asks what happened
next, the result, or the final outcome, answer that specific point. If the PDF
context does not state it, say that it is not specified. Use conversation history
only to resolve references in the resolved search question, never as a source.

Write the answer first.

On the final line, write:
SOURCE_IDS: followed by the source numbers used, or NONE.
"""

        answer_text = _generate(
            prompt,
            temperature=0.1,
        )

        if not answer_text:
            return (
                "Sorry, I couldn't generate an answer. "
                "Please try again."
            )

        match = re.search(
            r"(.*?)\s*SOURCE_IDS:\s*(.*)",
            answer_text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        if match:
            answer = match.group(1).strip()
            source_ids_text = match.group(2).strip()
        else:
            # Some model responses omit the source trailer. Keep the answer and
            # still apply deterministic output-format validation.
            answer = answer_text.strip()
            source_ids_text = "NONE"

        # Mermaid safety net: some model responses incorrectly return a temporary
        # browser blob URL instead of Mermaid source. Generate valid Mermaid code
        # from the same retrieved PDF context, while keeping the original SOURCE_IDS
        # trailer so the normal citation logic below remains authoritative.
        diagram_request = bool(re.search(
            r"\b(flowchart|flow chart|process diagram|decision tree|sequence diagram|concept map)\b",
            original_question,
            flags=re.IGNORECASE,
        ))
        has_mermaid = bool(re.search(
            r"```mermaid\s*\n.+?```", answer, flags=re.DOTALL | re.IGNORECASE
        ))
        has_temporary_diagram_link = bool(re.search(
            r"blob:https?://|mermaid\.ink|kroki\.io", answer, flags=re.IGNORECASE
        ))
        if diagram_request and (not has_mermaid or has_temporary_diagram_link):
            diagram_prompt = f"""You are CampusGPT, answering only from the supplied PDF context.
Create a valid Mermaid diagram that directly answers the user's request.
Use simple Mermaid syntax (flowchart TD is preferred), short node labels, and
quote labels that contain punctuation. Do not use HTML, external URLs, blob URLs,
markdown links, or image links. Return only one fenced Mermaid code block and
nothing else. Do not invent steps or facts not supported by the PDF context.
If the context does not contain enough information, return exactly:
INSUFFICIENT_CONTEXT

PDF CONTEXT:
{context}

USER REQUEST:
{original_question}
"""
            try:
                diagram_text = _generate(diagram_prompt, temperature=0.1)
                if diagram_text and "INSUFFICIENT_CONTEXT" not in diagram_text.upper():
                    code_match = re.search(
                        r"```(?:mermaid)?\s*\n(.*?)```",
                        diagram_text,
                        flags=re.DOTALL | re.IGNORECASE,
                    )
                    diagram_code = code_match.group(1).strip() if code_match else diagram_text.strip()
                    # Avoid accidentally accepting another temporary link as diagram source.
                    if diagram_code and not re.search(
                        r"blob:https?://|mermaid\.ink|kroki\.io", diagram_code, re.IGNORECASE
                    ):
                        diagram_code = re.sub(r"^```(?:mermaid)?\s*|\s*```$", "", diagram_code, flags=re.IGNORECASE).strip()
                        if re.match(r"^(flowchart|graph|sequenceDiagram|classDiagram|stateDiagram|erDiagram|journey|mindmap|timeline)\b", diagram_code):
                            # Remove the invalid generated link/code from the answer,
                            # but keep the explanatory text and its citation IDs.
                            answer = re.sub(r"(?im)^.*(?:blob:https?://|mermaid\.ink|kroki\.io).*\n?", "", answer).strip()
                            answer = re.sub(r"```mermaid\s*.*?```", "", answer, flags=re.DOTALL | re.IGNORECASE).strip()
                            answer = (answer + "\n\n" if answer else "") + "```mermaid\n" + diagram_code + "\n```"
            except Exception as diagram_exc:
                # Preserve the original answer if diagram recovery fails.
                print("CampusGPT Mermaid recovery warning:", type(diagram_exc).__name__)

        requested_bullets = _requested_bullet_count(original_question)
        if requested_bullets and re.search(
            r"\b(?:bullet\s+points?|bullets?)\b", original_question.lower()
        ):
            answer = _enforce_bullet_count(answer, requested_bullets)

        if not answer:
            return (
                "Sorry, I couldn't generate an answer. "
                "Please try again."
            )

        # The application adds the authoritative source list from SOURCE_IDS.
        # Remove any source list that Gemini accidentally included in the body,
        # otherwise users see duplicate or conflicting citations.
        answer = re.sub(
            r"(?is)\n\s*(?:\*{0,2}\s*)sources?\s*(?:\*{0,2})\s*:?\s*\n(?:\s*[-*+]\s+.*(?:\n|$))+\s*$",
            "",
            answer,
        ).strip()
        # Also remove a compact model-generated citation footer such as
        # 'Sources: file.pdf, page 10' while preserving normal answer text.
        answer = re.sub(
            r"(?im)\n\s*(?:\*{0,2}\s*)sources?\s*(?:\*{0,2})\s*:\s*[^\n]+\s*$",
            "",
            answer,
        ).strip()

        # The source trailer below is authoritative. Remove explicit page
        # references in model prose when that page is not among retrieved pages.
        allowed_pages = {
            str(page) for _, page in source_map.values()
            if page is not None and str(page).isdigit()
        }
        def keep_supported_page(match):
            mentioned_pages = re.findall(r"\d+", match.group(0))
            return match.group(0) if mentioned_pages and all(page in allowed_pages for page in mentioned_pages) else ""
        answer = re.sub(
            r"(?i)\b(pages?\s+)(\d+)(?:\s*[-–—,]\s*\d+)*\b",
            keep_supported_page,
            answer,
        )

        if (
            source_ids_text.upper().startswith("NONE")
            or NOT_FOUND.lower() in answer.lower()[:200]
        ):
            return answer

        labels = []

        for token in re.findall(
            r"\d+",
            source_ids_text,
        ):
            source_id = int(token)

            if source_id not in source_map:
                continue

            filename, page = source_map[source_id]

            label = f"{filename} — Page {page}"

            if label not in labels:
                labels.append(label)

        if labels:
            answer += (
                "\n\n**Sources:**\n"
                + "\n".join(
                    f"- {label}"
                    for label in labels
                )
            )

        return answer

    except RuntimeError as exc:
        print("CampusGPT API/runtime error:", type(exc).__name__, str(exc))
        return str(exc)

    except Exception as exc:
        print("CampusGPT error:", repr(exc))

        return (
            "Something went wrong while searching the uploaded PDFs. "
            "Please check the terminal for the error details and try again."
        )
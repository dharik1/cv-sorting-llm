"""
document_parser.py - Loading CVs / job descriptions (PDF, DOCX, TXT) and privacy helpers.

Text is extracted with pypdf (PDF) and python-docx (Word), then normalised. `redact_pii` masks
e-mail addresses, phone numbers, profile URLs and the candidate's name so that the LLM judge
performs blind screening (identity cues cannot influence the score).
"""
from __future__ import annotations

import os
import re

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md")

_CHAR_MAP = {"\ufb01": "fi", "\ufb02": "fl", "\u2013": "-", "\u2014": "-", "\u2022": "-",
             "\u25cf": "-", "\u00b7": "-", "\u00a0": " ", "\u2019": "'", "\u201c": '"', "\u201d": '"'}
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL_RE = re.compile(r"(?:https?://\S+|www\.\S+|\b(?:linkedin|github|gitlab)\.com/\S+)", re.IGNORECASE)
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{8,}\d")
_TITLES = {"mr", "mrs", "ms", "miss", "dr", "prof"}


def list_documents(folder: str) -> list:
    """Return the sorted paths of all supported documents directly inside `folder`."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"CV folder not found: {folder}")
    return sorted(os.path.join(folder, name) for name in os.listdir(folder)
                  if name.lower().endswith(SUPPORTED_EXTENSIONS) and not name.startswith((".", "~$")))


def read_document(path: str) -> str:
    """Extract and normalise plain text from a PDF, DOCX, TXT or Markdown file."""
    extension = os.path.splitext(path)[1].lower()
    if extension == ".pdf":
        text = _read_pdf(path)
    elif extension == ".docx":
        text = _read_docx(path)
    elif extension in (".txt", ".md"):
        with open(path, encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
    else:
        raise ValueError(f"Unsupported file type: {path}")
    return clean_text(text)


def _read_pdf(path: str) -> str:
    """Extract the text of every page of a PDF with pypdf (scanned image-only PDFs yield no text)."""
    from pypdf import PdfReader
    return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


def _read_docx(path: str) -> str:
    """Extract paragraph and table text from a Word document with python-docx."""
    import docx
    document = docx.Document(path)
    parts = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n".join(parts)


def clean_text(text: str) -> str:
    """Normalise ligatures, bullets, quotes and whitespace while keeping the line structure."""
    for source, target in _CHAR_MAP.items():
        text = text.replace(source, target)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _mask_phone(match: re.Match) -> str:
    """Mask a digit run only if it looks like a phone number (>= 10 digits, so year ranges survive)."""
    return "[PHONE]" if sum(ch.isdigit() for ch in match.group()) >= 10 else match.group()


def redact_pii(text: str, candidate_name: str = "") -> str:
    """Return a blind-screening copy of the CV with contact details and the candidate's name masked."""
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _URL_RE.sub("[URL]", text)
    text = _PHONE_RE.sub(_mask_phone, text)
    tokens = {t for t in re.split(r"[\s.,]+", candidate_name or "") if len(t) >= 3 and t.lower() not in _TITLES}
    for token in sorted(tokens, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(token)}\b", "[CANDIDATE]", text, flags=re.IGNORECASE)
    return text

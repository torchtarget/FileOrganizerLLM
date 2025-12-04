from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Iterable, List, Tuple

from docx import Document
from pypdf import PdfReader

from .config import SAMPLE_BYTES

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".rtf",
    ".csv",
    ".json",
    ".yaml",
    ".yml",
    ".ini",
    ".log",
    ".py",
    ".java",
    ".js",
    ".ts",
    ".html",
    ".css",
    ".xml",
}

PDF_MIME = {"application/pdf"}
DOCX_MIME = {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}


def is_textual(path: Path) -> bool:
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True
    mime, _ = mimetypes.guess_type(path)
    return bool(mime and (mime.startswith("text/") or mime in PDF_MIME or mime in DOCX_MIME))


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:SAMPLE_BYTES]
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="ignore")[:SAMPLE_BYTES]


def read_pdf(path: Path) -> str:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages[:3]:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)[:SAMPLE_BYTES]


def read_docx(path: Path) -> str:
    document = Document(path)
    paragraphs = [p.text for p in document.paragraphs]
    return "\n".join(paragraphs)[:SAMPLE_BYTES]


def safe_extract(path: Path) -> Tuple[str, List[str]]:
    errors: List[str] = []
    text = ""
    try:
        if path.suffix.lower() == ".pdf":
            text = read_pdf(path)
        elif path.suffix.lower() in {".docx", ".doc"}:
            text = read_docx(path)
        else:
            text = read_text_file(path)
    except Exception as exc:  # pylint: disable=broad-except
        errors.append(f"Failed to read {path.name}: {exc}")
    return text[:SAMPLE_BYTES], errors


def sample_files(files: Iterable[Path], limit: int) -> List[Path]:
    candidates = sorted(files, key=lambda p: p.stat().st_mtime)
    if len(candidates) <= limit:
        return candidates

    oldest = candidates[:3]
    newest = candidates[-3:]
    remaining_pool = [f for f in candidates if f not in oldest + newest]

    remaining_needed = max(0, limit - len(oldest) - len(newest))
    sampled: List[Path] = []
    if remaining_needed > 0 and remaining_pool:
        stride = max(1, len(remaining_pool) // remaining_needed)
        for idx in range(remaining_needed):
            pick_index = min(idx * stride, len(remaining_pool) - 1)
            sampled.append(remaining_pool[pick_index])

    combined = list(dict.fromkeys(oldest + sampled + newest))
    return combined[:limit]

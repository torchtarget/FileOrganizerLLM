from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


ROOT_CONSTRAINTS: Dict[str, str] = {
    "Business": (
        "Strictly commercial, financial, legal, strategic, and operational content. "
        "EXCLUDES: personal, family, domestic, medical, hobby, intimate, or unrelated materials."
    ),
    "Private": (
        "Strictly personal, family, health, education, hobbies, and private financial documents. "
        "EXCLUDES: corporate, client, revenue-generating, or organizational materials."
    ),
}

SCHEMA_VERSION = "1.1"
DEFAULT_LANGUAGE = "en"
DEFAULT_CONFIDENCE = 0.82

MIN_TEXT_FILES = 5
SAMPLE_LIMIT = 15
SAMPLE_BYTES = 2048


def detect_root_constraint(root_path: Path, current_path: Path) -> str:
    try:
        first_segment = current_path.relative_to(root_path).parts[0]
    except Exception:
        return "General: derive meaning only from content."

    return ROOT_CONSTRAINTS.get(first_segment, "General: derive meaning only from content.")


def build_path_context(root_path: Path, current_path: Path) -> str:
    try:
        parts = list(current_path.relative_to(root_path).parts)
    except Exception:
        parts = list(current_path.parts)
    return " > ".join(parts) if parts else current_path.name


@dataclass
class BuilderSettings:
    root_path: Path
    provider: str = "stub"
    follow_symlinks: bool = False
    allow_parallel: bool = False


from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from .config import DEFAULT_CONFIDENCE, DEFAULT_LANGUAGE, SCHEMA_VERSION


class NodeType(str, Enum):
    LEAF = "LEAF"
    BRANCH = "BRANCH"


class Meta(BaseModel):
    path: str
    node_type: NodeType
    depth: int
    language: str = DEFAULT_LANGUAGE
    confidence: float = DEFAULT_CONFIDENCE
    structural_hash: Optional[str] = None


class Constraints(BaseModel):
    path_context: str
    root_rule: str
    parent_constraint: Optional[str] = None  # Added for hierarchical constraints


class Persona(BaseModel):
    short_label: str
    description: str
    derived_from: List[str] = Field(default_factory=list)
    negative_constraints: List[str] = Field(default_factory=list)


class VectorData(BaseModel):
    hypothetical_user_queries: List[str] = Field(default_factory=list)
    embedding_model: Optional[str] = None
    embedding: Optional[List[float]] = None


class Audit(BaseModel):
    sample_count: int = 0
    outliers_found: int = 0
    errors: List[str] = Field(default_factory=list)


class FolderPersona(BaseModel):
    schema_version: str = SCHEMA_VERSION
    meta: Meta
    constraints: Constraints
    persona: Persona
    vector_data: VectorData = Field(default_factory=VectorData)
    audit: Audit = Field(default_factory=Audit)

    def write(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def from_file(cls, path: Path) -> "FolderPersona":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

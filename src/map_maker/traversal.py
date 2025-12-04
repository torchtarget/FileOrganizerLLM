from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, ValidationError

from .config import (
    DEFAULT_CONFIDENCE,
    MIN_TEXT_FILES,
    SAMPLE_LIMIT,
    BuilderSettings,
    build_path_context,
    detect_root_constraint,
)
from .database import PersonaDatabase
from .llm import BaseLLM, safe_generate

from .schema import Constraints, FolderPersona, Meta, NodeType, Persona, VectorData
from .text_extraction import is_textual, safe_extract, sample_files


# Pydantic schema for structured JSON output
class PersonaResponse(BaseModel):
    """Schema for LLM response"""
    persona: Persona
    vector_data: VectorData = Field(default_factory=VectorData)

SYSTEM_PROMPT_LEAF = """
You are Map Maker, a semantic file system classifier.
Obey path constraints strictly.
Reject outliers. Never hallucate details.

CRITICAL: You MUST return ONLY valid JSON matching this exact structure, with no additional text before or after:
{
  "persona": {
    "short_label": "Brief folder name",
    "description": "Concise semantic description",
    "derived_from": ["file1.pdf", "file2.docx"],
    "negative_constraints": ["what doesn't belong here"]
  },
  "vector_data": {
    "hypothetical_user_queries": [
      "Natural language question 1 someone might ask to find this folder",
      "Natural language question 2",
      "Natural language question 3"
    ]
  }
}
""".strip()

SYSTEM_PROMPT_BRANCH = """
You are Map Maker, a semantic aggregator.
You synthesize meaning from child folders only.
Never invent data not present in children or path constraints.

CRITICAL: You MUST return ONLY valid JSON matching this exact structure, with no additional text before or after:
{
  "persona": {
    "short_label": "Brief category name",
    "description": "Parent-level definition unifying children",
    "derived_from": ["Child1", "Child2"],
    "negative_constraints": ["what doesn't belong"]
  },
  "vector_data": {
    "hypothetical_user_queries": [
      "Natural language question 1 someone might ask to find this folder",
      "Natural language question 2",
      "Natural language question 3"
    ]
  }
}
""".strip()


class PersonaBuilder:
    def __init__(self, settings: BuilderSettings, llm: BaseLLM, database: PersonaDatabase):
        self.settings = settings
        self.llm = llm
        self.database = database
        self.visited: Set[Path] = set()

    def build_for_root(self, root: Path) -> FolderPersona:
        return self._process_directory(root, depth=0)

    def refine_with_parent_constraints(self, root: Path) -> None:
        """
        Second pass: Top-down refinement.
        Apply parent constraints to children and re-generate personas.
        """
        self._refine_directory(root, parent_persona=None, depth=0)

    def _refine_directory(self, path: Path, parent_persona: Optional[FolderPersona], depth: int) -> None:
        """
        Recursively refine personas with parent context (pre-order traversal).
        """
        # Load current persona from database
        current_persona = self.database.load_persona(str(path))
        if not current_persona:
            return  # Skip if not processed in first pass

        # Build parent constraint if parent exists
        parent_constraint = None
        if parent_persona:
            parent_constraint = (
                f"PARENT FOLDER: {parent_persona.persona.short_label}\n"
                f"PARENT SCOPE: {parent_persona.persona.description}\n"
                f"MUST BE: A subcategory or component of the parent scope\n"
                f"MUST NOT BE: {', '.join(parent_persona.persona.negative_constraints)}"
            )

        # Re-process if parent constraint exists and is different
        if parent_constraint and current_persona.constraints.parent_constraint != parent_constraint:
            print(f"  Refining with parent context: {path}")
            refined_persona = self._reprocess_with_parent_constraint(
                path, current_persona, parent_constraint, depth
            )
            if refined_persona:
                self.database.save_persona(refined_persona)
                current_persona = refined_persona

        # Recurse into children (pre-order: parent before children)
        try:
            children_dirs = [
                p for p in path.iterdir()
                if p.is_dir() and (self.settings.follow_symlinks or not p.is_symlink())
            ]
            for child in children_dirs:
                self._refine_directory(child, current_persona, depth + 1)
        except Exception:
            pass  # Skip if can't read directory

    def _reprocess_with_parent_constraint(
        self, path: Path, current_persona: FolderPersona, parent_constraint: str, depth: int
    ) -> Optional[FolderPersona]:
        """Re-generate persona with parent constraint"""
        try:
            # Get files or children
            files = [p for p in path.iterdir() if p.is_file()]
            textual_files = [f for f in files if is_textual(f)]
            children_dirs = [
                p for p in path.iterdir()
                if p.is_dir() and (self.settings.follow_symlinks or not p.is_symlink())
            ]

            root_rule = detect_root_constraint(self.settings.root_path, path)
            path_context = build_path_context(self.settings.root_path, path)

            # Re-build persona with parent constraint
            if current_persona.meta.node_type == NodeType.LEAF:
                persona, vector_data, audit_errors, sample_count = self._build_leaf_persona(
                    path, textual_files, root_rule, path_context, parent_constraint
                )
            else:
                # For BRANCH, load child personas from DB
                child_results = []
                for child in children_dirs:
                    child_persona = self.database.load_persona(str(child))
                    if child_persona:
                        child_results.append((child, child_persona))

                persona, vector_data, audit_errors, sample_count = self._build_branch_persona(
                    path, child_results, textual_files, [], root_rule, path_context, parent_constraint
                )

            # Create updated persona with parent constraint
            meta = Meta(
                path=str(path),
                node_type=current_persona.meta.node_type,
                depth=depth,
                confidence=DEFAULT_CONFIDENCE,
                structural_hash=current_persona.meta.structural_hash,
            )
            constraints = Constraints(
                path_context=path_context,
                root_rule=root_rule,
                parent_constraint=parent_constraint
            )

            folder_persona = FolderPersona(
                meta=meta,
                constraints=constraints,
                persona=persona,
                vector_data=vector_data,
            )
            folder_persona.audit.sample_count = sample_count
            folder_persona.audit.errors = audit_errors

            # Generate embedding for semantic search
            self._generate_embedding(folder_persona)

            return folder_persona
        except Exception as e:
            print(f"    Error refining {path}: {e}")
            return None

    def _process_directory(self, path: Path, depth: int) -> FolderPersona:
        # Check if folder already exists in database - skip if found
        existing_persona = self.database.load_persona(str(path))
        if existing_persona:
            print(f"  [SKIP] Already in DB: {path}")
            return existing_persona

        real = path.resolve()
        if real in self.visited and not self.settings.follow_symlinks:
            return self._symlink_placeholder(path, depth)
        self.visited.add(real)

        children_dirs = [
            p
            for p in path.iterdir()
            if p.is_dir() and (self.settings.follow_symlinks or not p.is_symlink())
        ]
        child_results: List[Tuple[Path, FolderPersona]] = []

        if children_dirs:
            if self.settings.allow_parallel:
                with ThreadPoolExecutor() as executor:
                    futures = {executor.submit(self._process_directory, c, depth + 1): c for c in children_dirs}
                    wait(futures)
                    for future, child in futures.items():
                        child_results.append((child, future.result()))
            else:
                for child in children_dirs:
                    child_results.append((child, self._process_directory(child, depth + 1)))

        files = [p for p in path.iterdir() if p.is_file()]
        textual_files = [f for f in files if is_textual(f)]
        text_file_count = len(textual_files)
        subfolder_count = len(children_dirs)
        is_empty = text_file_count == 0 and subfolder_count == 0

        loose_files = [f for f in files if f not in textual_files]

        node_type = self._classify_node(text_file_count, subfolder_count, is_empty)

        existing_persona = self._load_existing_persona(path)
        structural_hash = self._compute_structural_hash(path, files, child_results)
        if existing_persona and existing_persona.meta.structural_hash == structural_hash:
            return existing_persona

        root_rule = detect_root_constraint(self.settings.root_path, path)
        path_context = build_path_context(self.settings.root_path, path)

        if node_type == NodeType.LEAF:
            persona, vector_data, audit_errors, sample_count = self._build_leaf_persona(
                path, textual_files, root_rule, path_context
            )
        else:
            persona, vector_data, audit_errors, sample_count = self._build_branch_persona(
                path, child_results, textual_files, loose_files, root_rule, path_context
            )

        meta = Meta(
            path=str(path),
            node_type=node_type,
            depth=depth,
            confidence=DEFAULT_CONFIDENCE,
            structural_hash=structural_hash,
        )
        constraints = Constraints(path_context=path_context, root_rule=root_rule)

        folder_persona = FolderPersona(
            meta=meta,
            constraints=constraints,
            persona=persona,
            vector_data=vector_data,
        )
        folder_persona.audit.sample_count = sample_count
        folder_persona.audit.errors = audit_errors

        # Generate embedding for semantic search
        self._generate_embedding(folder_persona)

        # Save to database instead of writing to file
        self.database.save_persona(folder_persona)
        return folder_persona

    def _classify_node(self, text_count: int, subfolder_count: int, is_empty: bool) -> NodeType:
        if is_empty:
            return NodeType.BRANCH
        if text_count >= MIN_TEXT_FILES and text_count >= 2 * subfolder_count:
            return NodeType.LEAF
        return NodeType.BRANCH

    def _build_leaf_persona(
        self, path: Path, textual_files: List[Path], root_rule: str, path_context: str, parent_constraint: Optional[str] = None
    ) -> Tuple[Persona, VectorData, List[str], int]:
        samples = sample_files(textual_files, SAMPLE_LIMIT)
        snippets: List[str] = []
        errors: List[str] = []
        derived_from: List[str] = []
        for sample in samples:
            text, errs = safe_extract(sample)
            derived_from.append(sample.name)
            snippets.append(f"# {sample.name}\n{text[:500]}")
            errors.extend(errs)

        file_snippets = "\n".join(snippets)

        # Build prompt with optional parent constraint
        prompt_parts = [
            f"Absolute path: {path}",
            f"Hierarchy: {list(path.parts)}",
            f"GLOBAL CONSTRAINT: {root_rule}",
            f"PATH CONTEXT: {path_context}",
        ]

        if parent_constraint:
            prompt_parts.append(f"\n{parent_constraint}\n")

        prompt_parts.extend([
            f"FILES:\n{file_snippets}",
            "TASK: Identify the semantic category and produce concise JSON persona fields."
        ])

        user_prompt = "\n".join(prompt_parts)
        response = safe_generate(self.llm, SYSTEM_PROMPT_LEAF, user_prompt, response_schema=PersonaResponse)
        persona, vector_data, validation_errors = self._parse_llm_response(
            response.content,
            path.name or "Root",
            derived_from,
        )
        errors.extend(validation_errors)
        return persona, vector_data, errors, len(samples)

    def _build_branch_persona(
        self,
        path: Path,
        child_results: List[Tuple[Path, FolderPersona]],
        textual_files: List[Path],
        loose_files: List[Path],
        root_rule: str,
        path_context: str,
        parent_constraint: Optional[str] = None,
    ) -> Tuple[Persona, VectorData, List[str], int]:
        lines: List[str] = []
        derived_from: List[str] = []
        errors: List[str] = []
        for child_path, persona in child_results:
            lines.append(f"{child_path.name}: {persona.persona.description}")
            derived_from.append(persona.persona.short_label)

        if textual_files or loose_files:
            pseudo_summary = self._summarize_loose_files(textual_files + loose_files)
            lines.append(f"LooseFiles: {pseudo_summary}")
            derived_from.append("LooseFiles")

        children_block = "\n".join(lines)

        # Build prompt with optional parent constraint
        prompt_parts = [
            f"Absolute path: {path}",
            f"Hierarchy: {list(path.parts)}",
            f"GLOBAL CONSTRAINT: {root_rule}",
            f"PATH CONTEXT: {path_context}",
        ]

        if parent_constraint:
            prompt_parts.append(f"\n{parent_constraint}\n")

        prompt_parts.extend([
            f"CHILDREN:\n{children_block}",
            "TASK: Write a parent-level definition that unifies these children into a precise category header."
        ])

        user_prompt = "\n".join(prompt_parts)
        response = safe_generate(self.llm, SYSTEM_PROMPT_BRANCH, user_prompt, response_schema=PersonaResponse)
        persona, vector_data, validation_errors = self._parse_llm_response(
            response.content,
            path.name or "Root",
            derived_from,
        )
        errors.extend(validation_errors)
        sample_count = len(textual_files)
        return persona, vector_data, errors, sample_count

    def _parse_llm_response(
        self, content: str, default_label: str, derived_from: List[str]
    ) -> Tuple[Persona, VectorData, List[str]]:
        errors: List[str] = []
        fallback_description = content[:800]
        fallback_persona = Persona(
            short_label=default_label,
            description=fallback_description,
            derived_from=derived_from,
            negative_constraints=[],
        )
        fallback_vector = VectorData()

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            errors.append(f"LLM response was not valid JSON: {exc}")
            return fallback_persona, fallback_vector, errors

        if not isinstance(payload, dict):
            errors.append("LLM response JSON was not an object; using fallback persona.")
            return fallback_persona, fallback_vector, errors

        persona_payload = payload.get("persona", payload)
        if not isinstance(persona_payload, dict):
            errors.append("LLM persona payload was not an object; using fallback persona.")
            return fallback_persona, fallback_vector, errors

        persona_payload.setdefault("short_label", default_label)
        persona_payload.setdefault("description", fallback_description)
        persona_payload.setdefault("derived_from", derived_from)
        persona_payload.setdefault("negative_constraints", [])

        try:
            persona = Persona(**persona_payload)
        except ValidationError as exc:  # pragma: no cover - defensive guard
            errors.append(f"Persona validation failed; using fallback. Details: {exc}")
            persona = fallback_persona

        vector_payload = payload.get("vector_data", {})
        if not isinstance(vector_payload, dict):
            errors.append("LLM vector_data payload was not an object; ignoring vector data.")
            vector_payload = {}

        try:
            vector_data = VectorData(**vector_payload)
        except ValidationError as exc:  # pragma: no cover - defensive guard
            errors.append(f"Vector data validation failed; using empty vector data. Details: {exc}")
            vector_data = fallback_vector

        return persona, vector_data, errors

    def _summarize_loose_files(self, files: Iterable[Path]) -> str:
        names = [p.name for p in files]
        if not names:
            return "No loose files"
        if len(names) <= 6:
            return "Loose files: " + ", ".join(names)
        return "Loose files sample: " + ", ".join(names[:6]) + " (+more)"

    def _load_existing_persona(self, path: Path) -> Optional[FolderPersona]:
        # Load from database instead of file
        try:
            return self.database.load_persona(str(path))
        except Exception:
            return None

    def _generate_embedding(self, folder_persona: FolderPersona) -> None:
        """Generate embedding vector for semantic search"""
        # Only generate embeddings for Fireworks LLM
        from .llm import FireworksLLM
        if not isinstance(self.llm, FireworksLLM):
            return

        # Combine description and queries for embedding
        text_parts = [
            folder_persona.persona.short_label,
            folder_persona.persona.description,
        ]
        text_parts.extend(folder_persona.vector_data.hypothetical_user_queries)

        embedding_text = " | ".join(text_parts)

        try:
            embedding = self.llm.generate_embedding(embedding_text)
            if embedding:
                folder_persona.vector_data.embedding = embedding
                folder_persona.vector_data.embedding_model = self.llm.embedding_model
        except Exception as e:
            print(f"  Warning: Could not generate embedding: {e}")

    def _symlink_placeholder(self, path: Path, depth: int) -> FolderPersona:
        root_rule = detect_root_constraint(self.settings.root_path, path)
        path_context = build_path_context(self.settings.root_path, path)
        meta = Meta(
            path=str(path),
            node_type=NodeType.BRANCH,
            depth=depth,
            structural_hash="symlink-skip",
        )
        persona = Persona(
            short_label=path.name or "Symlink",
            description="Skipped because it resolves to a previously visited path.",
            derived_from=[],
            negative_constraints=[],
        )
        folder_persona = FolderPersona(
            meta=meta,
            constraints=Constraints(path_context=path_context, root_rule=root_rule),
            persona=persona,
        )
        folder_persona.audit.errors.append("Symlink loop detected; skipped.")

        # Save to database instead of writing to file
        try:
            self.database.save_persona(folder_persona)
        except Exception:
            pass
        return folder_persona

    def _compute_structural_hash(
        self, path: Path, files: List[Path], child_results: List[Tuple[Path, FolderPersona]]
    ) -> str:
        hasher = hashlib.sha256()
        file_entries = [f"{p.name}:{p.stat().st_mtime}" for p in sorted(files)]
        child_entries = [
            f"{child.name}:{persona.meta.structural_hash}"
            for child, persona in sorted(child_results, key=lambda c: c[0].name)
        ]
        payload = json.dumps({"files": file_entries, "children": child_entries}, sort_keys=True)
        hasher.update(payload.encode("utf-8"))
        return hasher.hexdigest()

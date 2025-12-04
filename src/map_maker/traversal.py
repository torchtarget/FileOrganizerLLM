from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

from .config import (
    DEFAULT_CONFIDENCE,
    MIN_TEXT_FILES,
    SAMPLE_LIMIT,
    BuilderSettings,
    build_path_context,
    detect_root_constraint,
)
from .llm import BaseLLM, safe_generate
from .schema import Constraints, FolderPersona, Meta, NodeType, Persona
from .text_extraction import is_textual, safe_extract, sample_files

SYSTEM_PROMPT_LEAF = """
You are Map Maker, a semantic file system classifier.
Obey path constraints strictly.
Reject outliers. Never hallucinate details.
Return concise JSON persona fields.
""".strip()

SYSTEM_PROMPT_BRANCH = """
You are Map Maker, a semantic aggregator.
You synthesize meaning from child folders only.
Never invent data not present in children or path constraints.
Return concise JSON persona fields.
""".strip()


class PersonaBuilder:
    def __init__(self, settings: BuilderSettings, llm: BaseLLM):
        self.settings = settings
        self.llm = llm
        self.visited: Set[Path] = set()

    def build_for_root(self, root: Path) -> FolderPersona:
        return self._process_directory(root, depth=0)

    def _process_directory(self, path: Path, depth: int) -> FolderPersona:
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
            persona, audit_errors, sample_count = self._build_leaf_persona(
                path, textual_files, root_rule, path_context
            )
        else:
            persona, audit_errors, sample_count = self._build_branch_persona(
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
        )
        folder_persona.audit.sample_count = sample_count
        folder_persona.audit.errors = audit_errors

        output_path = path / "folder_persona.json"
        folder_persona.write(output_path)
        return folder_persona

    def _classify_node(self, text_count: int, subfolder_count: int, is_empty: bool) -> NodeType:
        if is_empty:
            return NodeType.BRANCH
        if text_count >= MIN_TEXT_FILES and text_count >= 2 * subfolder_count:
            return NodeType.LEAF
        return NodeType.BRANCH

    def _build_leaf_persona(
        self, path: Path, textual_files: List[Path], root_rule: str, path_context: str
    ) -> Tuple[Persona, List[str], int]:
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
        user_prompt = (
            f"Absolute path: {path}\n"
            f"Hierarchy: {list(path.parts)}\n"
            f"GLOBAL CONSTRAINT: {root_rule}\n"
            f"PATH CONTEXT: {path_context}\n"
            f"FILES:\n{file_snippets}\n"
            "TASK: Identify the semantic category and produce concise JSON persona fields."
        )
        response = safe_generate(self.llm, SYSTEM_PROMPT_LEAF, user_prompt)
        description = response.content[:800]

        persona = Persona(
            short_label=path.name or "Root",
            description=description,
            derived_from=derived_from,
            negative_constraints=[],
        )
        return persona, errors, len(samples)

    def _build_branch_persona(
        self,
        path: Path,
        child_results: List[Tuple[Path, FolderPersona]],
        textual_files: List[Path],
        loose_files: List[Path],
        root_rule: str,
        path_context: str,
    ) -> Tuple[Persona, List[str], int]:
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
        user_prompt = (
            f"Absolute path: {path}\n"
            f"Hierarchy: {list(path.parts)}\n"
            f"GLOBAL CONSTRAINT: {root_rule}\n"
            f"PATH CONTEXT: {path_context}\n"
            f"CHILDREN:\n{children_block}\n"
            "TASK: Write a parent-level definition that unifies these children into a precise category header."
        )
        response = safe_generate(self.llm, SYSTEM_PROMPT_BRANCH, user_prompt)
        description = response.content[:800]

        persona = Persona(
            short_label=path.name or "Root",
            description=description,
            derived_from=derived_from,
            negative_constraints=[],
        )
        sample_count = len(textual_files)
        return persona, errors, sample_count

    def _summarize_loose_files(self, files: Iterable[Path]) -> str:
        names = [p.name for p in files]
        if not names:
            return "No loose files"
        if len(names) <= 6:
            return "Loose files: " + ", ".join(names)
        return "Loose files sample: " + ", ".join(names[:6]) + " (+more)"

    def _load_existing_persona(self, path: Path) -> Optional[FolderPersona]:
        persona_path = path / "folder_persona.json"
        if persona_path.exists():
            try:
                return FolderPersona.from_file(persona_path)
            except Exception:
                return None
        return None

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
        output_path = path / "folder_persona.json"
        try:
            folder_persona.write(output_path)
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

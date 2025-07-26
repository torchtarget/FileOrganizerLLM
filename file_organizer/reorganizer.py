"""Folder tree analysis and reorganization planning."""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, List, Tuple

from .llm_provider import get_llm
from .mapper import cosine_similarity, map_files, apply_mapping, call_llm


class Reorganizer:
    """Analyze an existing organized tree and suggest changes."""

    def __init__(
        self,
        root: str,
        *,
        model: str = "llama3",
        provider: str = "ollama",
        openai_api_key: str | None = None,
        fireworks_api_key: str | None = None,
    ) -> None:
        self.root = root
        self.model = model
        self.provider = provider
        self.openai_api_key = openai_api_key
        self.fireworks_api_key = fireworks_api_key
        self.llm = get_llm(
            model,
            provider,
            openai_api_key=openai_api_key,
            fireworks_api_key=fireworks_api_key,
        )
        self.plan_file = os.path.join(self.root, "reorg_plan.json")

    def load_data(self) -> tuple[Dict[str, list], Dict[str, str]]:
        vectors_path = os.path.join(self.root, "folder_vectors.json")
        contexts_path = os.path.join(self.root, "folder_contexts.json")
        with open(vectors_path, "r", encoding="utf-8") as f:
            vectors: Dict[str, list] = json.load(f)
        with open(contexts_path, "r", encoding="utf-8") as f:
            contexts: Dict[str, str] = json.load(f)
        return vectors, contexts

    def confirm_merge_with_llm(
        self,
        f1: str,
        f2: str,
        folder_contexts: Dict[str, str],
    ) -> bool:
        """Use the LLM to confirm that two folders should be merged."""

        ctx1 = folder_contexts.get(f1, "")
        ctx2 = folder_contexts.get(f2, "")
        prompt = (
            "You are assisting with reorganizing a folder tree. "
            "Given the short summaries of two folders, determine if they "
            "cover the same topic and should be merged.\n\n"
            f"Folder A ({f1}): {ctx1}\n"
            f"Folder B ({f2}): {ctx2}\n"
            "Respond with 'yes' or 'no'."
        )
        response = call_llm(prompt, self.llm).lower()
        return response.startswith("y")

    def suggest_merges(
        self,
        folder_vectors: Dict[str, list],
        folder_contexts: Dict[str, str],
        *,
        threshold: float = 0.9,
    ) -> List[Tuple[str, str]]:
        folders = list(folder_vectors)
        merges: List[Tuple[str, str]] = []
        for i in range(len(folders)):
            for j in range(i + 1, len(folders)):
                f1, f2 = folders[i], folders[j]
                sim = cosine_similarity(folder_vectors[f1], folder_vectors[f2])
                if sim >= threshold and self.confirm_merge_with_llm(
                    f1, f2, folder_contexts
                ):
                    merges.append((f1, f2))
        return merges

    def suggest_moves(
        self,
        folder_vectors: Dict[str, list],
        folder_contexts: Dict[str, str],
        *,
        top_n: int = 3,
        min_similarity: float = 0.0,
    ) -> Dict[str, str]:
        mapping = map_files(
            self.root,
            folder_vectors,
            folder_contexts,
            self.llm,
            top_n=top_n,
            min_similarity=min_similarity,
        )
        filtered = {}
        for src, dest in mapping.items():
            current = os.path.dirname(src)
            if dest and os.path.abspath(dest) != os.path.abspath(current):
                filtered[src] = dest
        return filtered

    def analyze(self) -> None:
        folder_vectors, folder_contexts = self.load_data()
        merges = self.suggest_merges(folder_vectors, folder_contexts)
        moves = self.suggest_moves(folder_vectors, folder_contexts)
        plan = {"merge_candidates": merges, "move_suggestions": moves}
        with open(self.plan_file, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)
        logging.info("Saved reorganization plan to %s", self.plan_file)

    def apply(self) -> None:
        if not os.path.exists(self.plan_file):
            raise FileNotFoundError(self.plan_file)
        with open(self.plan_file, "r", encoding="utf-8") as f:
            plan = json.load(f)
        moves = plan.get("move_suggestions", {})
        apply_mapping(moves)
        logging.info("Moved files according to plan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze and reorganize folders")
    parser.add_argument("--root", required=True, help="Root directory to analyze")
    parser.add_argument("--apply", action="store_true", help="Apply move suggestions from plan file")
    parser.add_argument("--model", default=os.environ.get("FO_OLLAMA_MODEL", "llama3"))
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai", "fireworks"],
        default=os.environ.get("FO_PROVIDER", "ollama"),
    )
    parser.add_argument("--openai-api-key", dest="openai_api_key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--fireworks-api-key", dest="fireworks_api_key", default=os.environ.get("FIREWORKS_API_KEY"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reorganizer = Reorganizer(
        args.root,
        model=args.model,
        provider=args.provider,
        openai_api_key=args.openai_api_key,
        fireworks_api_key=args.fireworks_api_key,
    )
    if args.apply:
        reorganizer.apply()
    else:
        reorganizer.analyze()


if __name__ == "__main__":
    main()

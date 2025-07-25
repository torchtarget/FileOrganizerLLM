import os
import json
import argparse
import shutil
import logging
from typing import Dict, List

from .organizer import get_embedding, extract_text_file
from .llm_provider import get_llm


def call_llm(prompt: str, llm, retries: int = 1) -> str:
    """Invoke ``llm`` with ``prompt`` handling transient errors."""
    for attempt in range(retries + 1):
        try:
            return llm.invoke(prompt).content.strip()
        except Exception as e:  # pragma: no cover - network errors
            if attempt >= retries:
                logging.error("[llm error] %s", e)
                return ""
            logging.warning("LLM call failed, retrying...")


def choose_best_folder_via_llm(
    file_text: str,
    candidate_folders: List[str],
    folder_contexts: Dict[str, str],
    llm,
) -> str:
    """Return the folder that best matches the file using the LLM."""

    folder_descriptions = []
    for i, folder in enumerate(candidate_folders, 1):
        desc = folder_contexts.get(folder, "")
        folder_descriptions.append(f"{i}. {folder}: {desc}")

    prompt = (
        "You are helping to organize files into folders. "
        "Choose the number of the folder that best matches the given file "
        "content.\n\nFile content:\n"
        f"{file_text}\n\nFolders:\n" + "\n".join(folder_descriptions) +
        "\n\nRespond with just the number of the best folder."
    )

    response = call_llm(prompt, llm)
    for token in response.split():
        if token.isdigit():
            idx = int(token)
            if 1 <= idx <= len(candidate_folders):
                return candidate_folders[idx - 1]
    return candidate_folders[0]


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not v1 or not v2:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(a * a for a in v1) ** 0.5
    norm2 = sum(a * a for a in v2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def load_folder_data(roots: List[str]) -> tuple[Dict[str, list], Dict[str, str]]:
    """Load and merge vectors and contexts from one or more organizer outputs."""

    merged_vectors: Dict[str, list] = {}
    merged_contexts: Dict[str, str] = {}
    for root in roots:
        vectors_path = os.path.join(root, "folder_vectors.json")
        contexts_path = os.path.join(root, "folder_contexts.json")
        with open(vectors_path, "r", encoding="utf-8") as f:
            vectors: Dict[str, list] = json.load(f)
        with open(contexts_path, "r", encoding="utf-8") as f:
            contexts: Dict[str, str] = json.load(f)
        merged_vectors.update(vectors)
        merged_contexts.update(contexts)

    return merged_vectors, merged_contexts


def suggest_folder_for_file(
    filepath: str,
    folder_vectors: Dict[str, list],
    folder_contexts: Dict[str, str],
    llm,
    *,
    top_n: int = 3,
    min_similarity: float = 0.0,
) -> str:
    """Return the folder that best fits the file content."""

    text = extract_text_file(filepath, n_chars=2000)
    embedding = get_embedding(text)

    scored = [
        (folder, cosine_similarity(embedding, vec))
        for folder, vec in folder_vectors.items()
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored or scored[0][1] < min_similarity:
        return ""

    candidates = [f for f, _ in scored[:top_n]]
    return choose_best_folder_via_llm(text, candidates, folder_contexts, llm)


def map_files(
    source: str,
    folder_vectors: Dict[str, list],
    folder_contexts: Dict[str, str],
    llm,
    *,
    top_n: int = 3,
    min_similarity: float = 0.0,
) -> Dict[str, str]:
    """Map each file in ``source`` recursively to a destination folder."""

    mapping: Dict[str, str] = {}
    for root, _, files in os.walk(source):
        for name in files:
            path = os.path.join(root, name)
            dest = suggest_folder_for_file(
                path,
                folder_vectors,
                folder_contexts,
                llm,
                top_n=top_n,
                min_similarity=min_similarity,
            )
            mapping[path] = dest
    return mapping


def apply_mapping(mapping: Dict[str, str]) -> None:
    """Move files according to ``mapping``."""
    for src, dst_folder in mapping.items():
        os.makedirs(dst_folder, exist_ok=True)
        shutil.move(src, os.path.join(dst_folder, os.path.basename(src)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map files to organized folders")
    parser.add_argument("--input", required=True, help="Folder with files to map")
    parser.add_argument(
        "--root",
        action="append",
        required=True,
        help=(
            "Root folder containing folder_vectors.json and folder_contexts.json. "
            "Use multiple --root options to combine several organized trees."
        ),
    )
    parser.add_argument(
        "--apply", action="store_true", help="Move files based on generated mapping"
    )
    parser.add_argument("--top-n", type=int, default=3, help="Number of top vector matches to consider")
    parser.add_argument(
        "--min-sim",
        type=float,
        default=0.0,
        help="Minimum cosine similarity required to assign a folder",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("FO_OLLAMA_MODEL", "llama3"),
        help="Model name for the selected provider",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai", "fireworks"],
        default=os.environ.get("FO_PROVIDER", "ollama"),
        help="LLM provider to use",
    )
    parser.add_argument(
        "--openai-api-key",
        dest="openai_api_key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="API key for OpenAI provider",
    )
    parser.add_argument(
        "--fireworks-api-key",
        dest="fireworks_api_key",
        default=os.environ.get("FIREWORKS_API_KEY"),
        help="API key for Fireworks provider",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder_vectors, folder_contexts = load_folder_data(args.root)
    llm = get_llm(
        args.model,
        args.provider,
        openai_api_key=args.openai_api_key,
        fireworks_api_key=args.fireworks_api_key,
    )
    mapping = map_files(
        args.input,
        folder_vectors,
        folder_contexts,
        llm,
        top_n=args.top_n,
        min_similarity=args.min_sim,
    )
    mapping_file = os.path.join(args.input, "file_mappings.json")
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    if args.apply:
        apply_mapping(mapping)
    print(f"Saved mapping to {mapping_file}")


if __name__ == "__main__":
    main()

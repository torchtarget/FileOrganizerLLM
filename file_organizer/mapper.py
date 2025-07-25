import os
import json
import argparse
import shutil
from typing import Dict

from .organizer import get_embedding, extract_text_file


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


def load_folder_data(root: str) -> tuple[Dict[str, list], Dict[str, str]]:
    """Load folder vectors and contexts from the organizer output."""
    vectors_path = os.path.join(root, "folder_vectors.json")
    contexts_path = os.path.join(root, "folder_contexts.json")
    with open(vectors_path, "r", encoding="utf-8") as f:
        vectors: Dict[str, list] = json.load(f)
    with open(contexts_path, "r", encoding="utf-8") as f:
        contexts: Dict[str, str] = json.load(f)
    return vectors, contexts


def suggest_folder_for_file(
    filepath: str,
    folder_vectors: Dict[str, list],
) -> str:
    """Return the folder with the highest embedding similarity."""
    text = extract_text_file(filepath, n_chars=2000)
    embedding = get_embedding(text)
    best_folder = list(folder_vectors.keys())[0]
    best_score = float("-inf")
    for folder, vec in folder_vectors.items():
        score = cosine_similarity(embedding, vec)
        if score > best_score:
            best_score = score
            best_folder = folder
    return best_folder


def map_files(source: str, folder_vectors: Dict[str, list]) -> Dict[str, str]:
    """Map each file in ``source`` recursively to a destination folder."""
    mapping: Dict[str, str] = {}
    for root, _, files in os.walk(source):
        for name in files:
            path = os.path.join(root, name)
            dest = suggest_folder_for_file(path, folder_vectors)
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
        required=True,
        help="Root folder that contains folder_vectors.json and folder_contexts.json",
    )
    parser.add_argument(
        "--apply", action="store_true", help="Move files based on generated mapping"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder_vectors, folder_contexts = load_folder_data(args.root)
    mapping = map_files(args.input, folder_vectors)
    mapping_file = os.path.join(args.input, "file_mappings.json")
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    if args.apply:
        apply_mapping(mapping)
    print(f"Saved mapping to {mapping_file}")


if __name__ == "__main__":
    main()

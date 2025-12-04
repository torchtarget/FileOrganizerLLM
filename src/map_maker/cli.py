from __future__ import annotations

import argparse
from pathlib import Path

from .config import BuilderSettings
from .llm import build_llm
from .traversal import PersonaBuilder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic folder persona builder (Map Maker)")
    parser.add_argument("root", type=Path, help="Root directory to index")
    parser.add_argument(
        "--provider",
        choices=["stub", "fireworks", "ollama"],
        default="stub",
        help="LLM provider to use",
    )
    parser.add_argument(
        "--follow-symlinks",
        action="store_true",
        help="Allow following symlinks (disabled by default for safety)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process child folders in parallel where possible",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    root_path: Path = args.root.resolve()
    settings = BuilderSettings(
        root_path=root_path,
        provider=args.provider,
        follow_symlinks=args.follow_symlinks,
        allow_parallel=args.parallel,
    )

    llm = build_llm(args.provider)
    builder = PersonaBuilder(settings=settings, llm=llm)
    persona = builder.build_for_root(root_path)
    print(persona.model_dump_json(indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()

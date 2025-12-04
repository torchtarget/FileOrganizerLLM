from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import BuilderSettings
from .config_loader import load_config, MapMakerConfig
from .database import PersonaDatabase
from .llm import build_llm
from .traversal import PersonaBuilder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic folder persona builder (Map Maker)")
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        help="Root directory to index (can also be set in config file)",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Path to config.yaml file (default: ./config.yaml)",
    )
    parser.add_argument(
        "--provider",
        choices=["stub", "fireworks", "ollama"],
        help="LLM provider to use (overrides config file)",
    )
    parser.add_argument(
        "--follow-symlinks",
        action="store_true",
        help="Allow following symlinks (overrides config file)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process child folders in parallel (overrides config file)",
    )
    parser.add_argument(
        "--generate-config",
        action="store_true",
        help="Generate a default config.yaml file and exit",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("map_maker.db"),
        help="Path to SQLite database file (default: map_maker.db in current directory)",
    )
    parser.add_argument(
        "--export-json",
        action="store_true",
        help="After processing, export personas to folder_persona.json files in each directory",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show database statistics after processing",
    )
    return parser


def generate_config_file(output_path: Path) -> None:
    """Generate a default config.yaml file"""
    if output_path.exists():
        response = input(f"{output_path} already exists. Overwrite? (y/N): ")
        if response.lower() != "y":
            print("Aborted.")
            return

    from .config_loader import create_default_config

    create_default_config(output_path)
    print(f"Default config file created at: {output_path}")
    print("Edit this file with your settings and run: map-maker --config config.yaml")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Handle config generation
    if args.generate_config:
        output_path = args.config or Path("config.yaml")
        generate_config_file(output_path)
        return

    # Load config from file if specified
    config: MapMakerConfig | None = None
    if args.config:
        config = load_config(args.config)
    elif Path("config.yaml").exists():
        print("Using config.yaml from current directory")
        config = load_config(Path("config.yaml"))

    # Determine settings
    if config:
        # Use config file values, allow CLI args to override
        root_path = Path(args.root) if args.root else Path(config.root_path)
        provider = args.provider or config.llm.provider
        follow_symlinks = args.follow_symlinks or config.processing.follow_symlinks
        allow_parallel = args.parallel or config.processing.allow_parallel

        # Set environment variables from config if needed
        if config.llm.api_key:
            if provider == "fireworks":
                os.environ["FIREWORKS_API_KEY"] = config.llm.api_key
        if config.llm.model:
            if provider == "fireworks":
                os.environ["FIREWORKS_MODEL"] = config.llm.model
            elif provider == "ollama":
                os.environ["OLLAMA_MODEL"] = config.llm.model
        if config.llm.host and provider == "ollama":
            os.environ["OLLAMA_HOST"] = config.llm.host

        # Update global config values
        from . import config as global_config

        global_config.MIN_TEXT_FILES = config.processing.min_text_files
        global_config.SAMPLE_LIMIT = config.processing.sample_limit
        global_config.SAMPLE_BYTES = config.processing.sample_bytes
        global_config.ROOT_CONSTRAINTS = config.root_constraints_dict
        global_config.SCHEMA_VERSION = config.schema_version
        global_config.DEFAULT_LANGUAGE = config.default_language
        global_config.DEFAULT_CONFIDENCE = config.default_confidence
    else:
        # Use CLI args only (backward compatibility)
        if not args.root:
            parser.error("root directory is required when not using a config file")
        root_path = args.root
        provider = args.provider or "stub"
        follow_symlinks = args.follow_symlinks
        allow_parallel = args.parallel

    root_path = root_path.resolve()

    settings = BuilderSettings(
        root_path=root_path,
        provider=provider,
        follow_symlinks=follow_symlinks,
        allow_parallel=allow_parallel,
    )

    # Initialize database
    db_path = args.db.resolve()
    print(f"Using database: {db_path}")

    with PersonaDatabase(db_path) as db:
        llm = build_llm(provider)
        builder = PersonaBuilder(settings=settings, llm=llm, database=db)

        print(f"Processing: {root_path}")
        print("\n=== PASS 1: Bottom-Up (Building from children to parent) ===")
        persona = builder.build_for_root(root_path)
        print("Pass 1 complete.")

        print("\n=== PASS 2: Top-Down (Applying parent constraints) ===")
        builder.refine_with_parent_constraints(root_path)
        print("Pass 2 complete.")

        # Reload root persona after refinement
        persona = db.load_persona(str(root_path))

        # Show stats if requested
        if args.stats:
            stats = db.get_stats()
            print("\n=== Database Statistics ===")
            print(f"Total folders processed: {stats['total_folders']}")
            print(f"Node type breakdown: {stats['breakdown']}")

        # Export to JSON files if requested
        if args.export_json:
            print("\nExporting personas to JSON files...")
            count = db.export_to_json_files(root_path)
            print(f"Exported {count} persona files")

        print("\n=== Root Persona ===")
        print(persona.model_dump_json(indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml


@dataclass
class LLMConfig:
    """Configuration for LLM provider"""
    provider: str = "stub"
    api_key: Optional[str] = None
    model: Optional[str] = None
    host: Optional[str] = None


@dataclass
class ProcessingConfig:
    """Configuration for processing options"""
    follow_symlinks: bool = False
    allow_parallel: bool = False
    min_text_files: int = 5
    sample_limit: int = 10
    sample_bytes: int = 2048


@dataclass
class RootConstraints:
    """Semantic constraints for root folders"""
    business: str = (
        "Strictly commercial, financial, legal, strategic, and operational content. "
        "EXCLUDES: personal, family, domestic, medical, hobby, intimate, or unrelated materials."
    )
    private: str = (
        "Strictly personal, family, health, education, hobbies, and private financial documents. "
        "EXCLUDES: corporate, client, revenue-generating, or organizational materials."
    )


@dataclass
class MapMakerConfig:
    """Complete Map Maker configuration"""
    root_path: str
    llm: LLMConfig
    processing: ProcessingConfig
    root_constraints: RootConstraints
    schema_version: str = "1.1"
    default_language: str = "en"
    default_confidence: float = 0.82

    @property
    def root_constraints_dict(self) -> Dict[str, str]:
        """Convert root_constraints to dictionary format"""
        return {
            "Business": self.root_constraints.business,
            "Private": self.root_constraints.private,
        }


def load_config(config_path: Path) -> MapMakerConfig:
    """Load configuration from YAML file"""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Parse LLM config
    llm_data = data.get("llm", {})
    llm_config = LLMConfig(
        provider=llm_data.get("provider", "stub"),
        api_key=llm_data.get("api_key") or os.getenv(llm_data.get("api_key_env", "")),
        model=llm_data.get("model"),
        host=llm_data.get("host"),
    )

    # Parse processing config
    proc_data = data.get("processing", {})
    processing_config = ProcessingConfig(
        follow_symlinks=proc_data.get("follow_symlinks", False),
        allow_parallel=proc_data.get("allow_parallel", False),
        min_text_files=proc_data.get("min_text_files", 5),
        sample_limit=proc_data.get("sample_limit", 10),
        sample_bytes=proc_data.get("sample_bytes", 2048),
    )

    # Parse root constraints
    constraints_data = data.get("root_constraints", {})
    root_constraints = RootConstraints(
        business=constraints_data.get("business", RootConstraints.business),
        private=constraints_data.get("private", RootConstraints.private),
    )

    # Create main config
    config = MapMakerConfig(
        root_path=data.get("root_path", ""),
        llm=llm_config,
        processing=processing_config,
        root_constraints=root_constraints,
        schema_version=data.get("schema_version", "1.1"),
        default_language=data.get("default_language", "en"),
        default_confidence=data.get("default_confidence", 0.82),
    )

    return config


def create_default_config(output_path: Path) -> None:
    """Create a default config.yaml file"""
    default_config = {
        "# Map Maker Configuration": None,
        "# Specify the root path to index": None,
        "root_path": "/path/to/your/NAS",
        "": None,
        "# LLM Provider Configuration": None,
        "llm": {
            "# Options: stub, fireworks, ollama": None,
            "provider": "stub",
            "": None,
            "# For Fireworks.ai": None,
            "# api_key_env: FIREWORKS_API_KEY  # or hardcode: api_key: your-key-here": None,
            "# model: accounts/fireworks/models/llama-v3-70b-instruct": None,
            "  ": None,
            "# For Ollama (local)": None,
            "# host: http://localhost:11434": None,
            "# model: llama3": None,
        },
        "  ": None,
        "# Processing Options": None,
        "processing": {
            "follow_symlinks": False,
            "allow_parallel": False,
            "min_text_files": 5,
            "sample_limit": 10,
            "sample_bytes": 2048,
        },
        "   ": None,
        "# Semantic Constraints": None,
        "root_constraints": {
            "business": "Strictly commercial, financial, legal, strategic, and operational content. EXCLUDES: personal, family, domestic, medical, hobby, intimate, or unrelated materials.",
            "private": "Strictly personal, family, health, education, hobbies, and private financial documents. EXCLUDES: corporate, client, revenue-generating, or organizational materials.",
        },
        "    ": None,
        "# Schema Settings": None,
        "schema_version": "1.1",
        "default_language": "en",
        "default_confidence": 0.82,
    }

    # Clean up the None values used for comments
    clean_config = {k: v for k, v in default_config.items() if v is not None and not k.strip() == ""}

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Map Maker Configuration File\n")
        f.write("# Generated default configuration\n\n")
        f.write("# Specify the root path to index\n")
        f.write("root_path: /path/to/your/NAS\n\n")
        f.write("# LLM Provider Configuration\n")
        f.write("llm:\n")
        f.write("  # Options: stub, fireworks, ollama\n")
        f.write("  provider: stub\n\n")
        f.write("  # For Fireworks.ai (uncomment and configure):\n")
        f.write("  # api_key_env: FIREWORKS_API_KEY  # reads from environment variable\n")
        f.write("  # api_key: your-key-here          # or hardcode (not recommended)\n")
        f.write("  # model: accounts/fireworks/models/llama-v3-70b-instruct\n\n")
        f.write("  # For Ollama (local LLM, uncomment and configure):\n")
        f.write("  # host: http://localhost:11434\n")
        f.write("  # model: llama3\n\n")
        f.write("# Processing Options\n")
        f.write("processing:\n")
        f.write("  follow_symlinks: false\n")
        f.write("  allow_parallel: false\n")
        f.write("  min_text_files: 5\n")
        f.write("  sample_limit: 10\n")
        f.write("  sample_bytes: 2048\n\n")
        f.write("# Semantic Constraints for Root Folders\n")
        f.write("root_constraints:\n")
        f.write("  business: \"Strictly commercial, financial, legal, strategic, and operational content. EXCLUDES: personal, family, domestic, medical, hobby, intimate, or unrelated materials.\"\n")
        f.write("  private: \"Strictly personal, family, health, education, hobbies, and private financial documents. EXCLUDES: corporate, client, revenue-generating, or organizational materials.\"\n\n")
        f.write("# Schema Settings\n")
        f.write("schema_version: \"1.1\"\n")
        f.write("default_language: en\n")
        f.write("default_confidence: 0.82\n")

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from .schema import FolderPersona


class PersonaDatabase:
    """SQLite database for storing folder personas"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Create database and tables if they don't exist"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS folder_personas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                node_type TEXT NOT NULL,
                depth INTEGER NOT NULL,
                structural_hash TEXT,
                persona_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create index on path for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_path ON folder_personas(path)
        """)

        # Create index on structural_hash for faster comparison
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_hash ON folder_personas(structural_hash)
        """)

        self.conn.commit()

    def save_persona(self, persona: FolderPersona) -> None:
        """Save or update a folder persona"""
        if not self.conn:
            raise RuntimeError("Database connection not initialized")

        cursor = self.conn.cursor()

        persona_json = persona.model_dump_json(indent=2)

        cursor.execute("""
            INSERT INTO folder_personas (path, node_type, depth, structural_hash, persona_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                node_type = excluded.node_type,
                depth = excluded.depth,
                structural_hash = excluded.structural_hash,
                persona_json = excluded.persona_json,
                updated_at = CURRENT_TIMESTAMP
        """, (
            persona.meta.path,
            persona.meta.node_type.value,
            persona.meta.depth,
            persona.meta.structural_hash,
            persona_json
        ))

        self.conn.commit()

    def load_persona(self, path: str) -> Optional[FolderPersona]:
        """Load a folder persona by path"""
        if not self.conn:
            raise RuntimeError("Database connection not initialized")

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT persona_json FROM folder_personas WHERE path = ?
        """, (path,))

        row = cursor.fetchone()
        if not row:
            return None

        try:
            data = json.loads(row["persona_json"])
            return FolderPersona.model_validate(data)
        except Exception:
            return None

    def get_all_personas(self) -> list[FolderPersona]:
        """Get all folder personas from the database"""
        if not self.conn:
            raise RuntimeError("Database connection not initialized")

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT persona_json FROM folder_personas ORDER BY path
        """)

        personas = []
        for row in cursor.fetchall():
            try:
                data = json.loads(row["persona_json"])
                personas.append(FolderPersona.model_validate(data))
            except Exception:
                continue

        return personas

    def export_to_json_files(self, base_path: Optional[Path] = None) -> int:
        """
        Export all personas to folder_persona.json files in their respective directories.
        Returns the number of files written.
        """
        if not self.conn:
            raise RuntimeError("Database connection not initialized")

        personas = self.get_all_personas()
        written_count = 0

        for persona in personas:
            try:
                folder_path = Path(persona.meta.path)
                if base_path and not str(folder_path).startswith(str(base_path)):
                    continue

                output_path = folder_path / "folder_persona.json"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                persona.write(output_path)
                written_count += 1
            except Exception:
                # Skip if we can't write to this location
                continue

        return written_count

    def get_stats(self) -> dict:
        """Get database statistics"""
        if not self.conn:
            raise RuntimeError("Database connection not initialized")

        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) as total FROM folder_personas")
        total = cursor.fetchone()["total"]

        cursor.execute("SELECT COUNT(DISTINCT node_type) as types FROM folder_personas")
        types = cursor.fetchone()["types"]

        cursor.execute("SELECT node_type, COUNT(*) as count FROM folder_personas GROUP BY node_type")
        by_type = {row["node_type"]: row["count"] for row in cursor.fetchall()}

        return {
            "total_folders": total,
            "node_types": types,
            "breakdown": by_type
        }

    def close(self) -> None:
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

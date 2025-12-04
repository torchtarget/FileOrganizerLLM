#!/usr/bin/env python3
"""
Simple helper tool to view folder paths and their semantic descriptions from the database.
Usage: python view_folders.py [path_to_database.db] [options]
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path


def view_folder_descriptions(db_path: str, show_queries: bool = False,
                            show_constraints: bool = False, show_derived: bool = False,
                            show_meta: bool = False, show_full: bool = False):
    """Display folder paths and their semantic descriptions"""

    if not Path(db_path).exists():
        print(f"Error: Database not found at {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT path, persona_json FROM folder_personas ORDER BY path
    """)

    rows = cursor.fetchall()

    if not rows:
        print("No folder personas found in database.")
        conn.close()
        return

    # If --full is specified, show everything
    if show_full:
        show_queries = show_constraints = show_derived = show_meta = True

    print(f"\n{'='*100}")
    print(f"FOLDER PERSONAS")
    print(f"{'='*100}")

    for path, persona_json in rows:
        try:
            data = json.loads(persona_json)
            persona = data.get("persona", {})
            meta = data.get("meta", {})
            constraints = data.get("constraints", {})
            vector_data = data.get("vector_data", {})

            short_label = persona.get("short_label", "N/A")
            description = persona.get("description", "N/A")

            print(f"\nPATH: {path}")
            print(f"  LABEL: {short_label}")
            print(f"  DESC:  {description}")

            # Show metadata
            if show_meta:
                node_type = meta.get("node_type", "N/A")
                depth = meta.get("depth", "N/A")
                confidence = meta.get("confidence", "N/A")
                language = meta.get("language", "N/A")
                print(f"   META: Type={node_type}, Depth={depth}, Confidence={confidence}, Language={language}")

            # Show what it was derived from
            if show_derived:
                derived_from = persona.get("derived_from", [])
                if derived_from:
                    print(f"   DERIVED FROM: {', '.join(derived_from)}")

            # Show constraints
            if show_constraints:
                parent_constraint = constraints.get("parent_constraint")
                if parent_constraint:
                    print(f"   PARENT CONSTRAINT: {parent_constraint[:100]}...")

                negative_constraints = persona.get("negative_constraints", [])
                if negative_constraints:
                    print(f"   NEGATIVE CONSTRAINTS: {', '.join(negative_constraints)}")

            # Show hypothetical queries
            if show_queries:
                queries = vector_data.get("hypothetical_user_queries", [])
                if queries:
                    print(f"   HYPOTHETICAL QUERIES:")
                    for i, query in enumerate(queries, 1):
                        print(f"      {i}. {query}")

                embedding_model = vector_data.get("embedding_model")
                has_embedding = vector_data.get("embedding") is not None
                if embedding_model:
                    emb_status = "YES" if has_embedding else "NO"
                    print(f"   EMBEDDING: {emb_status} ({embedding_model})")

            print(f"{'-'*100}")

        except Exception as e:
            print(f"\nPATH: {path}")
            print(f"  ERROR: {e}")
            print(f"{'-'*100}")

    conn.close()
    print(f"\nTotal folders: {len(rows)}\n")


def main():
    parser = argparse.ArgumentParser(
        description="View folder paths and semantic descriptions from Map Maker database"
    )
    parser.add_argument(
        "database",
        nargs="?",
        default="map_maker.db",
        help="Path to SQLite database file (default: map_maker.db)"
    )
    parser.add_argument(
        "-q", "--queries",
        action="store_true",
        help="Show hypothetical user queries and embedding info"
    )
    parser.add_argument(
        "-c", "--constraints",
        action="store_true",
        help="Show parent and negative constraints"
    )
    parser.add_argument(
        "-d", "--derived",
        action="store_true",
        help="Show what child folders this was derived from"
    )
    parser.add_argument(
        "-m", "--meta",
        action="store_true",
        help="Show metadata (node type, depth, confidence, language)"
    )
    parser.add_argument(
        "-f", "--full",
        action="store_true",
        help="Show all available information"
    )

    args = parser.parse_args()

    view_folder_descriptions(
        args.database,
        show_queries=args.queries,
        show_constraints=args.constraints,
        show_derived=args.derived,
        show_meta=args.meta,
        show_full=args.full
    )


if __name__ == "__main__":
    main()

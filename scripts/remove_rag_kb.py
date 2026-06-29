"""
remove_rag_kb.py — Delete a knowledge base (Chroma collection + ragbase folder).

Usage:
    source langgraph_env/bin/activate
    python scripts/remove_rag_kb.py ./ragbase/rnai_records
    python scripts/remove_rag_kb.py ./ragbase/rnai_records --yes   # skip confirmation

Reads metadata.md for the Chroma collection name, deletes the collection,
cleans up orphaned UUID folders in chroma_db/, then removes the ragbase folder.
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("remove_rag")

# ============================================================
# Paths (self-contained — no tool_config dependency)
# ============================================================
import os
from dotenv import load_dotenv
load_dotenv()

RAG_CHROMA_DIR = Path(os.getenv(
    "RAG_CHROMA_DIR",
    _PROJECT_ROOT / "ragbase" / "chroma_db",
))


def delete_kb(kb_path_str: str, auto_yes: bool = False):
    """Delete Chroma collection + ragbase folder for a knowledge base."""
    kb_path = Path(kb_path_str).resolve()

    # ---- Validate ----
    if not kb_path.exists():
        logger.error("Folder not found: %s", kb_path)
        sys.exit(1)
    if not kb_path.is_dir():
        logger.error("Not a directory: %s", kb_path)
        sys.exit(1)

    md_path = kb_path / "metadata.md"
    if not md_path.exists():
        logger.error("Not a valid KB folder (metadata.md missing): %s", kb_path)
        sys.exit(1)

    # ---- Read collection name ----
    import yaml
    raw = md_path.read_text(encoding="utf-8")
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            raw = parts[1]
    meta = yaml.safe_load(raw)
    coll_name = meta.get("name", kb_path.name) if isinstance(meta, dict) else kb_path.name

    # ---- Confirm ----
    if not auto_yes:
        print(f"⚠️  About to delete KB '{coll_name}' ({kb_path})")
        print(f"   Will remove: Chroma collection + ragbase folder")
        print(f"   This cannot be undone. Continue? (y/N): ", end="", flush=True)
        try:
            answer = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer != "y":
            logger.info("User cancelled deletion")
            return

    # ---- Delete Chroma collection ----
    if RAG_CHROMA_DIR.exists():
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            client = chromadb.PersistentClient(
                path=str(RAG_CHROMA_DIR),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            client.delete_collection(coll_name)
            logger.info("✅ Chroma collection '%s' deleted", coll_name)
        except Exception as e:
            logger.warning("Failed to delete Chroma collection (may not exist): %s", e)

        # ---- Clean up orphaned UUID folders ----
        try:
            import sqlite3
            conn = sqlite3.connect(str(RAG_CHROMA_DIR / "chroma.sqlite3"))
            cursor = conn.execute(
                "SELECT DISTINCT id FROM segments WHERE id IS NOT NULL"
            )
            active_uuids = {row[0] for row in cursor.fetchall()}
            conn.close()

            cleaned = 0
            for entry in sorted(RAG_CHROMA_DIR.iterdir()):
                if entry.is_dir() and len(entry.name) == 36 and "-" in entry.name:
                    if entry.name not in active_uuids:
                        shutil.rmtree(entry)
                        cleaned += 1
            if cleaned:
                logger.info("✅ Cleaned up %d orphaned UUID folders", cleaned)
        except Exception as e:
            logger.warning("UUID folder cleanup failed: %s", e)
    else:
        logger.warning("Chroma DB directory not found: %s", RAG_CHROMA_DIR)

    # ---- Delete ragbase folder ----
    shutil.rmtree(kb_path)
    logger.info("✅ Folder deleted: %s", kb_path)

    print(f"\nKB '{coll_name}' fully deleted.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/remove_rag_kb.py <kb_folder_path> [--yes]")
        print("Example: python scripts/remove_rag_kb.py ./ragbase/rnai_records")
        sys.exit(1)

    auto_yes = "--yes" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--yes"]
    if not args:
        print("Usage: python scripts/remove_rag_kb.py <kb_folder_path> [--yes]")
        sys.exit(1)

    delete_kb(args[0], auto_yes)


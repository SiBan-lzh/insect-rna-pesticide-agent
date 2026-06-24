"""
save_memory/tool.py — Save information to long-term memory.

Persists key-value records into a ChromaDB collection (agent_memory).
Records are grouped by slot (arbitrary category string) and key (unique ID).
Slot is fully flexible — any string is accepted (e.g. preferences, recent_work,
task_progress, user-defined categories).

ChromaDB path: memory/chroma_db/
Collection:    agent_memory
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_DIR = _PROJECT_ROOT / "memory" / "chroma_db"
COLLECTION_NAME = "agent_memory"

logger = logging.getLogger("RPA_Tools.SaveMemory")

# ============================================================
# Lazy ChromaDB client
# ============================================================
_client = None
_collection = None


def _get_collection():
    """Lazy-init ChromaDB persistent client and collection."""
    global _client, _collection
    if _collection is not None:
        return _collection

    import chromadb
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    _collection = _client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


# ============================================================
# Pydantic input schema
# ============================================================
class SaveMemoryInput(BaseModel):
    """Memory save parameters."""

    slot: str = Field(
        description="Category label for the memory (e.g. 'preferences', 'recent_work', or any custom category)"
    )
    key: str = Field(
        description="Unique identifier for this record within the slot"
    )
    value: str = Field(
        description="Content to remember"
    )


# ============================================================
# LangChain Tool
# ============================================================
class SaveMemoryTool(BaseTool):
    """Save a piece of information to long-term memory."""

    name: str = "save_memory"
    description: str = (
        "Save a piece of information to long-term memory. "
        "The memory is persisted in a vector database and can be "
        "retrieved later with load_memory. "
        "Use slot to group related information (e.g. 'preferences', "
        "'recent_work', 'task_progress', or any custom category). "
        "Use key as a unique identifier within that slot. "
        "Same slot+key overwrites the previous value."
    )
    args_schema: type = SaveMemoryInput

    def _run(self, slot: str, key: str, value: str) -> str:
        """Save a memory record to ChromaDB."""
        try:
            collection = _get_collection()
        except Exception as exc:
            logger.exception("Failed to initialize ChromaDB")
            return f"ERROR: Memory system unavailable: {exc}"

        doc_id = f"{slot}::{key}"
        metadata = {
            "slot": slot,
            "key": key,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # upsert: insert or update if same id exists
            collection.upsert(
                ids=[doc_id],
                documents=[value],
                metadatas=[metadata],
            )
            logger.info("Saved memory: slot=%s key=%s", slot, key)
            return f"Saved to slot '{slot}' with key '{key}'."
        except Exception as exc:
            logger.exception("ChromaDB upsert failed")
            return f"ERROR: Failed to save memory: {exc}"


# ============================================================
# Singleton export
# ============================================================
save_memory_tool = SaveMemoryTool()

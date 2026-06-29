"""
save_memory/tool.py — Save information to long-term memory.

Persists key-value records into a ChromaDB collection (agent_memory).
Each memory_name maps to a separate physical directory under memory/,
providing topic isolation between different projects.

ChromaDB path: memory/<memory_name>/
Collection:    agent_memory
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_COLLECTION_NAME = "agent_memory"

def _safe_memory_dir(memory_name: str) -> Path:
    """Sanitize memory_name and return the corresponding directory."""
    safe = re.sub(r'[^a-zA-Z0-9_-]', '_', memory_name)
    return _PROJECT_ROOT / "memory" / safe

logger = logging.getLogger("RPA_Tools.SaveMemory")

# ============================================================
# ChromaDB client cache (one client per memory_name)
# ============================================================
_clients: dict[str, tuple] = {}  # memory_name -> (client, collection)


def _get_collection(memory_name: str):
    """Lazy-init ChromaDB persistent client for the given memory_name."""
    if memory_name in _clients:
        return _clients[memory_name][1]

    import chromadb
    mem_dir = _safe_memory_dir(memory_name)
    mem_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(mem_dir))
    collection = client.get_or_create_collection(name=_COLLECTION_NAME)
    _clients[memory_name] = (client, collection)
    return collection


# ============================================================
# Pydantic input schema
# ============================================================
class SaveMemoryInput(BaseModel):
    """Memory save parameters."""

    memory_name: str = Field(
        description="Topic label for physical memory isolation. "
        "Use a short, meaningful name based on the current topic "
        "(e.g. 'target-spodoptera', 'dsrna-bemisia', 'preferences'). "
        "Each name creates a separate storage directory under memory/."
    )
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
        "Uses memory_name for topic isolation — each name creates a separate "
        "storage directory. Use slot to group related information "
        "(e.g. 'preferences', 'recent_work', 'task_progress'). "
        "Use key as a unique identifier within that slot. "
        "Same memory_name+slot+key overwrites the previous value."
    )
    args_schema: type = SaveMemoryInput

    def _run(self, memory_name: str, slot: str, key: str, value: str) -> str:
        """Save a memory record to ChromaDB."""
        if not memory_name or not memory_name.strip():
            return '{"status": "error", "error": "memory_name is required."}'

        try:
            collection = _get_collection(memory_name)
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
            collection.upsert(
                ids=[doc_id],
                documents=[value],
                metadatas=[metadata],
            )
            logger.info("Saved memory: name=%s slot=%s key=%s", memory_name, slot, key)
            return f"Saved to memory_name='{memory_name}', slot '{slot}' with key '{key}'."
        except Exception as exc:
            logger.exception("ChromaDB upsert failed")
            return f"ERROR: Failed to save memory: {exc}"


# ============================================================
# Singleton export
# ============================================================
save_memory_tool = SaveMemoryTool()

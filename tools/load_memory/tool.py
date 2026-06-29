"""
load_memory/tool.py — Load information from long-term memory.

Retrieves records from ChromaDB (agent_memory collection).
Each memory_name maps to a separate physical directory under memory/,
providing topic isolation between different projects.

ChromaDB path: memory/<memory_name>/
Collection:    agent_memory
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_COLLECTION_NAME = "agent_memory"


def _safe_memory_dir(memory_name: str) -> Path:
    """Sanitize memory_name and return the corresponding directory."""
    safe = re.sub(r'[^a-zA-Z0-9_-]', '_', memory_name)
    return _PROJECT_ROOT / "memory" / safe

logger = logging.getLogger("RPA_Tools.LoadMemory")

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
class LoadMemoryInput(BaseModel):
    """Memory retrieval parameters."""

    memory_name: str = Field(
        description="Topic label identifying which storage directory to read from. "
        "Must match the memory_name used when saving (e.g. 'target-spodoptera', "
        "'dsrna-bemisia', 'preferences')."
    )
    slot: Optional[str] = Field(
        default=None,
        description="Filter by slot category. If omitted, returns records from all slots."
    )
    query: str = Field(
        default="",
        description="Natural language query for semantic search. If empty, returns all matching records."
    )


# ============================================================
# LangChain Tool
# ============================================================
class LoadMemoryTool(BaseTool):
    """Load information from long-term memory."""

    name: str = "load_memory"
    description: str = (
        "Load information from long-term memory. "
        "Requires memory_name to identify which topic storage to read from "
        "(must match the memory_name used when saving). "
        "Optionally filter by slot (e.g. 'preferences', 'recent_work') "
        "and/or use a natural language query for semantic search. "
        "Returns a formatted summary of matching memory records."
    )
    args_schema: type = LoadMemoryInput

    def _run(self, memory_name: str, slot: Optional[str] = None, query: str = "") -> str:
        """Load memory records from ChromaDB."""
        if not memory_name or not memory_name.strip():
            return '{"status": "error", "error": "memory_name is required."}'

        try:
            collection = _get_collection(memory_name)
        except Exception as exc:
            logger.exception("Failed to initialize ChromaDB")
            return f"ERROR: Memory system unavailable: {exc}"

        # Build metadata filter
        where = None
        if slot:
            where = {"slot": slot}

        try:
            n_results = 20
            if query and query.strip():
                # Semantic search
                results = collection.query(
                    query_texts=[query.strip()],
                    where=where,
                    n_results=n_results,
                )
            else:
                # Fetch all (with optional slot filter)
                results = collection.get(
                    where=where,
                    limit=n_results,
                )
        except Exception as exc:
            logger.exception("ChromaDB query failed")
            return f"ERROR: Failed to load memory: {exc}"

        # Format results
        ids = results.get("ids", [[]]) if isinstance(results.get("ids"), list) else []
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        documents = results.get("documents", [[]]) if isinstance(results.get("documents"), list) else []
        if isinstance(documents, list) and documents and isinstance(documents[0], list):
            documents = documents[0]
        metadatas = results.get("metadatas", [[]]) if isinstance(results.get("metadatas"), list) else []
        if isinstance(metadatas, list) and metadatas and isinstance(metadatas[0], list):
            metadatas = metadatas[0]

        if not ids:
            if slot:
                return f"No memories found in slot '{slot}'."
            return "No memories found."

        lines = [f"Memory records ({len(ids)}):", ""]
        for i in range(len(ids)):
            meta = metadatas[i] if i < len(metadatas) else {}
            doc = documents[i] if i < len(documents) else ""
            slot_label = meta.get("slot", "?")
            key_label = meta.get("key", "?")
            ts = meta.get("timestamp", "")
            lines.append(f"  [{slot_label}] {key_label}")
            if ts:
                lines.append(f"    saved: {ts}")
            lines.append(f"    value: {doc}")
            lines.append("")

        return "\n".join(lines)


# ============================================================
# Singleton export
# ============================================================
load_memory_tool = LoadMemoryTool()

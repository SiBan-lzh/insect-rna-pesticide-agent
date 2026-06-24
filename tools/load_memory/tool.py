"""
load_memory/tool.py — Load information from long-term memory.

Retrieves records from ChromaDB (agent_memory collection).
Supports optional slot filtering and semantic search via natural language query.
When query is empty, returns all records for the given slot (or all slots).

ChromaDB path: memory/chroma_db/
Collection:    agent_memory
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_DIR = _PROJECT_ROOT / "memory" / "chroma_db"
COLLECTION_NAME = "agent_memory"

logger = logging.getLogger("RPA_Tools.LoadMemory")

# ============================================================
# Lazy ChromaDB client (shared with save_memory)
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
class LoadMemoryInput(BaseModel):
    """Memory retrieval parameters."""

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
        "Optionally filter by slot (e.g. 'preferences', 'recent_work') "
        "and/or use a natural language query for semantic search. "
        "Returns a formatted summary of matching memory records."
    )
    args_schema: type = LoadMemoryInput

    def _run(self, slot: Optional[str] = None, query: str = "") -> str:
        """Load memory records from ChromaDB."""
        try:
            collection = _get_collection()
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

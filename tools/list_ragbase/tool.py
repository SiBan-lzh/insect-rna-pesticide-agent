"""
list_ragbase.py — Tool to browse available knowledge base metadata.

Usage:
    from tools.list_ragbase import list_ragbase_tool

    result = list_ragbase_tool.invoke({})
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from .config import RAGBASE_ROOT, RAG_CHROMA_DIR

logger = logging.getLogger("RPA_Tools.ListRagbase")


# ============================================================
# KB discovery from metadata.md
# ============================================================
def _discover_knowledge_bases() -> dict[str, dict]:
    """Scan ragbase/ for metadata.md files and return KB metadata.

    Only includes KBs that have been built in Chroma.
    KBs with metadata.md but no Chroma collection are listed as "not built".
    """
    kbs: dict[str, dict] = {}
    if not RAGBASE_ROOT.exists():
        return kbs

    # Check which collections actually exist in Chroma (case-insensitive)
    chroma_collections: dict[str, str] = {}
    if RAG_CHROMA_DIR.exists():
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            client = chromadb.PersistentClient(
                path=str(RAG_CHROMA_DIR),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            for c in client.list_collections():
                chroma_collections[c.name.lower()] = c.name
        except Exception:
            pass

    for entry in sorted(RAGBASE_ROOT.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        if entry.name == "chroma_db":
            continue
        if entry.name == "example":
            continue

        md_path = entry / "metadata.md"
        if not md_path.exists():
            continue

        try:
            import yaml

            raw = md_path.read_text(encoding="utf-8")
            if raw.startswith("---"):
                parts = raw.split("---", 2)
                if len(parts) >= 3:
                    raw = parts[1]
            meta = yaml.safe_load(raw)
        except Exception:
            logger.warning("Failed to parse %s", md_path)
            continue

        if not isinstance(meta, dict) or "name" not in meta:
            continue

        coll_name = meta["name"]
        kbs[coll_name] = {
            "name": coll_name,
            "description": meta.get("description", ""),
            "chunk_strategy": meta.get("chunk_strategy", "general"),
            "chunk_params": meta.get("chunk_params", {}),
            "recommended_search_mode": meta.get("recommended_search_mode", "hybrid"),
            "recommended_semantic_weight": meta.get("recommended_semantic_weight", 0.4),
            "directory": entry.name,
            "ready": coll_name.lower() in chroma_collections,
        }

    return kbs


# Cache KB metadata (refreshed each call)
_KNOWLEDGE_BASES: dict[str, dict] = {}


def refresh_knowledge_bases():
    """Re-scan ragbase/ for new KBs."""
    global _KNOWLEDGE_BASES
    _KNOWLEDGE_BASES = _discover_knowledge_bases()


def list_ragbase() -> str:
    """List available knowledge bases with metadata.

    Returns:
        Formatted KB listing for the LLM.
    """
    refresh_knowledge_bases()

    if not _KNOWLEDGE_BASES:
        return (
            "No knowledge bases found. "
            "Run 'python scripts/build_rag_kb.py' first to build them."
        )

    lines = ["Available knowledge bases:", ""]
    for name, meta in _KNOWLEDGE_BASES.items():
        desc = meta["description"]
        strategy = meta["chunk_strategy"]
        search_mode = meta["recommended_search_mode"]
        ready = meta.get("ready", False)
        status = "Ready" if ready else "Not built yet"
        lines.append(f"  {name}  [{status}]")
        lines.append(f"    Strategy: {strategy} | Recommended: {search_mode}")
        lines.append(f"    {desc[:150]}")
        lines.append("")

    lines.append("Use `search_knowledge` to search a specific knowledge base.")
    return "\n".join(lines)


# ============================================================
# Pydantic input schema
# ============================================================
class ListRagbaseInput(BaseModel):
    """No parameters needed — just lists all KBs."""


class ListRagbaseTool(BaseTool):
    """Tool for listing available knowledge bases."""

    name: str = "list_ragbase"
    description: str = (
        "List available knowledge bases with descriptions, chunk strategies, "
        "and build status. Call this BEFORE using `search_knowledge` to see "
        "which knowledge bases are available."
    )
    args_schema: type[BaseModel] = ListRagbaseInput

    def _run(self, **kwargs) -> str:
        """Run the KB listing."""
        return list_ragbase()

    async def _arun(self, **kwargs) -> str:
        """Async variant."""
        return self._run()


list_ragbase_tool = ListRagbaseTool()
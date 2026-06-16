"""
search_knowledge.py — Hybrid retrieval tool for knowledge bases.

Dynamically discovers available knowledge bases from ragbase/ directory
structure by reading each subdirectory's metadata.md.

The tool description is auto-generated from metadata, so the LLM always
sees the current list of available KBs and their descriptions.

Supported features:
  - Vector search (semantic) via Chroma + BAAI/bge-large-en-v1.5
  - Keyword search (full-text) via BM25
  - Weighted hybrid fusion controlled by semantic_weight parameter
  - Parent-child context retrieval (parent_content attached when available)
  - Dynamic KB discovery via metadata.md

Usage:
    from tools.search_knowledge import search_knowledge_tool

    result = search_knowledge_tool.invoke({
        "query": "Hunchback gene interference efficiency",
        "knowledge_base": "rnai_records",
        "top_k": 5,
    })
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from tool_config import RAGBASE_ROOT, RAG_CHROMA_DIR

logger = logging.getLogger("RPA_Tools.SearchKnowledge")

# ============================================================
# Dynamic KB discovery from metadata.md
# ============================================================
def _discover_knowledge_bases() -> dict[str, dict]:
    """Scan ragbase/ for metadata.md files and return KB metadata.

    Only includes knowledge bases that have been built in Chroma.
    KBs with metadata.md but no Chroma collection are listed as "not built".

    Returns: {collection_name: {description, chunk_strategy, chunk_params, ...}}
    """
    kbs: dict[str, dict] = {}
    if not RAGBASE_ROOT.exists():
        return kbs

    # Check which collections actually exist in Chroma (case-insensitive)
    chroma_collections: dict[str, str] = {}  # lowercase → actual name
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


# Cache KB metadata (refreshed once per import)
_KNOWLEDGE_BASES: dict[str, dict] = _discover_knowledge_bases()


def list_knowledge_bases() -> str:
    """Return a formatted list of available knowledge bases for the LLM.

    This is used both for the tool description and as a fallback when
    the LLM calls with an unknown knowledge_base name.
    """
    if not _KNOWLEDGE_BASES:
        return "No knowledge bases found. Run 'python scripts/build_rag_embeddings.py' first."

    lines = ["Available knowledge bases:", ""]
    for name, meta in _KNOWLEDGE_BASES.items():
        desc = meta["description"]
        strategy = meta["chunk_strategy"]
        search_mode = meta["recommended_search_mode"]
        ready = meta.get("ready", False)
        status = "✅ Ready" if ready else "⚠️ Not built yet"
        lines.append(f"  • {name}  [{status}]")
        lines.append(f"    Strategy: {strategy} | Recommended search: {search_mode}")
        lines.append(f"    {desc[:120]}...")
        lines.append("")
    return "\n".join(lines)


def refresh_knowledge_bases():
    """Re-scan ragbase/ for new KBs (call after adding a new collection)."""
    global _KNOWLEDGE_BASES
    _KNOWLEDGE_BASES = _discover_knowledge_bases()


# ============================================================
# Pydantic input schema
# ============================================================
class SearchKnowledgeInput(BaseModel):
    """Knowledge retrieval parameters."""

    query: str = Field(
        description="Natural language query describing the information needed. "
        "Be specific — include insect species, gene names, or keywords."
    )
    knowledge_base: str = Field(
        description="Which knowledge base to search. "
        "Use 'list' to see available knowledge bases.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top results to return (1-20). "
        "Use 1-3 for specific factual queries, 5+ for exploratory searches.",
    )
    semantic_weight: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Weight for semantic (vector) search vs keyword (BM25) search. "
        "0.0 = keyword only, 1.0 = semantic only. "
        "If not provided, uses the knowledge base's recommended default.",
    )


# ============================================================
# Chroma retriever (lazy-loaded)
# ============================================================
_chroma_client = None
_collections_cache: dict[str, object] = {}


def _get_chroma():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        _chroma_client = chromadb.PersistentClient(
            path=str(RAG_CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _chroma_client


def _get_collection(name: str):
    if name not in _collections_cache:
        client = _get_chroma()
        try:
            _collections_cache[name] = client.get_collection(name)
        except Exception:
            raise ValueError(
                f"Knowledge base '{name}' not found in Chroma. "
                f"Run 'python scripts/build_rag_embeddings.py' to build it first."
            )
    return _collections_cache[name]


# ============================================================
# Embedding model (lazy-loaded, singleton)
# ============================================================
_embed_model = None


def _get_embedder():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        _embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    return _embed_model


# ============================================================
# BM25 index (lazy-loaded per collection)
# ============================================================
_bm25_indices: dict[str, object] = {}
_bm25_docs: dict[str, list[str]] = {}


def _get_bm25(collection_name: str):
    """Build or retrieve BM25 index for a collection."""
    if collection_name in _bm25_indices:
        return _bm25_indices[collection_name], _bm25_docs[collection_name]

    from rank_bm25 import BM25Okapi

    collection = _get_collection(collection_name)
    all_docs = collection.get(include=["documents"])
    documents: list[str] = all_docs.get("documents", [])

    if not documents:
        logger.warning("Collection '%s' has no documents", collection_name)
        return None, []

    tokenized = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized)

    _bm25_indices[collection_name] = bm25
    _bm25_docs[collection_name] = documents
    return bm25, documents


# ============================================================
# Core search function
# ============================================================
def search_knowledge(
    query: str,
    knowledge_base: str = "list",
    top_k: int = 5,
    semantic_weight: Optional[float] = None,
) -> str:
    """Search knowledge bases using hybrid vector + keyword retrieval.

    Args:
        query: Natural language query.
        knowledge_base: Which knowledge base to search. Use "list" to see available KBs.
        top_k: Number of results to return.
        semantic_weight: Weight for semantic search (0.0-1.0). If None, uses KB default.

    Returns:
        JSON string with search results and scores.
    """
    # Refresh KB list in case new KBs were added since import
    refresh_knowledge_bases()
    # ---- Handle "list" command ----
    if knowledge_base == "list" or knowledge_base == "?":
        return json.dumps({
            "status": "success",
            "mode": "list",
            "available_knowledge_bases": {
                name: {
                    "description": meta["description"],
                    "chunk_strategy": meta["chunk_strategy"],
                    "recommended_search_mode": meta["recommended_search_mode"],
                    "recommended_semantic_weight": meta["recommended_semantic_weight"],
                    "ready": meta.get("ready", False),
                }
                for name, meta in _KNOWLEDGE_BASES.items()
            },
            "instruction": (
                "Call this tool again with 'knowledge_base' set to one of the above names "
                "and a specific 'query' to search that knowledge base."
            ),
        }, ensure_ascii=False, indent=2)

    # ---- Validate knowledge_base ----
    if knowledge_base not in _KNOWLEDGE_BASES:
        available = ", ".join(sorted(_KNOWLEDGE_BASES.keys()))
        return json.dumps({
            "status": "error",
            "error": f"Unknown knowledge_base '{knowledge_base}'.",
            "available_knowledge_bases": available,
            "hint": "Call with knowledge_base='list' to see all available knowledge bases.",
        }, ensure_ascii=False, indent=2)

    kb_meta = _KNOWLEDGE_BASES[knowledge_base]

    # Use KB default semantic_weight if not provided
    if semantic_weight is None:
        semantic_weight = kb_meta.get("recommended_semantic_weight", 0.4)

    # ---- Check Chroma DB exists ----
    if not RAG_CHROMA_DIR.exists():
        return json.dumps({
            "status": "error",
            "error": f"Chroma index not found at {RAG_CHROMA_DIR}. "
                     f"Run 'python scripts/build_rag_embeddings.py' first.",
        }, ensure_ascii=False, indent=2)

    # ---- Check collection exists ----
    try:
        collection = _get_collection(knowledge_base)
    except ValueError:
        return json.dumps({
            "status": "error",
            "error": f"Knowledge base '{knowledge_base}' not found in Chroma. "
                     f"Run 'python scripts/build_rag_embeddings.py' to rebuild.",
        }, ensure_ascii=False, indent=2)

    # ---- Step 1: Vector search ----
    embedder = _get_embedder()
    query_emb = embedder.encode([query])[0].tolist()

    # Retrieve at least 10 candidates for hybrid fusion to work meaningfully
    chroma_top_k = max(top_k, 10)
    vector_results = collection.query(
        query_embeddings=[query_emb],
        n_results=chroma_top_k,
        include=["documents", "metadatas", "distances"],
    )

    # ---- Step 2: BM25 keyword search ----
    bm25, bm25_docs = _get_bm25(knowledge_base)
    bm25_scores: dict[int, float] = {}
    if bm25 and bm25_docs:
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        for idx, score in enumerate(scores):
            bm25_scores[idx] = float(score)

    # ---- Step 3: Hybrid fusion ----
    results_map: dict[str, dict] = {}

    # Add vector results
    if vector_results and vector_results.get("documents"):
        for i, doc_text in enumerate(vector_results["documents"][0]):
            # Chroma distance → similarity score (cosine: distance=0 → score=1)
            dist = vector_results["distances"][0][i] if vector_results.get("distances") else 1.0
            vec_score = 1.0 - dist

            # Find index in BM25 docs
            bm25_idx = bm25_docs.index(doc_text) if doc_text in bm25_docs else -1
            kw_score = bm25_scores.get(bm25_idx, 0.0)

            # Normalize BM25 score to 0-1 range
            max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
            kw_norm = kw_score / max_bm25 if max_bm25 > 0 else 0.0

            # Weighted hybrid score
            hybrid = semantic_weight * vec_score + (1 - semantic_weight) * kw_norm

            # Dedup by document content
            doc_key = doc_text[:100]
            if doc_key not in results_map or hybrid > results_map[doc_key]["score"]:
                meta = vector_results["metadatas"][0][i] if vector_results.get("metadatas") else {}
                result_entry = {
                    "content": doc_text,
                    "source": meta.get("source", "unknown"),
                    "section": meta.get("section", ""),
                    "score": round(hybrid, 4),
                    "semantic_score": round(vec_score, 4),
                    "keyword_score": round(kw_norm, 4),
                }
                # Attach parent context if available
                parent_content = meta.get("parent_content", "")
                if parent_content:
                    result_entry["parent_content"] = parent_content
                results_map[doc_key] = result_entry

    # Sort by hybrid score, take top_k
    sorted_results = sorted(results_map.values(), key=lambda x: x["score"], reverse=True)[:top_k]

    if not sorted_results:
        return json.dumps({
            "status": "success",
            "query": query,
            "knowledge_base": knowledge_base,
            "results": [],
            "note": f"No results found in '{knowledge_base}' for query: '{query}'.",
        }, ensure_ascii=False, indent=2)

    # Format output
    output = {
        "status": "success",
        "query": query,
        "knowledge_base": knowledge_base,
        "chunk_strategy": kb_meta.get("chunk_strategy", "general"),
        "semantic_weight_used": semantic_weight,
        "total_results": len(sorted_results),
        "results": sorted_results,
    }

    return json.dumps(output, ensure_ascii=False, indent=2)


# ============================================================
# LangChain Tool
# ============================================================
class SearchKnowledgeTool(BaseTool):
    """Hybrid retrieval tool for indexed knowledge bases."""

    name: str = "search_knowledge"
    description: str = ""
    args_schema: type[BaseModel] = SearchKnowledgeInput

    def _generate_description(self) -> str:
        """Dynamically generate tool description from discovered KBs."""
        if not _KNOWLEDGE_BASES:
            return (
                "Search indexed knowledge bases. No knowledge bases found yet. "
                "Run 'python scripts/build_rag_embeddings.py' first."
            )

        lines = [
            "Search indexed knowledge bases using hybrid semantic + keyword retrieval.",
            "",
            "First, call with knowledge_base='list' (or no knowledge_base) to see available",
            "knowledge bases and their descriptions. Then call again with a specific",
            "knowledge_base name and query.",
            "",
            "Available knowledge bases:",
        ]
        for name, meta in _KNOWLEDGE_BASES.items():
            desc = meta["description"]
            strategy = meta["chunk_strategy"]
            search_mode = meta["recommended_search_mode"]
            ready = meta.get("ready", False)
            status = "✅ Ready" if ready else "⚠️ Not built yet"
            lines.append(f"")
            lines.append(f"  • {name}  [{status}]")
            lines.append(f"    ({strategy} strategy, recommended: {search_mode})")
            # Truncate long descriptions
            short_desc = desc[:150] + "..." if len(desc) > 150 else desc
            lines.append(f"    {short_desc}")

        lines.append("")
        lines.append("Usage: search_knowledge(query='...', knowledge_base='rnai_records', top_k=5)")
        return "\n".join(lines)

    def _run(
        self,
        query: str,
        knowledge_base: str = "list",
        top_k: int = 5,
        semantic_weight: Optional[float] = None,
    ) -> str:
        """Run the knowledge search."""
        # Refresh description to reflect any newly added KBs
        self.description = self._generate_description()
        return search_knowledge(query, knowledge_base, top_k, semantic_weight)

    async def _arun(
        self,
        query: str,
        knowledge_base: str = "list",
        top_k: int = 5,
        semantic_weight: Optional[float] = None,
    ) -> str:
        """Async variant."""
        return self._run(query, knowledge_base, top_k, semantic_weight)


# Create tool instance with dynamic description
search_knowledge_tool = SearchKnowledgeTool()
# Set description after init (cannot be set in __init__ due to BaseModel)
search_knowledge_tool.description = search_knowledge_tool._generate_description()

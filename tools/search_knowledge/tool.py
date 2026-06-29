"""
search_knowledge.py — Hybrid retrieval tool for knowledge bases.

Requires a built Chroma index. Use `list_ragbase` first to see available KBs.

Supported features:
  - Vector search (semantic) via Chroma + BAAI/bge-large-en-v1.5
  - Keyword search (full-text) via BM25
  - Weighted hybrid fusion controlled by semantic_weight parameter
  - Parent-child context retrieval (parent_content attached when available)

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
from pathlib import Path
from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .config import RAGBASE_ROOT, RAG_CHROMA_DIR

logger = logging.getLogger("RPA_Tools.SearchKnowledge")


# ============================================================
# KB metadata lookup (read-only, for parameter defaults)
# ============================================================
def _get_kb_meta(name: str) -> dict | None:
    """Look up a single KB's metadata by collection name."""
    if not RAGBASE_ROOT.exists():
        return None
    for entry in RAGBASE_ROOT.iterdir():
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        if entry.name in ("chroma_db", "example"):
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
            if isinstance(meta, dict) and meta.get("name") == name:
                return meta
        except Exception:
            continue
    return None


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
        "Use `list_ragbase` to see available knowledge bases.",
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
                f"Use `list_ragbase` to see available knowledge bases."
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
    knowledge_base: str,
    top_k: int = 5,
    semantic_weight: Optional[float] = None,
) -> str:
    """Search a knowledge base using hybrid vector + keyword retrieval.

    Args:
        query: Natural language query.
        knowledge_base: Which knowledge base to search.
        top_k: Number of results to return.
        semantic_weight: Weight for semantic search (0.0-1.0). If None, uses KB default.

    Returns:
        JSON string with search results and scores.
    """
    # ---- Check Chroma DB exists ----
    if not RAG_CHROMA_DIR.exists():
        return json.dumps({
            "status": "error",
            "error": f"Chroma index not found at {RAG_CHROMA_DIR}. "
                     f"Run 'python scripts/build_rag_kb.py' first.",
        }, ensure_ascii=False, indent=2)

    # ---- Look up KB metadata for defaults ----
    kb_meta = _get_kb_meta(knowledge_base)

    # ---- Validate collection exists ----
    try:
        collection = _get_collection(knowledge_base)
    except ValueError as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
            "hint": "Use the `list_ragbase` tool to see available knowledge bases.",
        }, ensure_ascii=False, indent=2)

    # Use KB default semantic_weight if not provided
    if semantic_weight is None and kb_meta:
        semantic_weight = kb_meta.get("recommended_semantic_weight", 0.4)
    elif semantic_weight is None:
        semantic_weight = 0.4

    # ---- Step 1: Vector search ----
    embedder = _get_embedder()
    query_emb = embedder.encode([query])[0].tolist()

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

    if vector_results and vector_results.get("documents"):
        for i, doc_text in enumerate(vector_results["documents"][0]):
            dist = vector_results["distances"][0][i] if vector_results.get("distances") else 1.0
            vec_score = 1.0 - dist

            bm25_idx = bm25_docs.index(doc_text) if doc_text in bm25_docs else -1
            kw_score = bm25_scores.get(bm25_idx, 0.0)

            max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
            kw_norm = kw_score / max_bm25 if max_bm25 > 0 else 0.0

            hybrid = semantic_weight * vec_score + (1 - semantic_weight) * kw_norm

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
                parent_content = meta.get("parent_content", "")
                if parent_content:
                    result_entry["parent_content"] = parent_content
                results_map[doc_key] = result_entry

    sorted_results = sorted(results_map.values(), key=lambda x: x["score"], reverse=True)[:top_k]

    if not sorted_results:
        return json.dumps({
            "status": "success",
            "query": query,
            "knowledge_base": knowledge_base,
            "results": [],
            "note": f"No results found in '{knowledge_base}' for query: '{query}'.",
        }, ensure_ascii=False, indent=2)

    output = {
        "status": "success",
        "query": query,
        "knowledge_base": knowledge_base,
        "chunk_strategy": kb_meta.get("chunk_strategy", "general") if kb_meta else "general",
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
    description: str = (
        "Search an indexed knowledge base using hybrid semantic + keyword retrieval. "
        "Use `list_ragbase` first to see available knowledge bases and their descriptions. "
        "Then call this tool with a specific knowledge_base name and query."
    )
    args_schema: type[BaseModel] = SearchKnowledgeInput

    def _run(
        self,
        query: str,
        knowledge_base: str,
        top_k: int = 5,
        semantic_weight: Optional[float] = None,
    ) -> str:
        """Run the knowledge search."""
        return search_knowledge(query, knowledge_base, top_k, semantic_weight)

search_knowledge_tool = SearchKnowledgeTool()
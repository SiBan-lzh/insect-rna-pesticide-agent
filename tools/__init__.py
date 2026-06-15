"""
tools —— Bioinformatics tool wrappers for LangChain.

Each original FastAPI service → one LangChain BaseTool subclass.
HTTP layer stripped; direct function calls with zero network overhead.

Usage:
    from tools import insect_blast_tool

    result = insect_blast_tool.invoke({
        "sequence": "ATGCGTACG...",
        "species": "Bombyx_mori"
    })
"""

from .insect_blast import insect_blast_tool
from .nto_blast import nto_blast_tool
from .primer3 import primer3_tool
from .oligowalk import oligowalk_tool
from .insect_anno import insect_anno_tool
from .clustal import clustal_tool
from .fetch_seq import fetch_nto_seq_tool, fetch_insect_cds_tool
from .kinship import kinship_tool
from .pubmed_esearch import pubmed_esearch_tool
from .pubmed_efetch import pubmed_efetch_tool
from .openalex_search import openalex_search_tool
from .clip_seq import clip_seq_tool
from .clean_seq import clean_seq_tool
from .search_skills import search_skills_tool
from .list_tools import list_tools_tool
from .lookup_data import lookup_data_tool

# ============================================================
# Tool registry
# ============================================================
ALL_TOOLS = [
    insect_blast_tool,
    nto_blast_tool,
    primer3_tool,
    oligowalk_tool,
    insect_anno_tool,
    clustal_tool,
    fetch_nto_seq_tool,
    fetch_insect_cds_tool,
    kinship_tool,
    pubmed_esearch_tool,
    pubmed_efetch_tool,
    openalex_search_tool,
    clip_seq_tool,
    clean_seq_tool,
    search_skills_tool,
    list_tools_tool,
    lookup_data_tool,
]

# ============================================================
# Tool categories (for list_tools / LLM discovery)
# ============================================================
TOOL_CATEGORIES = {
    "target": {
        "label": "🎯 Target discovery",
        "tools": [insect_blast_tool, insect_anno_tool, fetch_insect_cds_tool],
    },
    "sequence": {
        "label": "🧹 Sequence processing",
        "tools": [clean_seq_tool, clip_seq_tool],
    },
    "dsrna": {
        "label": "🧬 dsRNA & primer design",
        "tools": [oligowalk_tool, primer3_tool],
    },
    "safety": {
        "label": "🛡️ Safety assessment",
        "tools": [nto_blast_tool, clustal_tool, fetch_nto_seq_tool],
    },
    "literature": {
        "label": "📚 Literature search",
        "tools": [pubmed_esearch_tool, pubmed_efetch_tool, openalex_search_tool],
    },
    "kinship": {
        "label": "🌳 Phylogenetic analysis",
        "tools": [kinship_tool],
    },
    "auxiliary": {
        "label": "🛠️ Auxiliary tools",
        "tools": [search_skills_tool, list_tools_tool, lookup_data_tool],
    },
}

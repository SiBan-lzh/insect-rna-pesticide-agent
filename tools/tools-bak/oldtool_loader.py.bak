"""
tool_loader.py — Centralized tool registry.

Usage:
    from tools.tool_loader import TOOL_REGISTRY, TOOL_CATEGORIES

    tool = TOOL_REGISTRY[0]
    result = tool.invoke({"sequence": "ATGC..."})
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
from .list_tools import list_tools_tool
from .list_skills import list_skills_tool
from .read_skill import read_skill_tool
from .list_ragbase import list_ragbase_tool
from .list_database import list_database_tool
from .search_knowledge import search_knowledge_tool

# ============================================================
# Tool registry
# ============================================================
TOOL_REGISTRY = [
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
    list_tools_tool,
    list_skills_tool,
    read_skill_tool,
    list_ragbase_tool,
    list_database_tool,
    search_knowledge_tool,
]

# ============================================================
# Tool categories (for list_tools / LLM discovery)
# ============================================================
TOOL_CATEGORIES = {
    "target": {
        "label": "Target discovery",
        "tools": [insect_blast_tool, insect_anno_tool, fetch_insect_cds_tool],
    },
    "sequence": {
        "label": "Sequence processing",
        "tools": [clean_seq_tool, clip_seq_tool],
    },
    "dsrna": {
        "label": "dsRNA & primer design",
        "tools": [oligowalk_tool, primer3_tool],
    },
    "safety": {
        "label": "Safety assessment",
        "tools": [nto_blast_tool, clustal_tool, fetch_nto_seq_tool],
    },
    "literature": {
        "label": "Literature search",
        "tools": [pubmed_esearch_tool, pubmed_efetch_tool, openalex_search_tool],
    },
    "kinship": {
        "label": "Phylogenetic analysis",
        "tools": [kinship_tool],
    },
    "discovery": {
        "label": "Discovery tools",
        "tools": [list_tools_tool, list_skills_tool, read_skill_tool,
                 list_ragbase_tool, list_database_tool, search_knowledge_tool],
    },
}

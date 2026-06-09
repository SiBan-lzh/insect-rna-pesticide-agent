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
]

# ============================================================
# Agent tool whitelists (Harness permission isolation)
# ============================================================

# Species phylogenetic relationship analysis
KINSHIP_ANALYST_TOOLS = [
    kinship_tool,
]

# Online literature search (PubMed + OpenAlex)
KNOWLEDGE_RETRIEVER_TOOLS = [
    pubmed_esearch_tool,
    pubmed_efetch_tool,
    openalex_search_tool,
]

# Target gene discovery in insect genomes
TARGET_DESIGNER_TOOLS = [
    insect_blast_tool,
    insect_anno_tool,
    fetch_insect_cds_tool,
    clip_seq_tool,
]

# dsRNA design & PCR primer design
DSRNA_DESIGNER_TOOLS = [
    oligowalk_tool,
    primer3_tool,
]

# Off-target risk assessment for non-target organisms
SAFETY_INSPECTOR_TOOLS = [
    nto_blast_tool,
    clustal_tool,
    fetch_nto_seq_tool,
]

"""
tool_config.py — Centralized bioinformatics tool data path config.
"""

import os
from pathlib import Path


# ============================================================
# Project root
# ============================================================
PROJECT_ROOT = Path(__file__).parent.resolve()

# ============================================================
# Database root (override via DATABASE_ROOT in .env)
# ============================================================
DATABASE_ROOT = Path(os.getenv("DATABASE_ROOT", PROJECT_ROOT / "database"))

# ============================================================
# BLAST databases
# ============================================================
# Target insect genomes: {species}/{species}.n*
INSECT_BLAST_DB = Path(os.getenv(
    "INSECT_BLAST_DB",
    DATABASE_ROOT / "blast" / "insect_blastndb"
))

# NTO (non-target organism) genomes: {species}/{species}.n*
NTO_BLAST_DB = Path(os.getenv(
    "NTO_BLAST_DB",
    DATABASE_ROOT / "blast" / "nto_blastndb"
))

# ============================================================
# Genome annotations (GFF3)
# ============================================================
INSECT_GFF3_DB = Path(os.getenv(
    "INSECT_GFF3_DB",
    DATABASE_ROOT / "annotation" / "insect_gff3"
))

# ============================================================
# Reference sequences (samtools extraction)
# ============================================================
# Insect CDS: {species}/*.cds.fa
INSECT_CDS_DB = Path(os.getenv(
    "INSECT_CDS_DB",
    DATABASE_ROOT / "refseq" / "insect_cds"
))

# NTO reference sequences with .fai index: {species}/*.fa
NTOS_REFSEQ_DB = Path(os.getenv(
    "NTOS_REFSEQ_DB",
    DATABASE_ROOT / "refseq" / "nto_refseq"
))

# ============================================================
# Kinship — species tree + NCBI taxonomy
# ============================================================
KINSHIP_DB = Path(os.getenv(
    "KINSHIP_DB",
    DATABASE_ROOT / "kinship"
))

INSECT_TREE_PATH = Path(os.getenv(
    "INSECT_TREE_PATH",
    KINSHIP_DB / "insects_species.nwk"
))

INSECT_TAXA_DB_PATH = Path(os.getenv(
    "INSECT_TAXA_DB_PATH",
    KINSHIP_DB / "taxa.sqlite"
))

# ============================================================
# RNAstructure — OligoWalk thermodynamics
# ============================================================
RNASTRUCTURE_HOME = Path(os.getenv(
    "RNASTRUCTURE_HOME",
    DATABASE_ROOT / "RNAstructure"
))

OLIGOWALK_BIN = Path(os.getenv(
    "OLIGOWALK_BIN",
    RNASTRUCTURE_HOME / "OligoWalk"
))

# DATAPATH env var required by OligoWalk at runtime
RNASTRUCTURE_DATAPATH = Path(os.getenv(
    "RNASTRUCTURE_DATAPATH",
    RNASTRUCTURE_HOME / "data_tables"
))

# ============================================================
# PubMed E-utilities
# ============================================================
PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "user@example.com")
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY", "")
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# ============================================================
# OpenAlex API
# ============================================================
OPENALEX_BASE_URL = "https://api.openalex.org"

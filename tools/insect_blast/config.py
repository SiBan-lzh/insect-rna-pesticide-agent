"""
insect_blast/config.py — BLAST database path configuration.
"""

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(".env", usecwd=True), override=True)

# ============================================================
# Project root (tools/insect_blast/ → tools/ → project root)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ============================================================
# Database root (override via DATABASE_ROOT in .env)
# ============================================================
DATABASE_ROOT = Path(os.getenv("DATABASE_ROOT", PROJECT_ROOT / "database"))

# ============================================================
# BLAST database for target insect genomes
# Structure: {INSECT_BLAST_DB}/{species}/{species}.n*
# ============================================================
INSECT_BLAST_DB = Path(os.getenv(
    "INSECT_BLAST_DB",
    DATABASE_ROOT / "blast" / "insect_blastndb"
))

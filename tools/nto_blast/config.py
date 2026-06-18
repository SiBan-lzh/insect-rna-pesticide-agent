"""
nto_blast/config.py — NTO BLAST database path configuration.
"""
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(".env", usecwd=True), override=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATABASE_ROOT = Path(os.getenv("DATABASE_ROOT", PROJECT_ROOT / "database"))
NTO_BLAST_DB = Path(os.getenv("NTO_BLAST_DB", DATABASE_ROOT / "blast" / "nto_blastndb"))

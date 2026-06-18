"""
fetch_insect_cds/config.py — Insect CDS database path configuration.
"""
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(".env", usecwd=True), override=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATABASE_ROOT = Path(os.getenv("DATABASE_ROOT", PROJECT_ROOT / "database"))
INSECT_CDS_DB = Path(os.getenv("INSECT_CDS_DB", DATABASE_ROOT / "refseq" / "insect_cds"))

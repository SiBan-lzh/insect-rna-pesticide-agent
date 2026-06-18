"""
kinship/config.py — Species tree and taxonomy database paths.
"""
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(".env", usecwd=True), override=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATABASE_ROOT = Path(os.getenv("DATABASE_ROOT", PROJECT_ROOT / "database"))
KINSHIP_DB = Path(os.getenv("KINSHIP_DB", DATABASE_ROOT / "kinship"))
INSECT_TREE_PATH = Path(os.getenv("INSECT_TREE_PATH", KINSHIP_DB / "insects_species.nwk"))
INSECT_TAXA_DB_PATH = Path(os.getenv("INSECT_TAXA_DB_PATH", KINSHIP_DB / "taxa.sqlite"))

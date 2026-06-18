"""
oligowalk/config.py — RNAstructure OligoWalk path configuration.
"""
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(".env", usecwd=True), override=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATABASE_ROOT = Path(os.getenv("DATABASE_ROOT", PROJECT_ROOT / "database"))
RNASTRUCTURE_HOME = Path(os.getenv("RNASTRUCTURE_HOME", DATABASE_ROOT / "RNAstructure"))
OLIGOWALK_BIN = Path(os.getenv("OLIGOWALK_BIN", RNASTRUCTURE_HOME / "OligoWalk"))
RNASTRUCTURE_DATAPATH = Path(os.getenv("RNASTRUCTURE_DATAPATH", RNASTRUCTURE_HOME / "data_tables"))

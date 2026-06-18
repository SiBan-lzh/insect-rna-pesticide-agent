"""
list_ragbase/config.py — RAG Chroma index path configuration.
"""
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(".env", usecwd=True), override=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAGBASE_ROOT = Path(os.getenv("RAGBASE_ROOT", PROJECT_ROOT / "ragbase"))
RAG_CHROMA_DIR = Path(os.getenv("RAG_CHROMA_DIR", RAGBASE_ROOT / "chroma_db"))

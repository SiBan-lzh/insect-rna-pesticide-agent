"""
config_check.py — Standalone environment configuration checker.

Checks:
  1. Database paths — all data directories exist on disk
  2. Tool binaries — required external executables are available
  3. Python packages — required libraries are installed
  4. .env configuration — required environment variables are set

Usage:
    cd /path/to/langgraph
    source langgraph_env/bin/activate
    python tests/config_check.py

This script is self-contained and does not depend on tool_config.py.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ============================================================
# Project root (this file is at tests/config_check.py)
# ============================================================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ============================================================
# .env loader (same pattern as tool_config.py)
# ============================================================
def _load_env():
    """Load .env and expand ${VAR} references."""
    from dotenv import dotenv_values, find_dotenv, load_dotenv

    dotenv_path = find_dotenv(".env", usecwd=True)
    load_dotenv(dotenv_path, override=True)
    if dotenv_path:
        for _k in dotenv_values(dotenv_path):
            _v = os.environ.get(_k, "")
            if "$" in _v:
                os.environ[_k] = os.path.expandvars(_v)
    return dotenv_path


# ============================================================
# Path definitions (mirrors tool_config.py)
# ============================================================
DATABASE_ROOT = Path(os.getenv("DATABASE_ROOT", _PROJECT_ROOT / "database"))

INSECT_BLAST_DB = Path(os.getenv(
    "INSECT_BLAST_DB", DATABASE_ROOT / "blast" / "insect_blastndb"))
NTO_BLAST_DB = Path(os.getenv(
    "NTO_BLAST_DB", DATABASE_ROOT / "blast" / "nto_blastndb"))

INSECT_GFF3_DB = Path(os.getenv(
    "INSECT_GFF3_DB", DATABASE_ROOT / "annotation" / "insect_gff3"))

INSECT_CDS_DB = Path(os.getenv(
    "INSECT_CDS_DB", DATABASE_ROOT / "refseq" / "insect_cds"))
NTOS_REFSEQ_DB = Path(os.getenv(
    "NTOS_REFSEQ_DB", DATABASE_ROOT / "refseq" / "nto_refseq"))

KINSHIP_DB = Path(os.getenv(
    "KINSHIP_DB", DATABASE_ROOT / "kinship"))
INSECT_TREE_PATH = Path(os.getenv(
    "INSECT_TREE_PATH", KINSHIP_DB / "insects_species.nwk"))
INSECT_TAXA_DB_PATH = Path(os.getenv(
    "INSECT_TAXA_DB_PATH", KINSHIP_DB / "taxa.sqlite"))

RNASTRUCTURE_HOME = Path(os.getenv(
    "RNASTRUCTURE_HOME", DATABASE_ROOT / "RNAstructure"))
OLIGOWALK_BIN = Path(os.getenv(
    "OLIGOWALK_BIN", RNASTRUCTURE_HOME / "OligoWalk"))
RNASTRUCTURE_DATAPATH = Path(os.getenv(
    "RNASTRUCTURE_DATAPATH", RNASTRUCTURE_HOME / "data_tables"))

RAGBASE_ROOT = Path(os.getenv(
    "RAGBASE_ROOT", _PROJECT_ROOT / "ragbase"))
RAG_CHROMA_DIR = Path(os.getenv(
    "RAG_CHROMA_DIR", RAGBASE_ROOT / "chroma_db"))

SKILLS_DIR = Path(os.getenv("SKILLS_DIR", _PROJECT_ROOT / "skills"))
TOOLS_DIR = Path(os.getenv("TOOLS_DIR", _PROJECT_ROOT / "tools"))

PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "")
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY", "")
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OPENALEX_BASE_URL = "https://api.openalex.org"

# All Path variables for validation (auto-collected)
_PATH_VARS: dict[str, Path] = {}
for _name, _val in list(locals().items()):
    if isinstance(_val, Path):
        _PATH_VARS[_name] = _val


# ============================================================
# Check functions
# ============================================================

def check_paths() -> list[str]:
    """Check all data paths exist on disk. Returns list of missing names."""
    missing: list[str] = []
    for name, path in sorted(_PATH_VARS.items()):
        ok = path.exists()
        status = "✅" if ok else "❌"
        print(f"  {status} {name} = {path}")
        if not ok:
            missing.append(name)
    return missing


def check_tool_binaries() -> list[str]:
    """Check required external tool binaries are in PATH or at configured paths.

    Returns list of missing binaries.
    """
    binaries: dict[str, str | Path] = {
        "blastn": "blastn",
        "bedtools": "bedtools",
        "samtools": "samtools",
        "clustalo": "clustalo",
        "OligoWalk": OLIGOWALK_BIN,
    }
    missing: list[str] = []
    for label, path in sorted(binaries.items()):
        if path == OLIGOWALK_BIN:
            # OligoWalk has a configured path, not necessarily in PATH
            ok = Path(path).exists() or shutil.which(str(path)) is not None
        else:
            ok = shutil.which(str(path)) is not None
        status = "✅" if ok else "❌"
        print(f"  {status} {label} ({path})")
        if not ok:
            missing.append(label)
    return missing


def check_python_packages() -> list[str]:
    """Check required Python packages are importable.

    Returns list of missing packages.
    """
    packages: dict[str, str] = {
        "primer3": "primer3-py",
        "chromadb": "chromadb",
        "sentence_transformers": "sentence-transformers",
        "yaml": "PyYAML",
        "requests": "requests",
        "ete4": "ete4",
        "openpyxl": "openpyxl (Excel files)",
        "docx": "python-docx (Word files)",
        "pypdf": "pypdf (PDF files)",
        "rank_bm25": "rank-bm25",
        "dotenv": "python-dotenv",
        "langchain_core": "langchain-core",
        "langgraph": "langgraph",
        "langchain_openai": "langchain-openai",
        "langchain_anthropic": "langchain-anthropic",
        "langchain_google_genai": "langchain-google-genai",
    }
    missing: list[str] = []
    for modname, pip_name in sorted(packages.items()):
        try:
            importlib.import_module(modname)
            status = "✅"
        except ImportError:
            status = "❌"
            missing.append(pip_name)
        print(f"  {status} {pip_name}")
    return missing


def check_env_config() -> list[str]:
    """Check .env has required LLM configuration.

    Returns list of missing config items.
    """
    required = ["MODEL_PROVIDER", "MODEL_NAME", "API_KEY"]
    missing: list[str] = []
    for key in required:
        val = os.getenv(key, "")
        ok = bool(val.strip())
        status = "✅" if ok else "❌"
        print(f"  {status} {key} = {'<set>' if ok else '<empty>'}")
        if not ok:
            missing.append(key)
    return missing


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("  RNAi Pesticide Agent — Environment Check")
    print("=" * 60)
    print()

    # ---- Load .env ----
    dotenv_path = _load_env()
    print(f"[1/5] .env file: {dotenv_path or 'NOT FOUND'}")
    if not dotenv_path:
        print("  ⚠️  No .env file found. Copy .env.example to .env and configure it.")
    print()

    # ---- Database paths ----
    print(f"[2/5] Database paths ({len(_PATH_VARS)} variables)")
    missing_paths = check_paths()
    print()

    # ---- Tool binaries ----
    print("[3/5] External tool binaries")
    missing_bins = check_tool_binaries()
    print()

    # ---- Python packages ----
    print("[4/5] Python packages")
    missing_pkgs = check_python_packages()
    print()

    # ---- .env config ----
    print("[5/5] .env LLM configuration")
    missing_env = check_env_config()
    print()

    # ---- Summary ----
    print("=" * 60)
    total_missing = len(missing_paths) + len(missing_bins) + len(missing_pkgs) + len(missing_env)
    if total_missing == 0:
        print("  ✅ All checks passed!")
    else:
        print(f"  ⚠️  {total_missing} issue(s) found:")
        if missing_paths:
            print(f"     - {len(missing_paths)} data path(s) missing")
        if missing_bins:
            print(f"     - {len(missing_bins)} binary(ies) not found")
        if missing_pkgs:
            print(f"     - {len(missing_pkgs)} package(s) missing: {', '.join(missing_pkgs)}")
        if missing_env:
            print(f"     - {len(missing_env)} .env config(s) missing")
    print()

    # ---- Hints ----
    if missing_bins:
        print("  Install missing system binaries:")
        print("    sudo apt install ncbi-blast+ bedtools samtools clustalo")
        print("    cd database/RNAstructure && make")
        print()
    if missing_pkgs:
        print(f"  Install missing packages:")
        print(f"    pip install {' '.join(missing_pkgs)}")
        print()
    if missing_paths:
        print("  Missing data paths — download required genome/annotation files:")
        print("    See database/README.md for download instructions")
        print()


if __name__ == "__main__":
    main()

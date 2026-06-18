"""
list_database.py — Tool to check availability of locally stored bioinformatics data.

The LLM calls this tool when it needs to know whether a specific species' genome
data (BLAST DB, GFF3 annotation, CDS sequences, etc.) exists in the local
database/ directory before running analysis tools.

Usage:
    from tools.list_database import list_database_tool

    # Check if Bombyx_mori has a BLAST database
    result = list_database_tool.invoke({
        "species": "Bombyx_mori",
        "data_type": "blast_insect"
    })

    # List all available species for a data type
    result = list_database_tool.invoke({
        "data_type": "blast_nto"
    })
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .config import DATABASE_ROOT, INSECT_BLAST_DB, NTO_BLAST_DB, INSECT_GFF3_DB, INSECT_CDS_DB, NTOS_REFSEQ_DB, KINSHIP_DB, RNASTRUCTURE_HOME


# ============================================================
# Pydantic input schema
# ============================================================
class ListDatabaseInput(BaseModel):
    """Local data lookup parameters."""

    species: Optional[str] = Field(
        default=None,
        description="Species name (Latin, underscore-separated), e.g. 'Bombyx_mori'. "
        "Leave empty to list ALL available species for the given data_type.",
    )
    data_type: str = Field(
        description="Type of data to look up. One of:\n"
        "- 'blast_insect' — target insect BLAST database\n"
        "- 'blast_nto' — non-target organism BLAST database\n"
        "- 'annotation' — insect genome GFF3 annotation\n"
        "- 'refseq_cds' — insect CDS reference sequences\n"
        "- 'refseq_nto' — NTO reference genome sequences\n"
        "- 'nto_list' — NTO species list JSON files\n"
        "- 'kinship' — species tree and taxonomy database\n"
        "- 'rnastructure' — RNAstructure thermodynamics tools",
    )


# ============================================================
# Data type → path resolver
# ============================================================
DATA_TYPE_MAP: dict[str, Path] = {
    "blast_insect": INSECT_BLAST_DB,
    "blast_nto": NTO_BLAST_DB,
    "annotation": INSECT_GFF3_DB,
    "refseq_cds": INSECT_CDS_DB,
    "refseq_nto": NTOS_REFSEQ_DB,
    "nto_list": DATABASE_ROOT / "NTOs_lists",
    "kinship": KINSHIP_DB,
    "rnastructure": RNASTRUCTURE_HOME,
}

# Types where each species is a subdirectory
DIR_PER_SPECIES_TYPES = {
    "blast_insect", "blast_nto", "annotation",
    "refseq_cds", "refseq_nto",
}

# Types where the directory contains flat files
FLAT_FILE_TYPES = {"nto_list", "kinship", "rnastructure"}


def _list_species_in_dir(data_dir: Path) -> list[str]:
    """List species subdirectories in a given data directory."""
    if not data_dir.is_dir():
        return []
    return sorted(
        d.name for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def _list_files_in_dir(data_dir: Path, ext: str = "") -> list[str]:
    """List files in a given data directory, optionally filtered by extension."""
    if not data_dir.is_dir():
        return []
    return sorted(
        f.name for f in data_dir.iterdir()
        if f.is_file() and not f.name.startswith(".")
        and (not ext or f.name.endswith(ext))
    )


def list_database(
    species: Optional[str] = None,
    data_type: str = "blast_insect",
) -> str:
    """Check availability of locally stored bioinformatics data.

    Args:
        species: Species name (Latin, underscore-separated). If None, lists all
                 available species for the given data_type.
        data_type: Type of data repository to search.

    Returns:
        Availability report with paths and species lists.
    """
    data_type = data_type.lower().strip()

    # Validate data_type
    if data_type not in DATA_TYPE_MAP:
        valid = ", ".join(sorted(DATA_TYPE_MAP.keys()))
        return (
            f"Unknown data_type: '{data_type}'.\n"
            f"Valid types: {valid}."
        )

    data_dir = DATA_TYPE_MAP[data_type]

    # Check root directory exists
    if not data_dir.exists():
        return (
            f"[{data_type}] Data directory not found:\n"
            f"  Expected at: {data_dir}\n"
            f"  Status: NOT AVAILABLE (data not downloaded yet)"
        )

    # Mode 1: species-level lookup (DIR_PER_SPECIES_TYPES)
    if data_type in DIR_PER_SPECIES_TYPES:
        if species:
            species_clean = species.replace(" ", "_")
            species_dir = data_dir / species_clean
            if species_dir.is_dir():
                files = [f.name for f in species_dir.iterdir() if f.is_file()]
                return (
                    f"[{data_type}] Species '{species_clean}' — AVAILABLE\n"
                    f"  Path: {species_dir}\n"
                    f"  Files: {len(files)} found"
                )
            else:
                return (
                    f"[{data_type}] Species '{species_clean}' — NOT AVAILABLE\n"
                    f"  Expected at: {species_dir}\n"
                    f"  Hint: Use list_database(data_type='{data_type}') without "
                    f"species to list all available species."
                )
        else:
            species_list = _list_species_in_dir(data_dir)
            if not species_list:
                return (
                    f"[{data_type}] Data directory exists but no species found:\n"
                    f"  Path: {data_dir}\n"
                    f"  Status: EMPTY"
                )
            return (
                f"[{data_type}] Available species ({len(species_list)}):\n"
                + "\n".join(f"  - {s}" for s in species_list)
                + f"\n\nPath: {data_dir}"
            )

    # Mode 2: flat file lookup (FLAT_FILE_TYPES)
    if data_type in FLAT_FILE_TYPES:
        if species and data_type == "nto_list":
            files = _list_files_in_dir(data_dir, ".json")
            species_lower = species.lower().replace(" ", "_")
            matches = [f for f in files if species_lower in f.lower()]
            if matches:
                return (
                    f"[{data_type}] Match found for '{species}':\n"
                    + "\n".join(f"  - {data_dir / f}" for f in matches)
                )
            else:
                return (
                    f"[{data_type}] No match for '{species}'.\n"
                    f"  Available files ({len(files)}):\n"
                    + "\n".join(f"  - {f}" for f in files)
                    + f"\n\nPath: {data_dir}"
                )

        files = _list_files_in_dir(data_dir)
        if not files:
            return (
                f"[{data_type}] Data directory exists but no files found:\n"
                f"  Path: {data_dir}\n"
                f"  Status: EMPTY"
            )
        return (
            f"[{data_type}] Available items ({len(files)}):\n"
            + "\n".join(f"  - {f}" for f in files[:30])
            + (f"\n  ... and {len(files) - 30} more" if len(files) > 30 else "")
            + f"\n\nPath: {data_dir}"
        )

    return f"Unexpected error for data_type='{data_type}'."


class ListDatabaseTool(BaseTool):
    """Tool for checking availability of locally stored bioinformatics data."""

    name: str = "list_database"
    description: str = (
        "Check whether specific bioinformatics data exists in the local database. "
        "Call this BEFORE running analysis tools to verify that the required genome "
        "data (BLAST database, GFF3 annotation, CDS sequences) is available for a "
        "given species. Can also list all available species for each data type, or "
        "show available NTO species lists and kinship/taxonomy data. "
        "Example: list_database(species='Bombyx_mori', data_type='blast_insect')."
    )
    args_schema: type[BaseModel] = ListDatabaseInput

    def _run(self, data_type: str, species: Optional[str] = None) -> str:
        """Run the data lookup."""
        return list_database(species, data_type)

    async def _arun(self, data_type: str, species: Optional[str] = None) -> str:
        """Async variant."""
        return self._run(data_type, species)


list_database_tool = ListDatabaseTool()
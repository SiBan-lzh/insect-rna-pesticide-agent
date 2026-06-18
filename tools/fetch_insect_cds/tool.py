"""
fetch_seq.py — Sequence extraction tools

Two tools:
  - fetch_seq:         Extract region from NTO reference genome via samtools faidx
  - fetch_insect_cds:  Search CDS sequence from insect CDS database by transcript ID
"""

import glob
import json
import logging
import os
import subprocess
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from .config import INSECT_CDS_DB

logger = logging.getLogger("RPA_Tools.FetchSeq")

# ============================================================
# Pydantic input schema
# ============================================================

class FetchRegionHit(BaseModel):
    """Target region to extract."""

    subject_id: str = Field(
        description="Reference sequence ID (identifier after '>' in FASTA header)"
    )
    s_start: int = Field(
        description="Extraction start position (1-based)"
    )
    s_end: int = Field(
        description="Extraction end position (1-based). If start > end, auto reverse-complement"
    )


class FetchSeqInput(BaseModel):
    """NTO reference sequence region extraction parameters."""

    species: str = Field(
        description="Species name, e.g. Apis_mellifera"
    )
    hits: List[FetchRegionHit] = Field(
        description="List of regions to extract, each with subject_id, s_start, s_end"
    )


class FetchInsectCDSInput(BaseModel):
    """Insect CDS sequence query parameters."""

    species: str = Field(
        description="Species name, e.g. Bombyx_mori"
    )
    transcript_id: str = Field(
        description="Transcript/mRNA ID, e.g. 'Bmor000255.1' (from insect_anno annotation)"
    )


# ============================================================
# Helper functions
# ============================================================

def _find_fasta(species: str, db_root: str, suffix: str = ".fa") -> Optional[str]:
    """Find the first FASTA file for a species in the database directory."""
    species_dir = os.path.join(db_root, species)
    pattern = os.path.join(species_dir, f"*{suffix}")
    files = glob.glob(pattern, recursive=False)
    if not files:
        files = glob.glob(os.path.join(species_dir, "**", f"*{suffix}"), recursive=True)
    return files[0] if files else None


def _reverse_complement(seq: str) -> str:
    """Compute reverse complement."""
    complement = str.maketrans("ATCGatcg", "TAGCtagc")
    return seq.translate(complement)[::-1]


def _fetch_region(fasta_path: str, seqid: str, start: int, end: int) -> str:
    """Extract region using samtools faidx."""
    reverse = False
    if start > end:
        start, end = end, start
        reverse = True

    region = f"{seqid}:{start}-{end}"
    cmd = ["samtools", "faidx", fasta_path, region]
    logger.info("Fetching region: %s", region)

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    lines = result.stdout.strip().split("\n")
    if len(lines) < 2:
        raise RuntimeError(f"Invalid samtools output for {region}")

    seq = "".join(lines[1:])

    if reverse:
        seq = _reverse_complement(seq)
    return seq


def _fetch_cds(species: str, transcript_id: str) -> Optional[str]:
    """Search for a transcript sequence in insect CDS FASTA files."""
    species_dir = os.path.join(str(INSECT_CDS_DB), species)
    cds_files = glob.glob(os.path.join(species_dir, "**", "*.cds.fa"), recursive=True)

    if not cds_files:
        logger.error("No .cds.fa files found for species: %s", species)
        return None

    target_id = f">{transcript_id}"
    for cds_path in cds_files:
        try:
            with open(cds_path, "r") as f:
                found = False
                sequence_parts = []
                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        if found:
                            break
                        if line.startswith(target_id):
                            found = True
                    elif found:
                        sequence_parts.append(line)

                if found:
                    return "".join(sequence_parts)
        except Exception as e:
            logger.error("Error reading %s: %s", cds_path, e)
            continue

    return None


# ============================================================
# Tool 2: fetch_insect_cds
# ============================================================

class FetchInsectCDSTool(BaseTool):
    """Search for CDS sequence by transcript ID in insect CDS database.

    Pure Python file search, no external binary dependency.
    Typical usage: receive transcript ID from insect_anno annotation, return CDS sequence.
    """

    name: str = "fetch_insect_cds"
    description: str = (
        "Search for CDS sequence by transcript/mRNA ID in insect CDS database."
        "Input species name and transcript ID (e.g. Bmor000255.1 from insect_anno annotation),"
        "returns the corresponding CDS sequence."
        "Pure Python file search, no external dependency."
    )
    args_schema: type = FetchInsectCDSInput

    def _run(
        self,
        species: str,
        transcript_id: str,
    ) -> str:
        """Search insect CDS sequence (synchronous)."""

        try:
            sequence = _fetch_cds(species, transcript_id)

            if not sequence:
                return json.dumps(
                    {
                        "status": "error",
                        "error": "transcript not found",
                        "details": (
                            f"Transcript ID '{transcript_id}' not found "
                            f"in {species} CDS database"
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )

            return json.dumps(
                {
                    "status": "success",
                    "species": species,
                    "transcript_id": transcript_id,
                    "length": len(sequence),
                    "sequence": sequence,
                },
                ensure_ascii=False,
                indent=2,
            )

        except Exception as e:
            logger.exception("Fetch CDS failed")
            return json.dumps(
                {
                    "status": "error",
                    "error": "internal error",
                    "details": str(e),
                },
                ensure_ascii=False,
                indent=2,
            )


# ============================================================
# Singleton export
# ============================================================
fetch_insect_cds_tool = FetchInsectCDSTool()

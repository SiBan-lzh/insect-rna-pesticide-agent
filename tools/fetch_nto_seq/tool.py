"""
fetch_seq.py -- Sequence extraction toolkit

Provides two tools:
  - fetch_seq:         Extract region sequences from NTO reference genomes via samtools faidx
  - fetch_insect_cds:  Search CDS sequences from insect CDS database by transcript ID
"""

import glob
import json
import logging
import os
import subprocess
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from .config import NTOS_REFSEQ_DB

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
        description="Extraction end position (1-based). Reverse complement if start > end"
    )


class FetchSeqInput(BaseModel):
    """NTO reference sequence extraction parameters."""

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
    """Extract a region using samtools faidx."""
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
# Tool 1: fetch_seq -- NTO reference sequence extraction
# ============================================================

class FetchNtoSeqTool(BaseTool):
    """Extract region sequences from NTO (non-target organism) reference genomes.

    Uses samtools faidx to extract sequences by chromosome/scaffold ID and coordinates,
    with automatic reverse complement when start > end.
    """

    name: str = "fetch_nto_seq"
    description: str = (
        "Extract region sequences from NTO reference genomes. "
        "Input: species name and hit list (each with subject_id, s_start, s_end). "
        "Uses samtools faidx to extract DNA sequences. "
        "Returns reverse complement automatically if s_start > s_end."
    )
    args_schema: type = FetchSeqInput

    def _run(
        self,
        species: str,
        hits: List[dict],
    ) -> str:
        """Extract NTO reference sequence regions (sync)."""

        try:
            fasta_path = _find_fasta(species, str(NTOS_REFSEQ_DB))
            if not fasta_path:
                return json.dumps(
                    {
                        "status": "error",
                        "error": f"No FASTA file found for species '{species}'",
                        "searched_in": str(NTOS_REFSEQ_DB),
                    },
                    ensure_ascii=False,
                    indent=2,
                )

            fai_path = fasta_path + ".fai"
            if not os.path.exists(fai_path):
                logger.warning("No .fai index found, creating one...")
                subprocess.run(
                    ["samtools", "faidx", fasta_path],
                    capture_output=True,
                    text=True,
                    check=True,
                )

            results = []
            for hit in hits:
                seq = _fetch_region(
                    fasta_path,
                    hit.subject_id,
                    hit.s_start,
                    hit.s_end,
                )
                results.append({
                    "subject_id": hit.subject_id,
                    "start": hit.s_start,
                    "end": hit.s_end,
                    "length": len(seq),
                    "sequence": seq,
                })

            return json.dumps(
                {
                    "status": "success",
                    "species": species,
                    "fasta_used": fasta_path,
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            )

        except subprocess.CalledProcessError as e:
            logger.exception("samtools faidx failed")
            return json.dumps(
                {
                    "status": "error",
                    "error": "samtools execution failed",
                    "stderr": e.stderr,
                    "stdout": e.stdout,
                },
                ensure_ascii=False,
                indent=2,
            )

        except FileNotFoundError:
            logger.exception("samtools not found")
            return json.dumps(
                {
                    "status": "error",
                    "error": "samtools binary not found",
                    "details": "Install samtools: sudo apt install -y samtools",
                },
                ensure_ascii=False,
                indent=2,
            )

        except Exception as e:
            logger.exception("Fetch sequence failed")
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
fetch_nto_seq_tool = FetchNtoSeqTool()

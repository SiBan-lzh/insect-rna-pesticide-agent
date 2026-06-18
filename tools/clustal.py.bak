"""
clustal.py — Pairwise alignment and continuous-match analysis via Clustal Omega.
"""

import json
import logging
import os
import subprocess
import tempfile
from typing import List
from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import BaseTool

logger = logging.getLogger("RPA_Tools.Clustal")


# ============================================================
# Pydantic input schema
# ============================================================
class SequenceItem(BaseModel):
    """A single named sequence."""

    name: str = Field(description="Sequence identifier, e.g. siRNA or gene ID")
    sequence: str = Field(description="Nucleotide sequence (5'→3')")


class ClustalInput(BaseModel):
    """Pairwise alignment parameters."""

    sequences: List[SequenceItem] = Field(
        min_length=2, max_length=2,
        description="Two sequences to align (exactly 2 required)"
    )
    window_size: int = Field(
        default=21, ge=18, le=25,
        description="Continuous-match threshold (nt). >= this value = off-target risk"
    )

    @field_validator("sequences")
    @classmethod
    def check_exactly_two(cls, v):
        if len(v) != 2:
            raise ValueError("Exactly two sequences required for pairwise alignment")
        return v


# ============================================================
# Core functions
# ============================================================
def run_clustal(sequences: List[SequenceItem]) -> str:
    """Run Clustal Omega, return FASTA alignment."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as input_file:
        for seq in sequences:
            input_file.write(f">{seq.name}\n{seq.sequence}\n")
        input_file.flush()
        input_path = input_file.name

    output_file = tempfile.NamedTemporaryFile(mode="r", suffix=".fa", delete=False)
    output_path = output_file.name
    output_file.close()

    cmd = [
        "clustalo", "-i", input_path, "-o", output_path,
        "--outfmt", "fa", "--threads", "1", "--force", "--dealign",
    ]

    logger.info("Running: %s", " ".join(cmd))

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        with open(output_path, "r") as f:
            alignment = f.read()
        if not alignment.strip():
            raise RuntimeError("clustalo returned empty alignment")
        return alignment
    finally:
        for p in [input_path, output_path]:
            if os.path.exists(p):
                os.unlink(p)


def parse_alignment(alignment: str) -> List[str]:
    """Extract two sequences from multi-FASTA alignment."""
    lines = alignment.strip().splitlines()
    seqs = []
    current_seq = []
    for line in lines:
        if line.startswith(">"):
            if current_seq:
                seqs.append("".join(current_seq))
                current_seq = []
        else:
            current_seq.append(line.strip())
    if current_seq:
        seqs.append("".join(current_seq))
    if len(seqs) != 2:
        raise ValueError(f"Expected 2 sequences, got {len(seqs)}")
    return seqs


def analyze_continuous_match(alignment: str, window_size: int) -> dict:
    """Detect contiguous match regions >= window_size in aligned sequences."""
    seq1, seq2 = parse_alignment(alignment)

    max_run = 0
    current_run = 0
    start_pos = None
    match_regions = []

    for i, (a, b) in enumerate(zip(seq1, seq2)):
        if a == b and a != "-":
            if current_run == 0:
                start_pos = i
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            if current_run >= window_size:
                match_regions.append({"start_alignment_pos": start_pos, "length": current_run})
            current_run = 0
            start_pos = None

    if current_run >= window_size:
        match_regions.append({"start_alignment_pos": start_pos, "length": current_run})

    return {
        "max_continuous_match": max_run,
        "threshold": window_size,
        "off_target_risk": max_run >= window_size,
        "matching_regions": match_regions,
    }


# ============================================================
# LangChain Tool
# ============================================================
class ClustalTool(BaseTool):
    """Pairwise alignment and continuous-match analysis via Clustal Omega."""

    name: str = "clustal"
    description: str = (
        "Align two sequences with Clustal Omega and detect continuous match regions. "
        "Input: two DNA/RNA sequences. "
        "Output: max continuous match length, regions >= window_size (default 21nt). "
        "Used for siRNA off-target assessment."
    )
    args_schema: type = ClustalInput

    def _run(self, sequences: List[dict], window_size: int = 21) -> str:
        """Run alignment and match analysis (sync)."""
        try:
            alignment = run_clustal(sequences)
            analysis = analyze_continuous_match(alignment, window_size)

            return json.dumps({
                "status": "success",
                "window_size": window_size,
                "analysis": analysis,
                "alignment": alignment,
            }, ensure_ascii=False, indent=2)

        except ValueError as e:
            logger.warning("Clustal validation error: %s", e)
            return json.dumps({"status": "error", "error": "validation error", "details": str(e)})
        except subprocess.CalledProcessError as e:
            logger.exception("Clustal Omega execution failed")
            return json.dumps({"status": "error", "error": "clustalo execution failed", "stderr": e.stderr, "stdout": e.stdout})
        except FileNotFoundError:
            logger.exception("clustalo not found")
            return json.dumps({"status": "error", "error": "clustalo binary not found", "details": "Install: sudo apt install -y clustalo"})
        except Exception as e:
            logger.exception("Clustal failed")
            return json.dumps({"status": "error", "error": "internal error", "details": str(e)})


# ============================================================
# Singleton export
# ============================================================
clustal_tool = ClustalTool()

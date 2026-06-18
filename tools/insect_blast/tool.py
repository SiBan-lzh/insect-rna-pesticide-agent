"""
insect_blast/tool.py — BLASTN against target insect genomes.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from typing import List

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .config import INSECT_BLAST_DB

logger = logging.getLogger("RPA_Tools.InsectBlast")


# ============================================================
# Pydantic input schema
# ============================================================
class InsectBlastInput(BaseModel):
    """BLAST search parameters."""

    sequence: str = Field(
        description="Query sequence (5'→3'), typically primer (18-25nt) or siRNA (19-21nt)"
    )
    species: str = Field(
        description="Target insect (Latin, underscore-separated), e.g. Bombyx_mori"
    )
    word_size: int = Field(default=7, description="BLAST word size")
    evalue: float = Field(default=1.0, description="E-value threshold")
    gapopen: int = Field(default=10, description="Gap opening penalty")
    gapextend: int = Field(default=4, description="Gap extension penalty")
    penalty: int = Field(default=-1, description="Mismatch penalty")
    reward: int = Field(default=1, description="Match reward")


# ============================================================
# Core parsing functions
# ============================================================
def parse_blast_tabular(raw_text: str) -> List[dict]:
    """Parse blastn -outfmt 6 tabular output (12 columns)."""
    results = []
    for i, line in enumerate(raw_text.strip().splitlines()):
        if not line:
            continue
        cols = line.split("\t")
        if len(cols) < 12:
            continue
        results.append({
            "hit_number": i + 1,
            "subject_id": cols[1],
            "identity_pct": float(cols[2]),
            "aln_length": int(cols[3]),
            "mismatches": int(cols[4]),
            "gap_opens": int(cols[5]),
            "q_start": int(cols[6]),
            "q_end": int(cols[7]),
            "s_start": int(cols[8]),
            "s_end": int(cols[9]),
            "evalue": float(cols[10]),
            "bit_score": float(cols[11]),
        })
    return results


# ============================================================
# LangChain Tool
# ============================================================
class InsectBlastTool(BaseTool):
    """BLASTN against target insect genomes."""

    name: str = "insect_blast"
    description: str = (
        "BLASTN against target insect genome. "
        "Input: DNA sequence + species name (Latin). "
        "Output: match positions, identity %, E-value."
    )
    args_schema: type = InsectBlastInput
    timeout: int = 120

    def _run(
        self,
        sequence: str,
        species: str,
        word_size: int = 7,
        evalue: float = 1.0,
        gapopen: int = 10,
        gapextend: int = 4,
        penalty: int = -1,
        reward: int = 1,
    ) -> str:
        """Run BLAST (sync)."""
        db_path = os.path.join(INSECT_BLAST_DB, species, species)
        if not os.path.exists(f"{db_path}.nsq"):
            return json.dumps({
                "status": "error",
                "error": f"BLAST database not found for species '{species}'",
                "expected_path": f"{db_path}.nsq",
            }, ensure_ascii=False, indent=2)

        tmp_in_path = None
        tmp_out_path = None

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as tmp_in:
                tmp_in.write(f">query\n{sequence}\n")
                tmp_in_path = tmp_in.name

            tmp_out_path = tmp_in_path + ".out"

            cmd = [
                "blastn", "-query", tmp_in_path, "-db", db_path,
                "-out", tmp_out_path, "-outfmt", "6",
                "-task", "blastn-short",
                "-word_size", str(word_size), "-evalue", str(evalue),
                "-gapopen", str(gapopen), "-gapextend", str(gapextend),
                "-penalty", str(penalty), "-reward", str(reward),
            ]

            logger.info("Executing BLAST: %s", " ".join(cmd))
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=self.timeout)

            with open(tmp_out_path, "r") as f:
                raw_results = f.read()

            hits = parse_blast_tabular(raw_results)

            return json.dumps({
                "status": "success",
                "species": species,
                "query_length": len(sequence),
                "hits_count": len(hits),
                "params": {"word_size": word_size, "evalue": evalue, "task": "blastn-short"},
                "results": hits,
            }, ensure_ascii=False, indent=2)

        except subprocess.TimeoutExpired:
            return json.dumps({"status": "error", "error": f"BLAST timed out after {self.timeout}s"})
        except subprocess.CalledProcessError as e:
            logger.error("BLAST Error: %s", e.stderr)
            return json.dumps({"status": "error", "error": "BLAST execution failed", "details": e.stderr})
        except Exception as e:
            logger.exception("Unexpected error")
            return json.dumps({"status": "error", "error": "Internal error", "details": str(e)})
        finally:
            for p in [tmp_in_path, tmp_out_path]:
                if p and os.path.exists(p):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass


# ============================================================
# Singleton export
# ============================================================
insect_blast_tool = InsectBlastTool()

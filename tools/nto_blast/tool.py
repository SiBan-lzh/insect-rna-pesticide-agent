"""
nto_blast.py — BLASTN against NTO genomes for off-target checking.
"""

import os
import json
import subprocess
import tempfile
import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from .config import NTO_BLAST_DB

logger = logging.getLogger("RPA_Tools.NTOBlast")


# ============================================================
# Pydantic input schema
# ============================================================
class NTOBlastInput(BaseModel):
    """BLAST search parameters."""

    sequence: str = Field(description="dsRNA sequence (5'→3')")
    species: str = Field(description="NTO species (Latin, underscore-separated), e.g. Apis_mellifera. Multiple species separated by commas.")
    word_size: int = Field(default=11, description="BLAST word size")
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
class NTOBlastTool(BaseTool):
    """BLASTN against NTO genomes for off-target checking."""

    name: str = "nto_blast"
    description: str = (
        "BLASTN against non-target organism (NTO) genome. "
        "Input: dsRNA sequence + NTO species name (Latin). "
        "Output: match positions, identity %, E-value."
    )
    args_schema: type = NTOBlastInput

    timeout: int = 120

    def _run(
        self,
        sequence: str,
        species: str,
        word_size: int = 11,
        evalue: float = 1.0,
        gapopen: int = 10,
        gapextend: int = 4,
        penalty: int = -1,
        reward: int = 1,
    ) -> str:
        """Run BLAST (sync)."""
        db_path = os.path.join(NTO_BLAST_DB, species, species)
        if not os.path.exists(f"{db_path}.nsq"):
            return json.dumps({
                "status": "error",
                "error": f"NTO BLAST database not found for species '{species}'",
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
                "-task", "blastn",
                "-word_size", str(word_size), "-evalue", str(evalue),
                "-gapopen", str(gapopen), "-gapextend", str(gapextend),
                "-penalty", str(penalty), "-reward", str(reward),
            ]

            logger.info("Executing NTO BLAST: %s", " ".join(cmd))
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=self.timeout)

            with open(tmp_out_path, "r") as f:
                raw_results = f.read()

            hits = parse_blast_tabular(raw_results)

            risky_hits = [h for h in hits if h["identity_pct"] >= 80]
            off_target_risk = "high" if risky_hits else (
                "medium" if any(h["identity_pct"] >= 60 for h in hits) else "low"
            )

            return json.dumps({
                "status": "success",
                "species": species,
                "query_length": len(sequence),
                "hits_count": len(hits),
                "risky_hits_count": len(risky_hits),
                "off_target_risk": off_target_risk,
                "params": {"word_size": word_size, "evalue": evalue, "task": "blastn"},
                "results": hits,
            }, ensure_ascii=False, indent=2)

        except subprocess.TimeoutExpired:
            return json.dumps({"status": "error", "error": f"BLAST timed out after {self.timeout}s"})
        except subprocess.CalledProcessError as e:
            logger.error("NTO BLAST Error: %s", e.stderr)
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
nto_blast_tool = NTOBlastTool()
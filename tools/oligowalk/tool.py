"""
oligowalk.py -- siRNA thermodynamics scoring tool

Subprocess tool that calls the OligoWalk binary to compute binding free energy
between siRNA and target mRNA.
"""

import json
import logging
import os
import re
import subprocess
import tempfile
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from .config import OLIGOWALK_BIN, RNASTRUCTURE_DATAPATH

logger = logging.getLogger("RPA_Tools.OligoWalk")


# ============================================================
# Pydantic input schema
# ============================================================
class OligoWalkInput(BaseModel):
    """OligoWalk siRNA thermodynamics parameters."""

    sequence: str = Field(
        description="Target mRNA sequence (5'->3')"
    )
    run_type: str = Field(
        default="fast",
        description="Run mode: 'fast' (Mode=3, Suboptimal=1) / 'research' (Mode=2, Suboptimal=2)"
    )
    oligo_length: int = Field(
        default=21, ge=18, le=23,
        description="siRNA/oligonucleotide length (nt)"
    )
    mode: Optional[int] = Field(
        default=None, ge=1, le=3,
        description="Folding mode (research only; 1=single-strand, 2=double-strand, 3=fast)"
    )
    suboptimal: Optional[int] = Field(
        default=None, ge=0, le=4,
        description="Number of suboptimal structures (research only; 0=optimal only, 1-4=suboptimal count)"
    )
    filter: Optional[int] = Field(
        default=None, ge=0, le=1,
        description="GU wobble filter (0=no filter, 1=filter GU; fast default 0, research default 1)"
    )
    top_n: int = Field(
        default=10, ge=1,
        description="Number of top candidate siRNAs to return"
    )
    dna: bool = Field(
        default=False,
        description="Use DNA thermodynamics parameters (default False, uses RNA parameters)"
    )


# ============================================================
# Command builder
# ============================================================
def build_command(
    req: OligoWalkInput,
    seq_file: str,
    report_file: str,
) -> List[str]:
    """Build OligoWalk command line from input parameters."""
    cmd = [str(OLIGOWALK_BIN), seq_file, report_file]

    # Oligonucleotide length
    cmd.extend(["-l", str(req.oligo_length)])

    if req.run_type == "fast":
        # Fast mode: Mode=3, Suboptimal=1, no GU filter
        cmd.extend(["-m", "3"])
        cmd.extend(["-s", "1"])
        cmd.extend(["-fi", str(req.filter if req.filter is not None else 0)])
        cmd.append("-score")
    elif req.run_type == "research":
        # Research mode: Mode=2, Suboptimal=2, GU filter enabled
        cmd.extend(["-m", str(req.mode or 2)])
        cmd.extend(["-s", str(req.suboptimal or 2)])
        cmd.extend(["-fi", str(req.filter if req.filter is not None else 1)])
        cmd.append("-score")
    else:
        raise ValueError(f"Invalid run_type: '{req.run_type}', must be 'fast' or 'research'")

    if req.dna:
        cmd.append("-d")

    return cmd


# ============================================================
# Report parser
# ============================================================
def parse_report(report_path: str) -> dict:
    """Parse OligoWalk text report, extract parameters and energy table."""
    parameters: dict = {}
    energy_table: list = []
    parsing_energy: bool = False
    header_fields: list = []

    with open(report_path, "r") as f:
        for line in f:
            line = line.strip()

            # Total size of target
            if "Total size of the target" in line:
                m = re.search(r"(\d+)", line)
                if m:
                    parameters["total_size"] = int(m.group(1))

            # Scanned position range
            elif line.startswith("Scanned position on target:"):
                m = re.search(
                    r"Scanned position on target:\s*(\d+)\s*to\s*(\d+)", line
                )
                if m:
                    parameters["scan_start"] = int(m.group(1))
                    parameters["scan_end"] = int(m.group(2))

            elif line.startswith("Oligonucleotides:"):
                continue

            # Oligonucleotide length
            elif line.startswith("Length:"):
                m = re.search(r"Length:\s*(\d+)", line)
                if m:
                    parameters["oligo_length"] = int(m.group(1))

            # Method options (Mode / Folding region size / Suboptimal)
            elif (
                line.startswith("Mode:")
                or line.startswith("Folding region size:")
                or line.startswith("Suboptimal:")
            ):
                if "method_options" not in parameters:
                    parameters["method_options"] = {}
                key, val = line.split(":", 1)
                parameters["method_options"][key.strip()] = val.strip()

            # Detect energy table start
            if line.startswith("Energy table:"):
                parsing_energy = True
                continue

            if parsing_energy:
                # Parse header: "Pos." "Oligo" ...
                if line.startswith("Pos.") and "Oligo" in line:
                    header_fields = [h.strip() for h in re.split(r"\t+", line)]
                    continue
                # Skip empty or non-data lines
                if not line or not re.match(r"^\d+", line):
                    continue

                # Parse data rows (tab-separated or double-space separated)
                values = re.split(r"\t+", line)
                if len(values) != len(header_fields):
                    values = re.split(r"\s{2,}", line)
                if len(values) == len(header_fields):
                    row_dict = {
                        header_fields[i]: values[i].strip()
                        for i in range(len(header_fields))
                    }
                    energy_table.append(row_dict)

    return {
        "parameters": parameters,
        "energy_table": energy_table,
    }


# ============================================================
# LangChain Tool
# ============================================================
class OligoWalkTool(BaseTool):
    """siRNA thermodynamics scoring tool.

    Given a target mRNA sequence, uses the OligoWalk sliding-window algorithm
    to scan all possible siRNA binding sites, computes the thermodynamic binding
    free energy for each site, and returns a sorted list of top candidates.

    Supports fast mode (large-scale screening) and research mode (higher precision).
    """

    name: str = "oligowalk"
    description: str = (
        "siRNA thermodynamics scoring tool. Given a target mRNA sequence, uses "
        "the OligoWalk algorithm to scan all possible siRNA binding sites, compute "
        "the thermodynamic binding free energy (Overall Delta-G) for each candidate, "
        "and return a sorted list of top candidates. Supports fast and research modes."
    )
    args_schema: type = OligoWalkInput

    def _run(
        self,
        sequence: str,
        run_type: str = "fast",
        oligo_length: int = 21,
        mode: Optional[int] = None,
        suboptimal: Optional[int] = None,
        filter: Optional[int] = None,
        top_n: int = 10,
        dna: bool = False,
    ) -> str:
        """Run OligoWalk thermodynamics scoring (synchronous)."""

        try:
            # Build input object
            inp = OligoWalkInput(
                sequence=sequence,
                run_type=run_type,
                oligo_length=oligo_length,
                mode=mode,
                suboptimal=suboptimal,
                filter=filter,
                top_n=top_n,
                dna=dna,
            )

            # Ensure DATAPATH environment variable is set
            if "DATAPATH" not in os.environ:
                data_tables = RNASTRUCTURE_DATAPATH
                if data_tables.exists():
                    os.environ["DATAPATH"] = str(data_tables)
                else:
                    logger.warning(
                        "DATAPATH not set and data_tables not found at %s. "
                        "OligoWalk may fail to find thermodynamic parameters.",
                        data_tables,
                    )

            # Prepare temporary files
            with tempfile.TemporaryDirectory() as tmpdir:
                seq_file = os.path.join(tmpdir, "input.fa")
                report_file = os.path.join(tmpdir, "output.txt")

                # Write FASTA file
                with open(seq_file, "w") as f:
                    f.write(f">query\n{inp.sequence}\n")

                # Build command
                cmd = build_command(inp, seq_file, report_file)
                logger.info("Executing: %s", " ".join(cmd))

                # Execute OligoWalk
                timeout = 120 if run_type == "research" else 40
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )

                logger.info("STDOUT:\n%s", result.stdout)
                if result.stderr:
                    logger.info("STDERR:\n%s", result.stderr)

                # Parse report
                data = parse_report(report_file)

                # Check execution status
                if result.returncode != 0:
                    logger.error("OligoWalk execution failed (rc=%d)", result.returncode)
                    return json.dumps(
                        {
                            "status": "error",
                            "error": "oligowalk execution failed",
                            "parameters": data.get("parameters", {}),
                            "stderr": result.stderr,
                            "stdout": result.stdout,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )

                # Sort by Overall binding free energy ascending (more negative = more stable)
                sorted_table = sorted(
                    data["energy_table"],
                    key=lambda x: float(x.get("Overall (kcal/mol)", 0)),
                )

                top_candidates = sorted_table[:top_n]

                return json.dumps(
                    {
                        "status": "success",
                        "mode": run_type,
                        "count": len(sorted_table),
                        "top_candidates": top_candidates,
                        "parameters": data["parameters"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )

        except subprocess.TimeoutExpired:
            logger.error("OligoWalk execution timeout")
            return json.dumps(
                {
                    "status": "error",
                    "error": "execution timeout",
                    "details": "OligoWalk runtime exceeded allowed time",
                },
                ensure_ascii=False,
                indent=2,
            )

        except FileNotFoundError:
            logger.exception("OligoWalk binary not found")
            return json.dumps(
                {
                    "status": "error",
                    "error": "OligoWalk binary not found",
                    "details": (
                        "OligoWalk is not installed. "
                        "Please compile RNAstructure from database/RNAstructure/ "
                        "and ensure OligoWalk is in your PATH."
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )

        except Exception as e:
            logger.exception("OligoWalk failed")
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
oligowalk_tool = OligoWalkTool()

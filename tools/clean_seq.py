"""
clean_seq.py — Sequence normalization and validation tool.

The LLM calls this tool to clean raw user-provided sequences (FASTA or plain)
into standardized format: single-line, uppercase, validated alphabet.
Always call this as the FIRST step when a user provides a sequence.

Usage:
    from tools.clean_seq import clean_seq_tool

    result = clean_seq_tool.invoke({
        "sequence": ">seq\\nATGCGATCG\\natcgttaa",
        "seq_type": "dna"
    })
"""

from __future__ import annotations

import re
import logging
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

logger = logging.getLogger("RPA_Tools.CleanSeq")

# ============================================================
# Pydantic input schema
# ============================================================
class CleanSeqInput(BaseModel):
    """Sequence cleaning parameters."""

    sequence: str = Field(
        description="Raw input sequence. Accepts FASTA format (header line starting with '>') "
        "or plain sequence with optional whitespace/newlines."
    )
    seq_type: str = Field(
        description="Sequence type for normalization: 'dna' or 'rna'. "
        "DNA mode: converts to uppercase ATCG (U→T, u→T). "
        "RNA mode: converts to uppercase AUCG (T→U, t→U)."
    )


# ============================================================
# Core function
# ============================================================
def clean_sequence(sequence: str, seq_type: str = "dna") -> dict:
    """Normalize and validate a raw sequence.

    Steps:
    1. Strip FASTA header if present (line starting with '>')
    2. Remove all whitespace and newlines
    3. Validate against allowed alphabet (ATCGUN for DNA, AUCGN for RNA)
    4. Convert case and apply DNA↔RNA transformations
    5. Return cleaned single-line sequence with metadata

    Args:
        sequence: Raw input (FASTA or plain)
        seq_type: 'dna' or 'rna'

    Returns:
        dict with cleaned sequence, length, or error details.
    """
    if not sequence or not sequence.strip():
        return {
            "status": "error",
            "error": "Input sequence is empty. Please provide a DNA or RNA sequence.",
        }

    seq_type = seq_type.lower().strip()
    if seq_type not in ("dna", "rna"):
        return {
            "status": "error",
            "error": f"Invalid seq_type '{seq_type}'. Must be 'dna' or 'rna'.",
        }

    # Step 1: Strip FASTA header
    raw = sequence.strip()
    lines = raw.splitlines()
    if lines and lines[0].startswith(">"):
        header = lines[0]
        raw = "".join(lines[1:])
    else:
        header = None
        raw = "".join(lines)

    # Step 2: Remove all whitespace
    raw = re.sub(r"\s+", "", raw)

    if not raw:
        return {
            "status": "error",
            "error": "Sequence is empty after removing FASTA header and whitespace.",
        }

    # Step 3: Validate characters
    allowed = set("ATCGUNatcgun") if seq_type == "dna" else set("AUCGNaucgn")
    invalid_positions = []
    for i, ch in enumerate(raw, start=1):
        if ch not in allowed:
            invalid_positions.append((i, ch))

    if invalid_positions:
        detail = "; ".join(f"bp {pos}: '{char}'" for pos, char in invalid_positions[:20])
        if len(invalid_positions) > 20:
            detail += f"; ... and {len(invalid_positions) - 20} more"
        return {
            "status": "error",
            "error": f"Invalid character(s) found at {len(invalid_positions)} position(s).",
            "invalid_positions": detail,
            "hint": "Please provide a sequence containing only A, T, C, G (and N for ambiguous bases). "
                    "Remove any numbers, special characters, or non-nucleotide symbols.",
        }

    # Step 4: Normalize
    cleaned = raw.upper()
    original_length = len(cleaned)

    if seq_type == "dna":
        # U → T
        u_count = cleaned.count("U")
        cleaned = cleaned.replace("U", "T")
        changes = []
        if u_count:
            changes.append(f"{u_count} U→T substitution(s)")

        # Re-validate after U→T
        allowed_dna = set("ATCGN")
        invalid_after = [(i + 1, c) for i, c in enumerate(cleaned) if c not in allowed_dna]
        if invalid_after:
            detail = "; ".join(f"bp {pos}: '{char}'" for pos, char in invalid_after[:20])
            return {
                "status": "error",
                "error": f"Invalid character(s) after DNA normalization at {len(invalid_after)} position(s).",
                "invalid_positions": detail,
            }
    else:  # rna
        # T → U
        t_count = cleaned.count("T")
        cleaned = cleaned.replace("T", "U")
        changes = []
        if t_count:
            changes.append(f"{t_count} T→U substitution(s)")

        # Re-validate after T→U
        allowed_rna = set("AUCGN")
        invalid_after = [(i + 1, c) for i, c in enumerate(cleaned) if c not in allowed_rna]
        if invalid_after:
            detail = "; ".join(f"bp {pos}: '{char}'" for pos, char in invalid_after[:20])
            return {
                "status": "error",
                "error": f"Invalid character(s) after RNA normalization at {len(invalid_after)} position(s).",
                "invalid_positions": detail,
            }

    # Compute GC content
    gc = sum(1 for b in cleaned if b in ("G", "C"))
    gc_pct = round(gc / len(cleaned) * 100, 1) if cleaned else 0.0

    result = {
        "status": "success",
        "cleaned_sequence": cleaned,
        "length": len(cleaned),
        "seq_type": seq_type.upper(),
        "gc_content": gc_pct,
        "fasta_header": header or None,
        "original_length": original_length,
    }

    if changes:
        result["normalization_notes"] = "; ".join(changes)

    return result


# ============================================================
# LangChain Tool
# ============================================================
class CleanSeqTool(BaseTool):
    """Sequence normalization and validation tool.

    Cleans raw user-provided sequences (FASTA or plain text) into
    standardized single-line format with validated nucleotide alphabet.
    Detects and reports invalid characters with exact position(s).
    """

    name: str = "clean_seq"
    description: str = (
        "Normalize and validate raw nucleotide sequences. "
        "Always call this as the FIRST step when a user provides a sequence. "
        "Accepts FASTA format (with '>' header) or plain sequences. "
        "Removes whitespace/newlines, converts to single-line uppercase, "
        "validates against DNA (ATCGN) or RNA (AUCGN) alphabet, and reports "
        "invalid characters with exact bp positions. "
        "DNA mode converts U→T; RNA mode converts T→U."
    )
    args_schema: type[BaseModel] = CleanSeqInput

    def _run(self, sequence: str, seq_type: str = "dna") -> str:
        """Clean and validate sequence (sync)."""
        import json
        try:
            result = clean_sequence(sequence, seq_type)
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.exception("clean_seq failed")
            return json.dumps({
                "status": "error",
                "error": f"Internal error: {e}",
            }, ensure_ascii=False, indent=2)

    async def _arun(self, sequence: str, seq_type: str = "dna") -> str:
        """Async variant."""
        return self._run(sequence, seq_type)


clean_seq_tool = CleanSeqTool()

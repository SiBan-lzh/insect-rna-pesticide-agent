"""
clip_seq.py -- Sequence fragment clipping tool

Pure Python tool: clips a sub-sequence from a CDS/gene by coordinates.
Supports sense and antisense (reverse complement) output.
"""

import json
import logging
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

logger = logging.getLogger("RPA_Tools.ClipSeq")

# ============================================================
# Complement mapping
# ============================================================
_COMPLEMENT = str.maketrans("ATGCatgc", "TACGtacg")


# ============================================================
# Pydantic input schema
# ============================================================
class ClipSeqInput(BaseModel):
    """Parameters for sequence clipping."""

    sequence: str = Field(
        description="Input DNA sequence (5'->3'), typically full CDS"
    )
    start: int = Field(
        default=1, ge=1,
        description="Clip start position (1-based), default 1"
    )
    length: int = Field(
        default=300, ge=50, le=1000,
        description="Clip length (bp), default 300. dsRNA target typically 200-500 bp"
    )
    as_reverse_complement: bool = Field(
        default=False,
        description="Return reverse complement (for antisense design)"
    )


# ============================================================
# Core function
# ============================================================
def compute_gc_content(seq: str) -> float:
    """Compute GC content (%)."""
    if not seq:
        return 0.0
    gc = sum(1 for base in seq.upper() if base in ("G", "C"))
    return round(gc / len(seq) * 100, 1)


def reverse_complement(seq: str) -> str:
    """Return reverse complement (5'->3')."""
    return seq.translate(_COMPLEMENT)[::-1]


def clip_sequence(
    sequence: str,
    start: int = 1,
    length: int = 300,
    as_reverse_complement: bool = False,
) -> dict:
    """Clip a sub-sequence from a DNA sequence at given coordinates.

    Args:
        sequence: Input sequence (5'->3')
        start: Start position (1-based)
        length: Clip length (bp)
        as_reverse_complement: Return reverse complement

    Returns:
        Dict with clipped fragment and metadata
    """
    seq = sequence.strip().upper()
    seq = "".join(seq.split())
    seq_len = len(seq)

    # 1-based -> 0-based
    zero_start = start - 1
    zero_end = zero_start + length

    if zero_start < 0 or zero_start >= seq_len:
        return {
            "status": "error",
            "error": f"Start position {start} is out of range (seq length: {seq_len})",
        }

    actual_end = min(zero_end, seq_len)
    clipped = seq[zero_start:actual_end]
    actual_length = len(clipped)
    truncated = actual_length < length

    result_seq = reverse_complement(clipped) if as_reverse_complement else clipped

    return {
        "status": "success",
        "original_length": seq_len,
        "clip_start": start,
        "clip_end": actual_end,
        "requested_length": length,
        "actual_length": actual_length,
        "truncated": truncated,
        "gc_content": compute_gc_content(result_seq),
        "strand": "antisense" if as_reverse_complement else "sense",
        "sequence": result_seq,
    }


# ============================================================
# LangChain Tool
# ============================================================
class ClipSeqTool(BaseTool):
    """Clip a sub-sequence by coordinates from a CDS/full-length gene.

    Used in RNAi target design for dsRNA fragment preparation.

    Typical dsRNA target requirements:
    - Length 200-500 bp (default 300 bp)
    - Within exon region
    - GC content 40-60%

    Pipeline: fetch_insect_cds -> clip_seq -> primer3/oligowalk
    """

    name: str = "clip_seq"
    description: str = (
        "Clip a sub-sequence from a DNA sequence (typically CDS) "
        "by start position and length for downstream dsRNA design. "
        "Supports sense or antisense output. "
        "Automatically computes GC content. Default: 300 bp sense fragment."
    )
    args_schema: type = ClipSeqInput

    def _run(
        self,
        sequence: str,
        start: int = 1,
        length: int = 300,
        as_reverse_complement: bool = False,
    ) -> str:
        """Clip sequence fragment (sync)."""

        if not sequence or not sequence.strip():
            return json.dumps({
                "status": "error",
                "error": "Input sequence is empty.",
            }, ensure_ascii=False, indent=2)

        try:
            result = clip_sequence(sequence, start, length, as_reverse_complement)
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.exception("clip_seq failed")
            return json.dumps({
                "status": "error",
                "error": f"Internal error: {e}",
            }, ensure_ascii=False, indent=2)


# ============================================================
# Singleton export
# ============================================================
clip_seq_tool = ClipSeqTool()

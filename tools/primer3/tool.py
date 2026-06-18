"""
primer3.py -- PCR primer design tool

Pure Python tool using primer3-py library, no subprocess/external binary dependency.
"""

import json
import logging
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
import primer3

logger = logging.getLogger("RPA_Tools.Primer3")

# ============================================================
# T7 promoter prefix (for in vitro transcription)
# ============================================================
T7_PREFIX = "TAATACGACTCACTATAGGG"


# ============================================================
# Pydantic input schema
# ============================================================
class Primer3Input(BaseModel):
    """PCR primer design parameters."""

    sequence: str = Field(
        description="Template DNA sequence (5'->3')"
    )
    sequence_id: str = Field(
        default="query",
        description="Sequence identifier"
    )
    num_return: int = Field(
        default=3, ge=1, le=5,
        description="Number of primer pairs to return"
    )

    # Target region (optional, specifies amplicon interval)
    target_start: Optional[int] = Field(
        default=None,
        description="Target start position on sequence (1-based)"
    )
    target_len: Optional[int] = Field(
        default=None,
        description="Target region length (bp)"
    )

    # Primer length
    primer_opt_size: int = Field(default=20, ge=10, le=30)
    primer_min_size: int = Field(default=18, ge=10, le=30)
    primer_max_size: int = Field(default=25, ge=10, le=35)

    # GC content
    primer_opt_gc_percent: float = Field(default=50.0, ge=20.0, le=80.0)
    primer_min_gc: float = Field(default=40.0, ge=20.0, le=80.0)
    primer_max_gc: float = Field(default=60.0, ge=20.0, le=80.0)

    # Annealing temperature
    primer_opt_tm: float = Field(default=60.0, ge=50.0, le=70.0)
    primer_min_tm: float = Field(default=57.0, ge=45.0, le=70.0)
    primer_max_tm: float = Field(default=63.0, ge=50.0, le=75.0)

    # Product size range
    primer_product_size_range: List[int] = Field(
        default_factory=lambda: [100, 300],
        description="Expected PCR product size range [min, max]"
    )


# ============================================================
# Core function
# ============================================================
def build_primer3_args(inp: Primer3Input):
    """Convert Pydantic input to primer3-py library parameter format."""
    seq_args = {
        "SEQUENCE_ID": inp.sequence_id,
        "SEQUENCE_TEMPLATE": inp.sequence,
    }

    product_range = list(inp.primer_product_size_range)

    if inp.target_start is not None and inp.target_len is not None:
        seq_args["SEQUENCE_TARGET"] = [inp.target_start, inp.target_len]
        if product_range[0] < inp.target_len:
            product_range[0] = inp.target_len

    global_args = {
        "PRIMER_OPT_SIZE": inp.primer_opt_size,
        "PRIMER_MIN_SIZE": inp.primer_min_size,
        "PRIMER_MAX_SIZE": inp.primer_max_size,
        "PRIMER_OPT_GC_PERCENT": inp.primer_opt_gc_percent,
        "PRIMER_MIN_GC": inp.primer_min_gc,
        "PRIMER_MAX_GC": inp.primer_max_gc,
        "PRIMER_OPT_TM": inp.primer_opt_tm,
        "PRIMER_MIN_TM": inp.primer_min_tm,
        "PRIMER_MAX_TM": inp.primer_max_tm,
        "PRIMER_PRODUCT_SIZE_RANGE": [product_range],
        "PRIMER_NUM_RETURN": inp.num_return,
    }

    return seq_args, global_args


def run_primer3_design(seq_args: dict, global_args: dict) -> dict:
    """Call primer3-py to design primers, return structured result."""
    results = primer3.bindings.design_primers(seq_args, global_args)

    if "PRIMER_LEFT_0_SEQUENCE" not in results:
        return {"status": "failed", "message": "No primers found"}

    primers_list = []
    actual_found = results.get("PRIMER_PAIR_NUM_RETURNED", 0)

    for i in range(actual_found):
        f_seq = results[f"PRIMER_LEFT_{i}_SEQUENCE"]
        r_seq = results[f"PRIMER_RIGHT_{i}_SEQUENCE"]

        primers_list.append({
            "pair_index": i + 1,
            "forward": f_seq,
            "reverse": r_seq,
            "forward_with_T7": T7_PREFIX + f_seq,
            "reverse_with_T7": T7_PREFIX + r_seq,
            "tm_left": round(results[f"PRIMER_LEFT_{i}_TM"], 2),
            "tm_right": round(results[f"PRIMER_RIGHT_{i}_TM"], 2),
            "product_size": results[f"PRIMER_PAIR_{i}_PRODUCT_SIZE"],
            "penalty": round(results[f"PRIMER_PAIR_{i}_PENALTY"], 4),
        })

    return {"status": "success", "count": actual_found, "primers": primers_list}


# ============================================================
# LangChain Tool
# ============================================================
class Primer3Tool(BaseTool):
    """Design PCR primers for amplifying target gene fragments.

    Takes template DNA sequence and primer design parameters (length, GC content, Tm, etc.),
    returns candidate primer pairs with thermodynamic properties,
    including T7-promoter-tagged versions (for in vitro dsRNA transcription).
    """

    name: str = "primer3"
    description: str = (
        "Design PCR primers. Input template DNA sequence (5'->3'), "
        "returns candidate primer pairs (forward/reverse, Tm, product size, penalty), "
        "plus T7-promoter-tagged versions (for in vitro dsRNA transcription). "
        "Parameters: primer length, GC content, Tm range, product size, etc."
    )
    args_schema: type = Primer3Input

    def _run(
        self,
        sequence: str,
        sequence_id: str = "query",
        num_return: int = 3,
        target_start: Optional[int] = None,
        target_len: Optional[int] = None,
        primer_opt_size: int = 20,
        primer_min_size: int = 18,
        primer_max_size: int = 25,
        primer_opt_gc_percent: float = 50.0,
        primer_min_gc: float = 40.0,
        primer_max_gc: float = 60.0,
        primer_opt_tm: float = 60.0,
        primer_min_tm: float = 57.0,
        primer_max_tm: float = 63.0,
        primer_product_size_range: Optional[List[int]] = None,
    ) -> str:
        """Design PCR primers (synchronous)."""

        if primer_product_size_range is None:
            primer_product_size_range = [100, 300]

        try:
            inp = Primer3Input(
                sequence=sequence,
                sequence_id=sequence_id,
                num_return=num_return,
                target_start=target_start,
                target_len=target_len,
                primer_opt_size=primer_opt_size,
                primer_min_size=primer_min_size,
                primer_max_size=primer_max_size,
                primer_opt_gc_percent=primer_opt_gc_percent,
                primer_min_gc=primer_min_gc,
                primer_max_gc=primer_max_gc,
                primer_opt_tm=primer_opt_tm,
                primer_min_tm=primer_min_tm,
                primer_max_tm=primer_max_tm,
                primer_product_size_range=primer_product_size_range,
            )

            seq_args, global_args = build_primer3_args(inp)
            result = run_primer3_design(seq_args, global_args)

            return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.exception("Primer3 design failed")
            return json.dumps({
                "status": "error",
                "error": "primer3 execution failed",
                "details": str(e),
            }, ensure_ascii=False, indent=2)


# ============================================================
# Singleton export
# ============================================================
primer3_tool = Primer3Tool()

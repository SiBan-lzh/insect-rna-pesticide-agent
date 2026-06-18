"""
insect_anno.py -- BLAST hit genome annotation tool

subprocess tool wrapping bedtools window to find gene features
(gene/mRNA/exon/CDS/UTR) near BLAST hit positions in GFF3 files.
"""

import json
import logging
import os
import subprocess
import tempfile
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from .config import INSECT_GFF3_DB

logger = logging.getLogger("RPA_Tools.InsectAnno")

# ============================================================
# GFF3 feature type labels
# ============================================================
FEATURE_TYPES = {
    "gene": "gene",
    "mRNA": "mRNA",
    "exon": "exon",
    "CDS": "CDS",
    "five_prime_UTR": "5'UTR",
    "three_prime_UTR": "3'UTR",
}


# ============================================================
# Pydantic input schema
# ============================================================
class BlastHitInput(BaseModel):
    """Single BLAST hit site info."""

    chromosome: str = Field(
        description="Chromosome/scaffold ID, e.g. BMSK_chr25"
    )
    start_position: int = Field(
        description="BLAST match start on chromosome (1-based)"
    )
    end_position: int = Field(
        description="BLAST match end on chromosome (1-based)"
    )
    name: str = Field(
        default="hit",
        description="Hit label to distinguish hits in results"
    )
    score: float = Field(
        default=0.0,
        description="BLAST bit score"
    )
    strand: str = Field(
        default="+",
        description="Strand direction (+ or -)"
    )


class InsectAnnoInput(BaseModel):
    """Genome annotation query parameters. Typically receives insect_blast output as input."""

    blast_hits: List[BlastHitInput] = Field(
        description="List of BLAST hit records, each with chromosome, start, end, etc."
    )
    species: str = Field(
        description="Species name, e.g. Bombyx_mori"
    )
    window_size: int = Field(
        default=100,
        ge=0,
        description="Annotation window (bp) extending around each hit"
    )


# ============================================================
# Core function
# ============================================================
def get_full_chromosome_id(gff3_path: str, simple_chrom_id: str) -> str:
    """Resolve full chromosome ID from GFF3 file using pure Python."""
    try:
        with open(gff3_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                cols = line.split("\t", 1)
                if cols[0] == simple_chrom_id:
                    return cols[0]
        return simple_chrom_id
    except Exception as e:
        logger.warning("ID resolution failed for %s: %s", simple_chrom_id, e)
        return simple_chrom_id


def build_bed_content(hits: List[BlastHitInput]) -> str:
    """Convert BLAST hit list to BED format text.

    BED format: chrom start(0-based) end name score strand
    """
    lines = []
    for hit in hits:
        start_0based = hit.start_position - 1
        lines.append(
            f"{hit.chromosome}\t{start_0based}\t{hit.end_position}"
            f"\t{hit.name}\t{hit.score}\t{hit.strand}"
        )
    return "\n".join(lines)


def run_bedtools_window(
    bed_content: str,
    gff3_path: str,
    window_size: int,
) -> str:
    """Find overlapping GFF3 annotations for BED regions via bedtools window."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".bed", delete=False
    ) as tmp:
        tmp.write(bed_content)
        tmp_path = tmp.name

    try:
        cmd = [
            "bedtools",
            "window",
            "-a", tmp_path,
            "-b", gff3_path,
            "-w", str(window_size),
        ]

        logger.info("Running: %s", " ".join(cmd))

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        return result.stdout

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def parse_attributes(attr_string: str) -> Dict[str, str]:
    """Parse GFF3 column 9 attribute string.

    e.g. "ID=Bmor000001.1;Parent=Bmor000001" -> {"ID": "Bmor000001.1", "Parent": "Bmor000001"}
    """
    attrs = {}
    for item in attr_string.split(";"):
        if "=" in item:
            key, val = item.split("=", 1)
            attrs[key.strip()] = val.strip()
    return attrs


# ============================================================
# LangChain Tool
# ============================================================
class InsectAnoTool(BaseTool):
    """Genome annotation tool.

    Accepts BLAST hit list and looks up nearby gene features in a species
    GFF3 annotation file. Typically chained after insect_blast to determine
    target gene structure (exons, introns, CDS, etc.).
    """

    name: str = "insect_anno"
    description: str = (
        "Genome annotation tool. Given a list of BLAST hits (chromosome, start, end), "
        "queries the species GFF3 annotation file for nearby gene features "
        "(gene, mRNA, exon, CDS, UTR). Typically used after insect_blast: "
        "BLAST to find homologous sites, then annotate their gene structure."
    )
    args_schema: type = InsectAnnoInput

    def _run(
        self,
        blast_hits: List[dict],
        species: str,
        window_size: int = 100,
    ) -> str:
        """Annotate genome features from BLAST hits (synchronous)."""

        try:
            hits = blast_hits

            gff3_path = os.path.join(
                str(INSECT_GFF3_DB), species, f"{species}.gff3"
            )

            if not os.path.exists(gff3_path):
                return json.dumps(
                    {
                        "status": "error",
                        "error": f"GFF3 file not found for species '{species}'",
                        "expected_path": gff3_path,
                    },
                    ensure_ascii=False,
                    indent=2,
                )

            for hit in hits:
                hit.chromosome = get_full_chromosome_id(gff3_path, hit.chromosome)

            bed_content = build_bed_content(hits)
            raw_output = run_bedtools_window(bed_content, gff3_path, window_size)

            results: Dict[str, dict] = {}
            if raw_output.strip():
                for line in raw_output.strip().split("\n"):
                    cols = line.split("\t")
                    if len(cols) < 15:
                        continue

                    query_name = cols[3]

                    if query_name not in results:
                        results[query_name] = {
                            "blast_hit": query_name,
                            "features": [],
                        }

                    feature = {
                        "type": cols[8],
                        "start": cols[9],
                        "end": cols[10],
                        "strand": cols[12],
                        "attributes": parse_attributes(cols[14]),
                    }
                    results[query_name]["features"].append(feature)

            type_counts: Dict[str, int] = {}
            for r in results.values():
                for feat in r["features"]:
                    t = feat["type"]
                    type_counts[t] = type_counts.get(t, 0) + 1

            return json.dumps(
                {
                    "status": "success",
                    "species": species,
                    "gff3_file": gff3_path,
                    "hits_annotated": len(results),
                    "total_features": sum(type_counts.values()),
                    "feature_types": type_counts,
                    "results": list(results.values()),
                },
                ensure_ascii=False,
                indent=2,
            )

        except subprocess.CalledProcessError as e:
            logger.exception("bedtools window failed")
            return json.dumps(
                {
                    "status": "error",
                    "error": "bedtools execution failed",
                    "stderr": e.stderr,
                    "stdout": e.stdout,
                },
                ensure_ascii=False,
                indent=2,
            )

        except FileNotFoundError:
            logger.exception("bedtools not found")
            return json.dumps(
                {
                    "status": "error",
                    "error": "bedtools binary not found",
                    "details": "Please install bedtools: sudo apt install bedtools",
                },
                ensure_ascii=False,
                indent=2,
            )

        except Exception as e:
            logger.exception("Insect annotation failed")
            return json.dumps(
                {
                    "status": "error",
                    "error": "annotation failed",
                    "details": str(e),
                },
                ensure_ascii=False,
                indent=2,
            )


# ============================================================
# Singleton export
# ============================================================
insect_anno_tool = InsectAnoTool()

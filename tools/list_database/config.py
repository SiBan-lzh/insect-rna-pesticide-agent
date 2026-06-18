"""
list_database/config.py — All database paths for data discovery tool.
"""
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(".env", usecwd=True), override=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATABASE_ROOT = Path(os.getenv("DATABASE_ROOT", PROJECT_ROOT / "database"))
INSECT_BLAST_DB = Path(os.getenv("INSECT_BLAST_DB", DATABASE_ROOT / "blast" / "insect_blastndb"))
NTO_BLAST_DB = Path(os.getenv("NTO_BLAST_DB", DATABASE_ROOT / "blast" / "nto_blastndb"))
INSECT_GFF3_DB = Path(os.getenv("INSECT_GFF3_DB", DATABASE_ROOT / "annotation" / "insect_gff3"))
INSECT_CDS_DB = Path(os.getenv("INSECT_CDS_DB", DATABASE_ROOT / "refseq" / "insect_cds"))
NTOS_REFSEQ_DB = Path(os.getenv("NTOS_REFSEQ_DB", DATABASE_ROOT / "refseq" / "nto_refseq"))
KINSHIP_DB = Path(os.getenv("KINSHIP_DB", DATABASE_ROOT / "kinship"))
RNASTRUCTURE_HOME = Path(os.getenv("RNASTRUCTURE_HOME", DATABASE_ROOT / "RNAstructure"))

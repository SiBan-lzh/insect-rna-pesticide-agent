"""
migrate_tools.py — 批量迁移工具到子文件夹结构。

用法：
    python scripts/migrate_tools.py

前置条件：
    tools/*.py.bak 备份文件已就位（所有原始 .py 会被保留）
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

TOOLS_DIR = Path("/home/lizonghuan/langgraph/tools")


# ============================================================
# 每个工具的定义
# ============================================================
# config_source:
#   None         — 不需要 config.py
#   "inline"     — 直接在此定义 config 内容
#   "skip"       — 手动处理（fetch_seq 拆分）
#
# metadata 只记录 name / tag / description，不记录参数细节

TOOL_DEFS = [
    # ── 不需要 config.py ──
    {
        "name": "clip_seq",
        "tag": "sequence processing",
        "description": "从 CDS/基因序列中按坐标截取目标片段，用于 dsRNA 靶标区域的精确提取。支持正链和反链互补。",
        "config_source": None,
        "bak_file": "clip_seq.py.bak",
    },
    {
        "name": "clustal",
        "tag": "safety assessment",
        "description": "Pairwise alignment and continuous-match analysis via Clustal Omega. Detects continuous 21nt matches that indicate off-target risk.",
        "config_source": None,
        "bak_file": "clustal.py.bak",
    },
    {
        "name": "primer3",
        "tag": "dsrna design",
        "description": "PCR 引物设计工具。调用 primer3-py 库设计 PCR 引物，支持 T7 启动子添加、跨内含子引物设计、多重 PCR 引物组合设计。",
        "config_source": None,
        "bak_file": "primer3.py.bak",
    },
    {
        "name": "list_skills",
        "tag": "discovery",
        "description": "List available skill document metadata by category. Call this to discover what skill documents are available.",
        "config_source": None,
        "bak_file": "list_skills.py.bak",
    },
    {
        "name": "list_tools",
        "tag": "discovery",
        "description": "List available bioinformatics tools and their parameter schemas. Call this to discover what tools are available or inspect a specific tool's parameters.",
        "config_source": None,
        "bak_file": "list_tools.py.bak",
    },
    {
        "name": "read_skill",
        "tag": "discovery",
        "description": "Load the full content of a skill document by name. Use list_skills first to discover available skill names.",
        "config_source": None,
        "bak_file": "read_skill.py.bak",
    },
    # ── 需要 config.py ──
    {
        "name": "nto_blast",
        "tag": "safety assessment",
        "description": "BLASTN against non-target organism (NTO) genomes for off-target checking. Input: dsRNA sequence + NTO species name. Output: match positions, identity %, E-value.",
        "config_source": "inline",
        "bak_file": "nto_blast.py.bak",
    },
    {
        "name": "insect_anno",
        "tag": "target discovery",
        "description": "Annotate BLAST hit positions with genome annotation (GFF3). Uses bedtools to find nearby gene features (gene/mRNA/exon/CDS/UTR).",
        "config_source": "inline",
        "bak_file": "insect_anno.py.bak",
    },
    {
        "name": "kinship",
        "tag": "phylogenetic analysis",
        "description": "Calculate phylogenetic distance and divergence time between insect species using ETE4 tree and NCBI taxonomy database.",
        "config_source": "inline",
        "bak_file": "kinship.py.bak",
    },
    {
        "name": "oligowalk",
        "tag": "dsrna design",
        "description": "siRNA thermostability scoring via OligoWalk. Calculates binding free energy (ΔG) between siRNA and target mRNA for fragment design.",
        "config_source": "inline",
        "bak_file": "oligowalk.py.bak",
    },
    {
        "name": "pubmed_esearch",
        "tag": "literature search",
        "description": "Search PubMed database via NCBI E-utilities ESearch API. Returns matching PMID list with optional year filtering.",
        "config_source": "inline",
        "bak_file": "pubmed_esearch.py.bak",
    },
    {
        "name": "pubmed_efetch",
        "tag": "literature search",
        "description": "Fetch PubMed article details (title, abstract, DOI, MeSH) by PMID via NCBI E-utilities EFetch API.",
        "config_source": "inline",
        "bak_file": "pubmed_efetch.py.bak",
    },
    {
        "name": "openalex_search",
        "tag": "literature search",
        "description": "Search OpenAlex open academic literature database by keywords, field filters, and year range. Free API, no key required.",
        "config_source": "inline",
        "bak_file": "openalex_search.py.bak",
    },
    {
        "name": "search_knowledge",
        "tag": "discovery",
        "description": "Search an indexed knowledge base using hybrid semantic + keyword retrieval. Use list_ragbase first to see available knowledge bases.",
        "config_source": "inline",
        "bak_file": "search_knowledge.py.bak",
    },
    {
        "name": "list_ragbase",
        "tag": "discovery",
        "description": "List available knowledge bases (ragbase collections) and their metadata. Call this first before using search_knowledge.",
        "config_source": "inline",
        "bak_file": "list_ragbase.py.bak",
    },
    {
        "name": "list_database",
        "tag": "discovery",
        "description": "Check whether specific bioinformatics data exists in the local database. Call this BEFORE running analysis tools to verify data availability.",
        "config_source": "inline",
        "bak_file": "list_database.py.bak",
    },
    # ── 从 fetch_seq.py.bak 拆分 ──
    {
        "name": "fetch_nto_seq",
        "tag": "safety assessment",
        "description": "Extract sequence regions from NTO reference genome using samtools faidx. Input: species + hit list (subject_id, start, end). Supports reverse complement.",
        "config_source": "inline",
        "bak_file": "fetch_seq.py.bak",
        "split": "nto",  # keep only fetch_nto_seq parts
    },
    {
        "name": "fetch_insect_cds",
        "tag": "target discovery",
        "description": "Search insect CDS database by transcript ID. Input: species + transcript ID (from insect_anno). Pure Python file search, no external tools.",
        "config_source": "inline",
        "bak_file": "fetch_seq.py.bak",
        "split": "cds",  # keep only fetch_insect_cds parts
    },
]


# ============================================================
# Config templates (from tool_config.py)
# ============================================================

PROJ_ROOT = "Path(__file__).resolve().parent.parent.parent"

CONFIG_TEMPLATES = {
    "nto_blast": f'''"""
nto_blast/config.py — NTO BLAST database path configuration.
"""
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(".env", usecwd=True), override=True)

PROJECT_ROOT = {PROJ_ROOT}
DATABASE_ROOT = Path(os.getenv("DATABASE_ROOT", PROJECT_ROOT / "database"))
NTO_BLAST_DB = Path(os.getenv("NTO_BLAST_DB", DATABASE_ROOT / "blast" / "nto_blastndb"))
''',
    "insect_anno": f'''"""
insect_anno/config.py — GFF3 annotation database path configuration.
"""
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(".env", usecwd=True), override=True)

PROJECT_ROOT = {PROJ_ROOT}
DATABASE_ROOT = Path(os.getenv("DATABASE_ROOT", PROJECT_ROOT / "database"))
INSECT_GFF3_DB = Path(os.getenv("INSECT_GFF3_DB", DATABASE_ROOT / "annotation" / "insect_gff3"))
''',
    "kinship": f'''"""
kinship/config.py — Species tree and taxonomy database paths.
"""
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(".env", usecwd=True), override=True)

PROJECT_ROOT = {PROJ_ROOT}
DATABASE_ROOT = Path(os.getenv("DATABASE_ROOT", PROJECT_ROOT / "database"))
KINSHIP_DB = Path(os.getenv("KINSHIP_DB", DATABASE_ROOT / "kinship"))
INSECT_TREE_PATH = Path(os.getenv("INSECT_TREE_PATH", KINSHIP_DB / "insects_species.nwk"))
INSECT_TAXA_DB_PATH = Path(os.getenv("INSECT_TAXA_DB_PATH", KINSHIP_DB / "taxa.sqlite"))
''',
    "oligowalk": f'''"""
oligowalk/config.py — RNAstructure OligoWalk path configuration.
"""
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(".env", usecwd=True), override=True)

PROJECT_ROOT = {PROJ_ROOT}
DATABASE_ROOT = Path(os.getenv("DATABASE_ROOT", PROJECT_ROOT / "database"))
RNASTRUCTURE_HOME = Path(os.getenv("RNASTRUCTURE_HOME", DATABASE_ROOT / "RNAstructure"))
OLIGOWALK_BIN = Path(os.getenv("OLIGOWALK_BIN", RNASTRUCTURE_HOME / "OligoWalk"))
RNASTRUCTURE_DATAPATH = Path(os.getenv("RNASTRUCTURE_DATAPATH", RNASTRUCTURE_HOME / "data_tables"))
''',
    "pubmed_esearch": '''"""
pubmed_esearch/config.py — PubMed E-utilities API configuration.
"""
import os
PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "user@example.com")
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY", "")
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
''',
    "pubmed_efetch": '''"""
pubmed_efetch/config.py — PubMed E-utilities API configuration.
"""
import os
PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "user@example.com")
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY", "")
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
''',
    "openalex_search": '''"""
openalex_search/config.py — OpenAlex API configuration.
"""
OPENALEX_BASE_URL = "https://api.openalex.org"
''',
    "search_knowledge": f'''"""
search_knowledge/config.py — RAG Chroma index path configuration.
"""
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(".env", usecwd=True), override=True)

PROJECT_ROOT = {PROJ_ROOT}
RAGBASE_ROOT = Path(os.getenv("RAGBASE_ROOT", PROJECT_ROOT / "ragbase"))
RAG_CHROMA_DIR = Path(os.getenv("RAG_CHROMA_DIR", RAGBASE_ROOT / "chroma_db"))
''',
    "list_ragbase": f'''"""
list_ragbase/config.py — RAG Chroma index path configuration.
"""
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(".env", usecwd=True), override=True)

PROJECT_ROOT = {PROJ_ROOT}
RAGBASE_ROOT = Path(os.getenv("RAGBASE_ROOT", PROJECT_ROOT / "ragbase"))
RAG_CHROMA_DIR = Path(os.getenv("RAG_CHROMA_DIR", RAGBASE_ROOT / "chroma_db"))
''',
    "list_database": f'''"""
list_database/config.py — All database paths for data discovery tool.
"""
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(".env", usecwd=True), override=True)

PROJECT_ROOT = {PROJ_ROOT}
DATABASE_ROOT = Path(os.getenv("DATABASE_ROOT", PROJECT_ROOT / "database"))
INSECT_BLAST_DB = Path(os.getenv("INSECT_BLAST_DB", DATABASE_ROOT / "blast" / "insect_blastndb"))
NTO_BLAST_DB = Path(os.getenv("NTO_BLAST_DB", DATABASE_ROOT / "blast" / "nto_blastndb"))
INSECT_GFF3_DB = Path(os.getenv("INSECT_GFF3_DB", DATABASE_ROOT / "annotation" / "insect_gff3"))
INSECT_CDS_DB = Path(os.getenv("INSECT_CDS_DB", DATABASE_ROOT / "refseq" / "insect_cds"))
NTOS_REFSEQ_DB = Path(os.getenv("NTOS_REFSEQ_DB", DATABASE_ROOT / "refseq" / "nto_refseq"))
KINSHIP_DB = Path(os.getenv("KINSHIP_DB", DATABASE_ROOT / "kinship"))
RNASTRUCTURE_HOME = Path(os.getenv("RNASTRUCTURE_HOME", DATABASE_ROOT / "RNAstructure"))
''',
    "fetch_nto_seq": f'''"""
fetch_nto_seq/config.py — NTO reference sequence path configuration.
"""
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(".env", usecwd=True), override=True)

PROJECT_ROOT = {PROJ_ROOT}
DATABASE_ROOT = Path(os.getenv("DATABASE_ROOT", PROJECT_ROOT / "database"))
NTOS_REFSEQ_DB = Path(os.getenv("NTOS_REFSEQ_DB", DATABASE_ROOT / "refseq" / "nto_refseq"))
''',
    "fetch_insect_cds": f'''"""
fetch_insect_cds/config.py — Insect CDS database path configuration.
"""
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(".env", usecwd=True), override=True)

PROJECT_ROOT = {PROJ_ROOT}
DATABASE_ROOT = Path(os.getenv("DATABASE_ROOT", PROJECT_ROOT / "database"))
INSECT_CDS_DB = Path(os.getenv("INSECT_CDS_DB", DATABASE_ROOT / "refseq" / "insect_cds"))
''',
}


# ============================================================
# Import rewrite
# ============================================================
def _rewrite_imports(code: str) -> str:
    """Replace 'from tool_config import X' with 'from .config import X'.

    Handles both single-line and multi-line parenthesized imports.
    """
    import re
    lines = code.splitlines()
    new_lines = []
    in_multiline = False
    multiline_names = []

    for line in lines:
        stripped = line.strip()

        # Detect start of multi-line import: from tool_config import (
        if stripped.startswith("from tool_config import (") or (
            "from tool_config import" in stripped and "(" in stripped
        ):
            in_multiline = True
            # Extract anything after the opening paren
            after_paren = stripped.split("(", 1)[1]
            if after_paren.strip() and after_paren.strip() != "":
                for name in after_paren.split(","):
                    name = name.strip().rstrip(")").strip()
                    if name:
                        multiline_names.append(name)
            continue

        # Inside multi-line import block
        if in_multiline:
            if ")" in stripped:
                # End of block
                for name in stripped.replace(")", "").split(","):
                    name = name.strip()
                    if name:
                        multiline_names.append(name)
                in_multiline = False
                new_lines.append(f"from .config import {', '.join(multiline_names)}")
                multiline_names = []
            else:
                for name in stripped.split(","):
                    name = name.strip()
                    if name:
                        multiline_names.append(name)
            continue

        # Single-line import
        if stripped.startswith("from tool_config import ") and "(" not in stripped:
            names = stripped[len("from tool_config import "):]
            new_lines.append(f"from .config import {names}")
        else:
            new_lines.append(line)

    return "\n".join(new_lines)


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("  工具迁移脚本")
    print("=" * 60)
    print()

    for tool in TOOL_DEFS:
        name = tool["name"]
        print(f"  [{name}] ...", end=" ")

        # 1. Create directory
        tool_dir = TOOLS_DIR / name
        tool_dir.mkdir(exist_ok=True)

        # 2. Read source
        bak_path = TOOLS_DIR / tool["bak_file"]
        if not bak_path.exists():
            print(f"⚠ source not found: {bak_path.name}")
            continue
        code = bak_path.read_text(encoding="utf-8")

        # 3. Handle fetch_seq splitting
        split = tool.get("split")
        if split:
            import re
            if split == "nto":
                # Keep FetchNtoSeqTool only — remove FetchInsectCDSTool block
                # Strategy: find the start of Tool 2 section and truncate
                idx = code.find("# ============================================================\n# Tool 2:")
                if idx == -1:
                    idx = code.find('class FetchInsectCDSTool')
                if idx > 0:
                    code = code[:idx]
                # Remove fetch_insect_cds_tool singleton line
                code = code.replace("fetch_insect_cds_tool = FetchInsectCDSTool()\n", "")
                code = code.replace("fetch_insect_cds_tool = FetchInsectCDSTool()", "")
            elif split == "cds":
                # Keep FetchInsectCDSTool + helpers — remove FetchNtoSeqTool block
                idx = code.find("# ============================================================\n# Tool 1:")
                if idx == -1:
                    idx = code.find("class FetchNtoSeqTool")
                if idx > 0:
                    # Keep only from Tool 2 onward
                    tool2_idx = code.find("# ============================================================\n# Tool 2:")
                    if tool2_idx == -1:
                        tool2_idx = code.find('class FetchInsectCDSTool')
                    if tool2_idx > 0:
                        # Keep header + helper functions + Tool 2
                        header_end = idx  # keep everything before Tool 1
                        code = code[:header_end] + code[tool2_idx:]
                code = code.replace("fetch_nto_seq_tool = FetchNtoSeqTool()\n", "")
                code = code.replace("fetch_nto_seq_tool = FetchNtoSeqTool()", "")

        # 4. Rewrite imports
        code = _rewrite_imports(code)

        # 5. Write tool.py
        (tool_dir / "tool.py").write_text(code, encoding="utf-8")

        # 6. Write config.py if needed
        config_source = tool.get("config_source")
        if config_source == "inline":
            config_content = CONFIG_TEMPLATES.get(name)
            if config_content:
                (tool_dir / "config.py").write_text(config_content, encoding="utf-8")

        # 7. Write metadata.md
        metadata = f"""---
name: {tool['name']}
tag: {tool['tag']}
description: "{tool['description']}"
---
"""
        (tool_dir / "metadata.md").write_text(metadata, encoding="utf-8")

        # 8. Write __init__.py
        instance_name = f"{name}_tool"
        (tool_dir / "__init__.py").write_text(
            f"from .tool import {instance_name}\n",
            encoding="utf-8",
        )

        print("✅")

    print()
    print("所有工具文件夹创建完成！")


if __name__ == "__main__":
    main()

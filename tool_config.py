"""
tool_config.py —— 生信工具数据路径集中配置

所有路径集中管理，支持环境变量覆盖。
用户只需在 .env 中设置对应变量即可自定义路径。

用法:
    from tool_config import INSECT_BLAST_DB, ...
    db_path = os.path.join(INSECT_BLAST_DB, species, species)
"""

import os
from pathlib import Path

# ============================================================
# 项目根目录
# ============================================================
PROJECT_ROOT = Path(__file__).parent.resolve()

# ============================================================
# 数据库总根（可在 .env 中用 DATABASE_ROOT 覆盖）
# ============================================================
DATABASE_ROOT = Path(os.getenv("DATABASE_ROOT", PROJECT_ROOT / "database"))

# ============================================================
# BLAST 数据库 — 基因组同源搜索
# ============================================================
# 昆虫（靶标）基因组 BLAST 库，含 .nsq .nin .nhr 等索引文件
# 内部结构: {species}/{species}.n*
INSECT_BLAST_DB = Path(os.getenv(
    "INSECT_BLAST_DB",
    DATABASE_ROOT / "blast" / "insect_blastndb"
))

# 非靶标生物（NTO）基因组 BLAST 库
# 内部结构: {species}/{species}.n*
NTO_BLAST_DB = Path(os.getenv(
    "NTO_BLAST_DB",
    DATABASE_ROOT / "blast" / "nto_blastndb"
))

# ============================================================
# 基因组注释 — GFF3 功能注释
# ============================================================
# 昆虫基因组 GFF3 注释文件
# 内部结构: {species}/{species}.gff3
INSECT_GFF3_DB = Path(os.getenv(
    "INSECT_GFF3_DB",
    DATABASE_ROOT / "annotation" / "insect_gff3"
))

# ============================================================
# 参考序列 — samtools 序列提取
# ============================================================
# 昆虫 CDS 序列文件
# 内部结构: {species}/*.cds.fa
INSECT_CDS_DB = Path(os.getenv(
    "INSECT_CDS_DB",
    DATABASE_ROOT / "refseq" / "insect_cds"
))

# 非靶标生物参考序列 FASTA 文件（含 .fai 索引）
# 内部结构: {species}/*.fa
NTOS_REFSEQ_DB = Path(os.getenv(
    "NTOS_REFSEQ_DB",
    DATABASE_ROOT / "refseq" / "nto_refseq"
))

# ============================================================
# 亲缘关系 — 物种演化树 + NCBI 分类
# ============================================================
# 昆虫物种树文件 + NCBI taxa 数据库
# 包含: insects_species.nwk, taxa.sqlite, taxa.sqlite.traverse.pkl
KINSHIP_DB = Path(os.getenv(
    "KINSHIP_DB",
    DATABASE_ROOT / "kinship"
))

# 物种树文件路径
INSECT_TREE_PATH = Path(os.getenv(
    "INSECT_TREE_PATH",
    KINSHIP_DB / "insects_species.nwk"
))

# NCBI ETE3 分类数据库
INSECT_TAXA_DB_PATH = Path(os.getenv(
    "INSECT_TAXA_DB_PATH",
    KINSHIP_DB / "taxa.sqlite"
))

# ============================================================
# RNAstructure — OligoWalk 热力学参数
# ============================================================
# RNAstructure 安装目录（只保留 OligoWalk + data_tables/）
RNASTRUCTURE_HOME = Path(os.getenv(
    "RNASTRUCTURE_HOME",
    DATABASE_ROOT / "RNAstructure"
))

# OligoWalk 二进制（预编译，无需 make）
OLIGOWALK_BIN = Path(os.getenv(
    "OLIGOWALK_BIN",
    RNASTRUCTURE_HOME / "OligoWalk"
))

# OligoWalk 需要的热力学参数表路径（环境变量名必须是 DATAPATH）
RNASTRUCTURE_DATAPATH = Path(os.getenv(
    "RNASTRUCTURE_DATAPATH",
    RNASTRUCTURE_HOME / "data_tables"
))

# ============================================================
# PubMed E-utilities — 文献检索 API 配置
# ============================================================
# NCBI 要求请求中必须包含 email 以标识请求者身份
# 可在 .env 中用 PUBMED_EMAIL 和 PUBMED_API_KEY 覆盖
PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "user@example.com")
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY", "")
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# ============================================================
# OpenAlex — 开放学术文献 API 配置
# ============================================================
OPENALEX_BASE_URL = "https://api.openalex.org"

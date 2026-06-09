"""
fetch_seq.py —— 序列提取工具集

提供两个工具:
  - fetch_seq:         用 samtools faidx 从 NTO 参考基因组提取指定区段序列
  - fetch_insect_cds:  从昆虫 CDS 数据库中按转录本 ID 搜索 CDS 序列

典型工作流:
  nto_blast → fetch_seq → clustal       (NTO 脱靶风险评估链路)
  insect_blast → insect_anno → fetch_insect_cds → primer3  (靶标设计链路)

从 old_tools/fetch_seq/executor.py 抽取核心逻辑。

调用链路：
  fetch_nto_seq:     LLM → FetchNtoSeqTool._run() → subprocess.run(["samtools", "faidx", ...]) → JSON
  fetch_insect_cds:  LLM → FetchInsectCDSTool._run() → file search → JSON
"""

import glob
import json
import logging
import os
import subprocess
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from tool_config import NTOS_REFSEQ_DB, INSECT_CDS_DB

logger = logging.getLogger("RPA_Tools.FetchSeq")

# ============================================================
# Pydantic 输入 Schema
# ============================================================

class FetchRegionHit(BaseModel):
    """要提取的目标区段。"""

    subject_id: str = Field(
        description="参考序列 ID（FASTA header 中 '>' 后面的标识符）"
    )
    s_start: int = Field(
        description="提取起始位置（1-based）"
    )
    s_end: int = Field(
        description="提取结束位置（1-based）。若 start > end 则自动取反向互补"
    )


class FetchSeqInput(BaseModel):
    """NTO 参考序列区段提取参数。"""

    species: str = Field(
        description="物种名称，如 Apis_mellifera"
    )
    hits: List[FetchRegionHit] = Field(
        description="要提取的区段列表，每项含 subject_id、s_start、s_end"
    )


class FetchInsectCDSInput(BaseModel):
    """昆虫 CDS 序列查询参数。"""

    species: str = Field(
        description="物种名称，如 Bombyx_mori"
    )
    transcript_id: str = Field(
        description="转录本/mRNA ID，如 'Bmor000255.1'（来自 insect_anno 注释结果）"
    )


# ============================================================
# 辅助函数（从原始 executor.py 移植）
# ============================================================

def _find_fasta(species: str, db_root: str, suffix: str = ".fa") -> Optional[str]:
    """在数据库目录中查找物种的第一个 FASTA 文件。"""
    species_dir = os.path.join(db_root, species)
    pattern = os.path.join(species_dir, f"*{suffix}")
    files = glob.glob(pattern, recursive=False)
    if not files:
        # 尝试递归搜索
        files = glob.glob(os.path.join(species_dir, "**", f"*{suffix}"), recursive=True)
    return files[0] if files else None


def _reverse_complement(seq: str) -> str:
    """计算反向互补序列。"""
    complement = str.maketrans("ATCGatcg", "TAGCtagc")
    return seq.translate(complement)[::-1]


def _fetch_region(fasta_path: str, seqid: str, start: int, end: int) -> str:
    """用 samtools faidx 提取指定区段。"""
    reverse = False
    if start > end:
        start, end = end, start
        reverse = True

    region = f"{seqid}:{start}-{end}"
    cmd = ["samtools", "faidx", fasta_path, region]
    logger.info("Fetching region: %s", region)

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    lines = result.stdout.strip().split("\n")
    if len(lines) < 2:
        raise RuntimeError(f"Invalid samtools output for {region}")

    seq = "".join(lines[1:])  # 跳过 FASTA header 行

    if reverse:
        seq = _reverse_complement(seq)
    return seq


def _fetch_cds(species: str, transcript_id: str) -> Optional[str]:
    """在昆虫 CDS FASTA 中搜索指定转录本的序列。"""
    species_dir = os.path.join(str(INSECT_CDS_DB), species)
    cds_files = glob.glob(os.path.join(species_dir, "**", "*.cds.fa"), recursive=True)

    if not cds_files:
        logger.error("No .cds.fa files found for species: %s", species)
        return None

    target_id = f">{transcript_id}"
    for cds_path in cds_files:
        try:
            with open(cds_path, "r") as f:
                found = False
                sequence_parts = []
                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        if found:
                            break
                        if line.startswith(target_id):
                            found = True
                    elif found:
                        sequence_parts.append(line)

                if found:
                    return "".join(sequence_parts)
        except Exception as e:
            logger.error("Error reading %s: %s", cds_path, e)
            continue

    return None


# ============================================================
# Tool 1: fetch_seq — NTO 参考序列区段提取
# ============================================================

class FetchNtoSeqTool(BaseTool):
    """从 NTO（非靶标生物）参考基因组中提取指定区段序列。

    使用 samtools faidx 按染色体/支架 ID 和坐标提取序列，
    支持自动反向互补（当 start > end 时视为反向链）。
    典型用法：接收 nto_blast 的输出，提取匹配区段序列用于 clustal 比对。
    """

    name: str = "fetch_nto_seq"
    description: str = (
        "从 NTO 参考基因组中提取指定区段序列。输入物种名和命中列表"
        "（每项含 subject_id、s_start、s_end），"
        "使用 samtools faidx 提取对应 DNA 序列。"
        "若 s_start > s_end 则自动返回反向互补链。"
        "典型用法：接在 nto_blast 之后，为 clustal 比对准备序列。"
    )
    args_schema: type = FetchSeqInput

    def _run(
        self,
        species: str,
        hits: List[dict],
    ) -> str:
        """提取 NTO 参考序列区段（同步）。"""

        try:
            # 查找 FASTA 文件
            fasta_path = _find_fasta(species, str(NTOS_REFSEQ_DB))
            if not fasta_path:
                return json.dumps(
                    {
                        "status": "error",
                        "error": f"No FASTA file found for species '{species}'",
                        "searched_in": str(NTOS_REFSEQ_DB),
                    },
                    ensure_ascii=False,
                    indent=2,
                )

            # 检查 .fai 索引
            fai_path = fasta_path + ".fai"
            if not os.path.exists(fai_path):
                logger.warning("No .fai index found, creating one...")
                subprocess.run(
                    ["samtools", "faidx", fasta_path],
                    capture_output=True,
                    text=True,
                    check=True,
                )

            results = []
            for hit in hits:  # hits 已被 Pydantic 解析为 FetchRegionHit 对象
                seq = _fetch_region(
                    fasta_path,
                    hit.subject_id,
                    hit.s_start,
                    hit.s_end,
                )
                results.append({
                    "subject_id": hit.subject_id,
                    "start": hit.s_start,
                    "end": hit.s_end,
                    "length": len(seq),
                    "sequence": seq,
                })

            return json.dumps(
                {
                    "status": "success",
                    "species": species,
                    "fasta_used": fasta_path,
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            )

        except subprocess.CalledProcessError as e:
            logger.exception("samtools faidx failed")
            return json.dumps(
                {
                    "status": "error",
                    "error": "samtools execution failed",
                    "stderr": e.stderr,
                    "stdout": e.stdout,
                },
                ensure_ascii=False,
                indent=2,
            )

        except FileNotFoundError:
            logger.exception("samtools not found")
            return json.dumps(
                {
                    "status": "error",
                    "error": "samtools binary not found",
                    "details": "Install samtools: sudo apt install -y samtools",
                },
                ensure_ascii=False,
                indent=2,
            )

        except Exception as e:
            logger.exception("Fetch sequence failed")
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
# Tool 2: fetch_insect_cds — 昆虫 CDS 序列查询
# ============================================================

class FetchInsectCDSTool(BaseTool):
    """按转录本 ID 从昆虫 CDS 数据库中搜索编码序列。

    不依赖外部二进制，纯 Python 文件搜索。
    典型用法：接收 insect_anno 的注释结果（转录本 ID），获取 CDS 序列。
    """

    name: str = "fetch_insect_cds"
    description: str = (
        "按转录本/mRNA ID 从昆虫 CDS 数据库中搜索编码序列（CDS）。"
        "输入物种名和转录本 ID（如来自 insect_anno 注释结果的 Bmor000255.1），"
        "返回对应的 CDS 序列。"
        "纯 Python 文件搜索，不依赖外部工具。"
    )
    args_schema: type = FetchInsectCDSInput

    def _run(
        self,
        species: str,
        transcript_id: str,
    ) -> str:
        """搜索昆虫 CDS 序列（同步）。"""

        try:
            sequence = _fetch_cds(species, transcript_id)

            if not sequence:
                return json.dumps(
                    {
                        "status": "error",
                        "error": "transcript not found",
                        "details": (
                            f"Transcript ID '{transcript_id}' not found "
                            f"in {species} CDS database"
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )

            return json.dumps(
                {
                    "status": "success",
                    "species": species,
                    "transcript_id": transcript_id,
                    "length": len(sequence),
                    "sequence": sequence,
                },
                ensure_ascii=False,
                indent=2,
            )

        except Exception as e:
            logger.exception("Fetch CDS failed")
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
# 单例导出
# ============================================================
fetch_nto_seq_tool = FetchNtoSeqTool()
fetch_insect_cds_tool = FetchInsectCDSTool()

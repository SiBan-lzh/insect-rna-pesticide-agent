"""
insect_anno.py —— BLAST命中基因组注释工具

subprocess 工具，调用 bedtools window 在 GFF3 注释文件中查找 BLAST 命中位点
附近的基因特征（gene/mRNA/exon/CDS/UTR）。

典型工作流: insect_blast → insect_anno
    BLAST 找到同源位点 → 注释位点的基因结构

从 old_tools/insectbase_anno/executor.py 抽取核心逻辑。

调用链路：
    LLM → InsectAnoTool._run() → subprocess.run(["bedtools", "window", ...]) → JSON
"""

import json
import logging
import os
import subprocess
import tempfile
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from tool_config import INSECT_GFF3_DB

logger = logging.getLogger("RPA_Tools.InsectAnno")

# ============================================================
# GFF3 特征类型说明（典型昆虫基因组注释）
# ============================================================
FEATURE_TYPES = {
    "gene": "基因",
    "mRNA": "转录本",
    "exon": "外显子",
    "CDS": "编码区",
    "five_prime_UTR": "5'UTR",
    "three_prime_UTR": "3'UTR",
}


# ============================================================
# Pydantic 输入 Schema
# ============================================================
class BlastHitInput(BaseModel):
    """单个 BLAST 命中的位点信息。"""

    chromosome: str = Field(
        description="染色体/支架 ID，如 BMSK_chr25"
    )
    start_position: int = Field(
        description="BLAST 匹配在染色体上的起始位置 (1-based)"
    )
    end_position: int = Field(
        description="BLAST 匹配在染色体上的结束位置 (1-based)"
    )
    name: str = Field(
        default="hit",
        description="命中标识，用于在结果中区分不同命中"
    )
    score: float = Field(
        default=0.0,
        description="BLAST bit score"
    )
    strand: str = Field(
        default="+",
        description="链方向 (+ 或 -)"
    )


class InsectAnnoInput(BaseModel):
    """基因组注释查询参数。通常接收 insect_blast 的输出作为输入。"""

    blast_hits: List[BlastHitInput] = Field(
        description="BLAST 搜索结果中的命中列表，每条含染色体、起止位点等信息"
    )
    species: str = Field(
        description="物种名称，如 Bombyx_mori"
    )
    window_size: int = Field(
        default=100,
        ge=0,
        description="注释窗口大小 (bp)，在命中位点两侧扩展搜索基因特征"
    )


# ============================================================
# 核心函数（从原始 executor.py 移植）
# ============================================================
def get_full_chromosome_id(gff3_path: str, simple_chrom_id: str) -> str:
    """在 GFF3 中查找染色体的完整 ID。

    用纯 Python 实现，替代原始 shell 管道 (grep -v '#' | grep | head | cut)。
    """
    try:
        with open(gff3_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                cols = line.split("\t", 1)
                if cols[0] == simple_chrom_id:
                    return cols[0]  # 第一列就是 seqid
        return simple_chrom_id
    except Exception as e:
        logger.warning("ID resolution failed for %s: %s", simple_chrom_id, e)
        return simple_chrom_id


def build_bed_content(hits: List[BlastHitInput]) -> str:
    """将 BLAST 命中列表转换为 BED 格式文本。

    BED 格式: chrom start(0-based) end name score strand
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
    """用 bedtools window 查找 BED 区域在 GFF3 中的重叠注释。"""
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
    """解析 GFF3 第9列属性字符串。

    例: "ID=Bmor000001.1;Parent=Bmor000001" → {"ID": "Bmor000001.1", "Parent": "Bmor000001"}
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
    """基因组注释工具。

    接收 BLAST 命中列表，在指定物种的 GFF3 注释文件中查找命中位点
    附近的基因特征。典型用法是将 insect_blast 的输出作为本工具的输入，
    从而确定靶标基因在基因组上的结构（外显子、内含子、CDS 等）。

    对应原始服务: old_tools/insectbase_anno/executor.py
    """

    name: str = "insect_anno"
    description: str = (
        "基因组注释工具。接收 BLAST 比对命中列表（含染色体、起止位点），"
        "在指定昆虫物种的 GFF3 注释文件中查找命中位点附近的基因特征，"
        "包括基因 (gene)、转录本 (mRNA)、外显子 (exon)、编码区 (CDS)、"
        "UTR 等结构信息。典型用法是与 insect_blast 组合使用："
        "先 BLAST 找同源位点，再用本工具注释位点的基因结构。"
    )
    args_schema: type = InsectAnnoInput

    def _run(
        self,
        blast_hits: List[dict],
        species: str,
        window_size: int = 100,
    ) -> str:
        """根据 BLAST 命中注释基因组特征（同步）。"""

        try:
            # blast_hits 已被 Pydantic 自动解析为 BlastHitInput 对象列表
            hits = blast_hits

            # 定位 GFF3 文件
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

            # 解析染色体完整 ID
            for hit in hits:
                hit.chromosome = get_full_chromosome_id(gff3_path, hit.chromosome)

            # 构建 BED → 运行 bedtools window
            bed_content = build_bed_content(hits)
            raw_output = run_bedtools_window(bed_content, gff3_path, window_size)

            # 解析 bedtools 输出
            # 格式: BED(6列) + GFF3(9列) = 15列
            results: Dict[str, dict] = {}
            if raw_output.strip():
                for line in raw_output.strip().split("\n"):
                    cols = line.split("\t")
                    if len(cols) < 15:
                        continue

                    query_name = cols[3]  # BED 第4列 = 命中名称

                    if query_name not in results:
                        results[query_name] = {
                            "blast_hit": query_name,
                            "features": [],
                        }

                    feature = {
                        "type": cols[8],       # GFF3 第3列 = feature type
                        "start": cols[9],      # GFF3 第4列 = start
                        "end": cols[10],       # GFF3 第5列 = end
                        "strand": cols[12],    # GFF3 第7列 = strand
                        "attributes": parse_attributes(cols[14]),  # GFF3 第9列
                    }
                    results[query_name]["features"].append(feature)

            # 统计各类型数量
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
                    "details": "请安装 bedtools: sudo apt install bedtools",
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
# 单例导出
# ============================================================
insect_anno_tool = InsectAnoTool()

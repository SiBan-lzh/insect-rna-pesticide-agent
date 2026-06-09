"""
clustal.py —— 双序列比对与连续匹配分析工具

subprocess 工具，调用 Clustal Omega (clustalo) 进行双序列比对，
检测 siRNA 与 NTO（非靶标生物）基因间的连续匹配区域，评估脱靶风险。

典型工作流: insect_blast → fetch_seq → clustal
    找到同源位点 → 提取对应序列 → 比对分析连续匹配

从 old_tools/clustal/executor.py 抽取核心逻辑。

调用链路：
    LLM → ClustalTool._run() → subprocess.run(["clustalo", ...]) → analyze_continuous_match() → JSON
"""

import json
import logging
import os
import subprocess
import tempfile
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import BaseTool

logger = logging.getLogger("RPA_Tools.Clustal")

# ============================================================
# Pydantic 输入 Schema
# ============================================================
class SequenceItem(BaseModel):
    """单条序列（名称 + 序列）。"""

    name: str = Field(
        description="序列标识符，如 'siRNA' 或基因 ID"
    )
    sequence: str = Field(
        description="序列 (5'→3')"
    )


class ClustalInput(BaseModel):
    """Clustal Omega 双序列比对参数。"""

    sequences: List[SequenceItem] = Field(
        min_length=2, max_length=2,
        description="要比对的两条序列（恰好 2 条）"
    )
    window_size: int = Field(
        default=21, ge=18, le=25,
        description="连续匹配窗口阈值 (nt)。≥该值的连续匹配区域视为脱靶风险"
    )

    @field_validator("sequences")
    @classmethod
    def check_exactly_two(cls, v):
        if len(v) != 2:
            raise ValueError(
                "Exactly two sequences required for pairwise off-target analysis"
            )
        return v


# ============================================================
# 核心函数（从原始 executor.py 移植）
# ============================================================
def run_clustal(sequences: List[SequenceItem]) -> str:
    """运行 Clustal Omega 双序列比对，返回 FASTA 格式的比对结果。"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".fasta", delete=False
    ) as input_file:
        for seq in sequences:
            input_file.write(f">{seq.name}\n{seq.sequence}\n")
        input_file.flush()
        input_path = input_file.name

    output_file = tempfile.NamedTemporaryFile(
        mode="r", suffix=".fa", delete=False
    )
    output_path = output_file.name
    output_file.close()

    cmd = [
        "clustalo",
        "-i", input_path,
        "-o", output_path,
        "--outfmt", "fa",
        "--threads", "1",
        "--force",
        "--dealign",
    ]

    logger.info("Running: %s", " ".join(cmd))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        with open(output_path, "r") as f:
            alignment = f.read()

        if not alignment.strip():
            raise RuntimeError("clustalo returned empty alignment")

        return alignment

    finally:
        for p in [input_path, output_path]:
            if os.path.exists(p):
                os.unlink(p)


def parse_alignment(alignment: str) -> List[str]:
    """从 multi-FASTA 比对结果中提取序列。

    返回两个序列字符串（已去除 '>' 头部行，合并多行）。
    """
    lines = alignment.strip().splitlines()
    seqs = []
    current_seq = []

    for line in lines:
        if line.startswith(">"):
            if current_seq:
                seqs.append("".join(current_seq))
                current_seq = []
        else:
            current_seq.append(line.strip())

    if current_seq:
        seqs.append("".join(current_seq))

    if len(seqs) != 2:
        raise ValueError(
            f"Alignment parsing error: expected 2 sequences, got {len(seqs)}"
        )

    return seqs


def analyze_continuous_match(alignment: str, window_size: int) -> dict:
    """在双序列比对中检测连续匹配区域。

    逐列扫描比对结果，记录所有 ≥ window_size 的连续匹配段。
    """
    seq1, seq2 = parse_alignment(alignment)

    max_run = 0
    current_run = 0
    start_pos = None
    match_regions = []

    for i, (a, b) in enumerate(zip(seq1, seq2)):
        if a == b and a != "-":
            if current_run == 0:
                start_pos = i
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            if current_run >= window_size:
                match_regions.append({
                    "start_alignment_pos": start_pos,
                    "length": current_run,
                })
            current_run = 0
            start_pos = None

    # 处理到末尾还在匹配中的情况
    if current_run >= window_size:
        match_regions.append({
            "start_alignment_pos": start_pos,
            "length": current_run,
        })

    return {
        "max_continuous_match": max_run,
        "threshold": window_size,
        "off_target_risk": max_run >= window_size,
        "matching_regions": match_regions,
    }


# ============================================================
# LangChain Tool
# ============================================================
class ClustalTool(BaseTool):
    """双序列比对与脱靶风险评估工具。

    使用 Clustal Omega 对 siRNA 序列与 NTO（非靶标生物）基因序列
    进行双序列比对，检测是否存在超过窗口阈值的连续匹配区域。
    连续匹配 ≥ window_size（默认 21nt）即判定存在脱靶风险。

    对应原始服务: old_tools/clustal/executor.py
    """

    name: str = "clustal"
    description: str = (
        "双序列比对工具。使用 Clustal Omega 对两条序列（如 siRNA 和 NTO 基因）"
        "进行比对，检测连续匹配区域。若存在 ≥ window_size（默认 21nt）的"
        "连续匹配片段，则判定为存在脱靶风险。适用于 siRNA 安全性评估。"
    )
    args_schema: type = ClustalInput

    def _run(
        self,
        sequences: List[dict],
        window_size: int = 21,
    ) -> str:
        """运行双序列比对并分析连续匹配（同步）。"""

        try:
            # Pydantic 自动验证恰好 2 条序列
            seq_items = sequences  # 已是 SequenceItem 对象列表

            # 运行 clustalo
            alignment = run_clustal(seq_items)

            # 分析连续匹配
            analysis = analyze_continuous_match(alignment, window_size)

            return json.dumps(
                {
                    "status": "success",
                    "window_size": window_size,
                    "analysis": analysis,
                    "alignment": alignment,
                },
                ensure_ascii=False,
                indent=2,
            )

        except ValueError as e:
            logger.warning("Clustal validation error: %s", e)
            return json.dumps(
                {
                    "status": "error",
                    "error": "validation error",
                    "details": str(e),
                },
                ensure_ascii=False,
                indent=2,
            )

        except subprocess.CalledProcessError as e:
            logger.exception("Clustal Omega execution failed")
            return json.dumps(
                {
                    "status": "error",
                    "error": "clustalo execution failed",
                    "stderr": e.stderr,
                    "stdout": e.stdout,
                },
                ensure_ascii=False,
                indent=2,
            )

        except FileNotFoundError:
            logger.exception("clustalo not found")
            return json.dumps(
                {
                    "status": "error",
                    "error": "clustalo binary not found",
                    "details": "Install clustalo: sudo apt install -y clustalo",
                },
                ensure_ascii=False,
                indent=2,
            )

        except Exception as e:
            logger.exception("Clustal failed")
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
clustal_tool = ClustalTool()

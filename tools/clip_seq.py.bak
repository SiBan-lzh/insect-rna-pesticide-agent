"""
clip_seq.py —— 序列片段截取工具

纯 Python 工具，从 CDS/基因序列中按坐标截取目标片段，用于 dsRNA 靶标区域
的精确提取。支持正链和反链互补。

在 Target Design 流水线中的位置：
    insect_blast → insect_anno → fetch_insect_cds → clip_seq
    → (产出靶标片段，供下游 dsRNA_designer 使用)

调用链路：
    LLM → ClipSeqTool._run() → 纯 Python 字符串操作 → JSON
"""

import json
import logging
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

logger = logging.getLogger("RPA_Tools.ClipSeq")

# ============================================================
# 碱基互补映射
# ============================================================
_COMPLEMENT = str.maketrans("ATGCatgc", "TACGtacg")


# ============================================================
# Pydantic 输入 Schema
# ============================================================
class ClipSeqInput(BaseModel):
    """序列截取参数。"""

    sequence: str = Field(
        description="输入 DNA 序列 (5'→3')，通常为 CDS 全长"
    )
    start: int = Field(
        default=1, ge=1,
        description="截取起始位置（1-based），默认为 1"
    )
    length: int = Field(
        default=300, ge=50, le=1000,
        description="截取片段长度（bp），默认 300。dsRNA 靶标通常为 200-500 bp"
    )
    as_reverse_complement: bool = Field(
        default=False,
        description="是否返回反向互补链（用于反义链设计）"
    )


# ============================================================
# 核心函数
# ============================================================
def compute_gc_content(seq: str) -> float:
    """计算序列 GC 含量（%）。"""
    if not seq:
        return 0.0
    gc = sum(1 for base in seq.upper() if base in ("G", "C"))
    return round(gc / len(seq) * 100, 1)


def reverse_complement(seq: str) -> str:
    """获取 DNA 序列的反向互补链 (5'→3')。"""
    return seq.translate(_COMPLEMENT)[::-1]


def clip_sequence(
    sequence: str,
    start: int = 1,
    length: int = 300,
    as_reverse_complement: bool = False,
) -> dict:
    """从 DNA 序列中截取指定位置的片段。

    Args:
        sequence: 输入序列 (5'→3')
        start: 起始位置（1-based）
        length: 截取长度（bp）
        as_reverse_complement: 是否返回反向互补链

    Returns:
        包含截取片段及元信息的字典
    """
    seq = sequence.strip().upper()
    # 移除可能存在的空白字符
    seq = "".join(seq.split())
    seq_len = len(seq)

    # 坐标转换: 1-based → 0-based
    zero_start = start - 1
    zero_end = zero_start + length

    if zero_start < 0 or zero_start >= seq_len:
        return {
            "status": "error",
            "error": f"Start position {start} is out of range (seq length: {seq_len})",
        }

    # 如果终止位置超出序列，自动截断
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
    """序列片段截取工具。

    从 CDS 或基因全长序列中按坐标精确截取目标区域，
    用于 RNAi 靶标设计中的 dsRNA 靶标片段制备。

    典型 dsRNA 靶标要求：
    - 长度 200-500 bp（默认 300 bp）
    - 位于外显子区域
    - GC 含量适中（40-60%）

    支持返回正义链（默认）或反义互补链。

    流水线位置: fetch_insect_cds → clip_seq → primer3/oligowalk
    """

    name: str = "clip_seq"
    description: str = (
        "序列片段截取工具。从 DNA 序列（通常是 CDS）中按指定起始位置和长度"
        "截取目标片段，用于后续 dsRNA 设计。"
        "支持返回正向链（sense）或反向互补链（antisense），"
        "自动计算 GC 含量。默认知名截取 300 bp 的正义链片段。"
    )
    args_schema: type = ClipSeqInput

    def _run(
        self,
        sequence: str,
        start: int = 1,
        length: int = 300,
        as_reverse_complement: bool = False,
    ) -> str:
        """截取序列片段（同步）。"""

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
# 单例导出
# ============================================================
clip_seq_tool = ClipSeqTool()

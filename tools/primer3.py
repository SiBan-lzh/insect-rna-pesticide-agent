"""
primer3.py —— PCR 引物设计工具

纯 Python 工具，调用 primer3-py 库，无 subprocess/外部二进制依赖。
从 old_tools/primer3/executor.py 抽取核心逻辑。

调用链路：
    LLM → Primer3Tool._run() → primer3.bindings.design_primers() → JSON
"""

import json
import logging
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
import primer3

logger = logging.getLogger("RPA_Tools.Primer3")

# ============================================================
# T7 启动子前缀（体外转录用）
# ============================================================
T7_PREFIX = "TAATACGACTCACTATAGGG"


# ============================================================
# Pydantic 输入 Schema
# ============================================================
class Primer3Input(BaseModel):
    """PCR 引物设计参数。"""

    sequence: str = Field(
        description="模板 DNA 序列 (5'→3')"
    )
    sequence_id: str = Field(
        default="query",
        description="序列标识符"
    )
    num_return: int = Field(
        default=3, ge=1, le=5,
        description="返回的最优引物对数"
    )

    # 靶标区域（可选，用于指定扩增区间）
    target_start: Optional[int] = Field(
        default=None,
        description="靶标区域在序列上的起始位置（1-based）"
    )
    target_len: Optional[int] = Field(
        default=None,
        description="靶标区域长度（bp）"
    )

    # 引物长度
    primer_opt_size: int = Field(default=20, ge=10, le=30)
    primer_min_size: int = Field(default=18, ge=10, le=30)
    primer_max_size: int = Field(default=25, ge=10, le=35)

    # GC 含量
    primer_opt_gc_percent: float = Field(default=50.0, ge=20.0, le=80.0)
    primer_min_gc: float = Field(default=40.0, ge=20.0, le=80.0)
    primer_max_gc: float = Field(default=60.0, ge=20.0, le=80.0)

    # 退火温度
    primer_opt_tm: float = Field(default=60.0, ge=50.0, le=70.0)
    primer_min_tm: float = Field(default=57.0, ge=45.0, le=70.0)
    primer_max_tm: float = Field(default=63.0, ge=50.0, le=75.0)

    # 产物大小范围
    primer_product_size_range: List[int] = Field(
        default_factory=lambda: [100, 300],
        description="期望的 PCR 产物大小范围 [min, max]"
    )


# ============================================================
# 核心函数（从原始 executor.py 移植，保留不变）
# ============================================================
def build_primer3_args(inp: Primer3Input):
    """将 Pydantic 输入转换为 primer3-py 库所需的参数格式。"""
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
    """调用 primer3-py 库设计引物，返回结构化结果。"""
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
    """设计 PCR 引物，用于扩增靶标基因片段。

    输入模板 DNA 序列和引物设计参数（长度、GC 含量、Tm 等），
    返回多对候选引物序列及其热力学属性，
    同时附带 T7 启动子序列版本（用于体外转录合成 dsRNA）。

    对应原始服务: old_tools/primer3/executor.py
    """

    name: str = "primer3"
    description: str = (
        "设计 PCR 引物。输入模板 DNA 序列 (5'→3')，"
        "返回多对候选引物序列（含正向/反向、Tm 值、产物大小、罚分），"
        "以及添加了 T7 启动子前缀的版本（用于体外转录合成 dsRNA）。"
        "可调节引物长度、GC 含量、退火温度范围、产物大小等参数。"
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
        """设计 PCR 引物（同步）。"""

        if primer_product_size_range is None:
            primer_product_size_range = [100, 300]

        try:
            # 构建参数 → 调用 primer3-py → 格式化结果
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
# 单例导出
# ============================================================
primer3_tool = Primer3Tool()

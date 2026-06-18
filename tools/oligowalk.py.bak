"""
oligowalk.py —— siRNA 热力学打分工具

subprocess 工具，调用 OligoWalk 二进制，计算 siRNA 与靶标 mRNA 的结合自由能。
从 old_tools/oligowalk/executor.py 抽取核心逻辑。

调用链路：
    LLM → OligoWalkTool._run() → subprocess.run(["OligoWalk", ...]) → parse_report() → JSON
"""

import json
import logging
import os
import re
import subprocess
import tempfile
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from tool_config import OLIGOWALK_BIN, RNASTRUCTURE_DATAPATH

logger = logging.getLogger("RPA_Tools.OligoWalk")


# ============================================================
# Pydantic 输入 Schema
# ============================================================
class OligoWalkInput(BaseModel):
    """OligoWalk siRNA 热力学参数。"""

    sequence: str = Field(
        description="靶标 mRNA 序列 (5'→3')"
    )
    run_type: str = Field(
        default="fast",
        description="运行模式: 'fast' (Mode=3, Suboptimal=1, 快速) / 'research' (Mode=2, Suboptimal=2, 研究级精度)"
    )
    oligo_length: int = Field(
        default=21, ge=18, le=23,
        description="siRNA/寡核苷酸长度 (nt)"
    )
    mode: Optional[int] = Field(
        default=None, ge=1, le=3,
        description="折叠模式 (仅 research 模式生效; 1=单链, 2=双链, 3=快速)"
    )
    suboptimal: Optional[int] = Field(
        default=None, ge=0, le=4,
        description="次优结构数量 (仅 research 模式生效; 0=仅最优, 1-4=次优结构数)"
    )
    filter: Optional[int] = Field(
        default=None, ge=0, le=1,
        description="GU 配对过滤 (0=不过滤, 1=过滤GU; fast默认0, research默认1)"
    )
    top_n: int = Field(
        default=10, ge=1,
        description="返回的最优候选 siRNA 数量"
    )
    dna: bool = Field(
        default=False,
        description="使用 DNA 热力学参数 (默认 False，使用 RNA 参数)"
    )


# ============================================================
# 命令构建（从原始 executor.py 移植）
# ============================================================
def build_command(
    req: OligoWalkInput,
    seq_file: str,
    report_file: str,
) -> List[str]:
    """根据输入参数构建 OligoWalk 命令行。"""
    cmd = [str(OLIGOWALK_BIN), seq_file, report_file]

    # 寡核苷酸长度
    cmd.extend(["-l", str(req.oligo_length)])

    if req.run_type == "fast":
        # 快速模式: Mode=3, Suboptimal=1, 不过滤GU
        cmd.extend(["-m", "3"])
        cmd.extend(["-s", "1"])
        cmd.extend(["-fi", str(req.filter if req.filter is not None else 0)])
        cmd.append("-score")
    elif req.run_type == "research":
        # 研究模式: Mode=2, Suboptimal=2, 过滤GU
        cmd.extend(["-m", str(req.mode or 2)])
        cmd.extend(["-s", str(req.suboptimal or 2)])
        cmd.extend(["-fi", str(req.filter if req.filter is not None else 1)])
        cmd.append("-score")
    else:
        raise ValueError(f"Invalid run_type: '{req.run_type}', must be 'fast' or 'research'")

    if req.dna:
        cmd.append("-d")

    return cmd


# ============================================================
# 报告解析（从原始 executor.py 移植）
# ============================================================
def parse_report(report_path: str) -> dict:
    """解析 OligoWalk 输出的文本报告，提取参数和能量表。"""
    parameters: dict = {}
    energy_table: list = []
    parsing_energy: bool = False
    header_fields: list = []

    with open(report_path, "r") as f:
        for line in f:
            line = line.strip()

            # 解析总长度
            if "Total size of the target" in line:
                m = re.search(r"(\d+)", line)
                if m:
                    parameters["total_size"] = int(m.group(1))

            # 解析扫描区间
            elif line.startswith("Scanned position on target:"):
                m = re.search(
                    r"Scanned position on target:\s*(\d+)\s*to\s*(\d+)", line
                )
                if m:
                    parameters["scan_start"] = int(m.group(1))
                    parameters["scan_end"] = int(m.group(2))

            elif line.startswith("Oligonucleotides:"):
                continue

            # 解析寡核苷酸长度
            elif line.startswith("Length:"):
                m = re.search(r"Length:\s*(\d+)", line)
                if m:
                    parameters["oligo_length"] = int(m.group(1))

            # 解析方法选项 (Mode / Folding region size / Suboptimal)
            elif (
                line.startswith("Mode:")
                or line.startswith("Folding region size:")
                or line.startswith("Suboptimal:")
            ):
                if "method_options" not in parameters:
                    parameters["method_options"] = {}
                key, val = line.split(":", 1)
                parameters["method_options"][key.strip()] = val.strip()

            # 检测能量表开始
            if line.startswith("Energy table:"):
                parsing_energy = True
                continue

            if parsing_energy:
                # 解析表头: "Pos." "Oligo" ...
                if line.startswith("Pos.") and "Oligo" in line:
                    header_fields = [h.strip() for h in re.split(r"\t+", line)]
                    continue
                # 空行或非数据行跳过
                if not line or not re.match(r"^\d+", line):
                    continue

                # 解析数据行（Tab 分隔或双空格分隔）
                values = re.split(r"\t+", line)
                if len(values) != len(header_fields):
                    values = re.split(r"\s{2,}", line)
                if len(values) == len(header_fields):
                    row_dict = {
                        header_fields[i]: values[i].strip()
                        for i in range(len(header_fields))
                    }
                    energy_table.append(row_dict)

    return {
        "parameters": parameters,
        "energy_table": energy_table,
    }


# ============================================================
# LangChain Tool
# ============================================================
class OligoWalkTool(BaseTool):
    """siRNA 热力学打分工具。

    给定靶标 mRNA 序列，使用 OligoWalk 滑动窗口算法扫描所有可能的
    siRNA 结合位点，计算每个位点的热力学结合自由能，返回排序后的
    最优候选 siRNA 列表。

    支持快速模式（fast，适合大规模筛选）和研究级模式（research，更高精度）。

    对应原始服务: old_tools/oligowalk/executor.py
    """

    name: str = "oligowalk"
    description: str = (
        "siRNA 热力学打分工具。给定靶标 mRNA 序列，使用 OligoWalk 算法"
        "扫描所有可能的 siRNA 结合位点，计算每个候选 siRNA 的"
        "热力学结合自由能（Overall ΔG），返回排序后的最优候选列表。"
        "支持 fast（快速筛选）和 research（高精度）两种模式。"
    )
    args_schema: type = OligoWalkInput

    def _run(
        self,
        sequence: str,
        run_type: str = "fast",
        oligo_length: int = 21,
        mode: Optional[int] = None,
        suboptimal: Optional[int] = None,
        filter: Optional[int] = None,
        top_n: int = 10,
        dna: bool = False,
    ) -> str:
        """运行 OligoWalk 热力学打分（同步）。"""

        try:
            # 构建输入对象
            inp = OligoWalkInput(
                sequence=sequence,
                run_type=run_type,
                oligo_length=oligo_length,
                mode=mode,
                suboptimal=suboptimal,
                filter=filter,
                top_n=top_n,
                dna=dna,
            )

            # 确保 DATAPATH 环境变量已设置
            if "DATAPATH" not in os.environ:
                data_tables = RNASTRUCTURE_DATAPATH
                if data_tables.exists():
                    os.environ["DATAPATH"] = str(data_tables)
                else:
                    logger.warning(
                        "DATAPATH not set and data_tables not found at %s. "
                        "OligoWalk may fail to find thermodynamic parameters.",
                        data_tables,
                    )

            # 准备临时文件
            with tempfile.TemporaryDirectory() as tmpdir:
                seq_file = os.path.join(tmpdir, "input.fa")
                report_file = os.path.join(tmpdir, "output.txt")

                # 写入 FASTA 文件
                with open(seq_file, "w") as f:
                    f.write(f">query\n{inp.sequence}\n")

                # 构建命令
                cmd = build_command(inp, seq_file, report_file)
                logger.info("Executing: %s", " ".join(cmd))

                # 执行 OligoWalk
                timeout = 120 if run_type == "research" else 40
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )

                logger.info("STDOUT:\n%s", result.stdout)
                if result.stderr:
                    logger.info("STDERR:\n%s", result.stderr)

                # 解析报告
                data = parse_report(report_file)

                # 检查执行状态
                if result.returncode != 0:
                    logger.error("OligoWalk execution failed (rc=%d)", result.returncode)
                    return json.dumps(
                        {
                            "status": "error",
                            "error": "oligowalk execution failed",
                            "parameters": data.get("parameters", {}),
                            "stderr": result.stderr,
                            "stdout": result.stdout,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )

                # 按 Overall 结合自由能升序排列（越负越稳定）
                sorted_table = sorted(
                    data["energy_table"],
                    key=lambda x: float(x.get("Overall (kcal/mol)", 0)),
                )

                top_candidates = sorted_table[:top_n]

                return json.dumps(
                    {
                        "status": "success",
                        "mode": run_type,
                        "count": len(sorted_table),
                        "top_candidates": top_candidates,
                        "parameters": data["parameters"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )

        except subprocess.TimeoutExpired:
            logger.error("OligoWalk execution timeout")
            return json.dumps(
                {
                    "status": "error",
                    "error": "execution timeout",
                    "details": "OligoWalk runtime exceeded allowed time",
                },
                ensure_ascii=False,
                indent=2,
            )

        except FileNotFoundError:
            logger.exception("OligoWalk binary not found")
            return json.dumps(
                {
                    "status": "error",
                    "error": "OligoWalk binary not found",
                    "details": (
                        "OligoWalk is not installed. "
                        "Please compile RNAstructure from database/RNAstructure/ "
                        "and ensure OligoWalk is in your PATH."
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )

        except Exception as e:
            logger.exception("OligoWalk failed")
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
oligowalk_tool = OligoWalkTool()

"""
insect_blast.py —— 靶标昆虫基因组 BLAST 同源搜索工具

从 tools/insectbase_blast/executor.py 抽取核心逻辑，
去掉 FastAPI 层，改为 LangChain BaseTool。

调用链路：
    LLM → InsectBlastTool._run() → subprocess blastn → 解析 tabular → JSON
"""

import os
import json
import subprocess
import tempfile
import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

# 项目路径配置
from tool_config import INSECT_BLAST_DB

logger = logging.getLogger("RPA_Tools.InsectBlast")


# ============================================================
# Pydantic 输入 Schema（LLM 用它来理解参数结构）
# ============================================================
class InsectBlastInput(BaseModel):
    """昆虫基因组 BLAST 搜索参数。"""

    sequence: str = Field(
        description="查询序列 (5'→3')，如 ATGCGTACGT..."
    )
    species: str = Field(
        description="靶标昆虫物种名（拉丁名，下划线分隔），如 Bombyx_mori"
    )
    word_size: int = Field(
        default=7,
        description="BLAST word size，短序列(<50bp)建议 7-9"
    )
    evalue: float = Field(
        default=10.0,
        description="期望值阈值，越大越宽松"
    )
    gapopen: int = Field(default=10, description="空位罚分（开 gap）")
    gapextend: int = Field(default=4, description="空位罚分（延伸）")
    penalty: int = Field(default=-1, description="碱基错配罚分")
    reward: int = Field(default=1, description="碱基匹配得分")


# ============================================================
# 核心解析函数（从原始 executor.py 移植）
# ============================================================
def parse_blast_tabular(raw_text: str) -> List[dict]:
    """解析 blastn -outfmt 6 的 tabular 输出。

    标准 12 列格式：
    qseqid sseqid pident length mismatch gapopen
    qstart qend sstart send evalue bitscore
    """
    results = []
    for i, line in enumerate(raw_text.strip().splitlines()):
        if not line:
            continue
        cols = line.split("\t")
        if len(cols) < 12:
            continue
        results.append({
            "hit_number": i + 1,
            "subject_id": cols[1],
            "identity_pct": float(cols[2]),
            "aln_length": int(cols[3]),
            "mismatches": int(cols[4]),
            "gap_opens": int(cols[5]),
            "q_start": int(cols[6]),
            "q_end": int(cols[7]),
            "s_start": int(cols[8]),
            "s_end": int(cols[9]),
            "evalue": float(cols[10]),
            "bit_score": float(cols[11]),
        })
    return results


# ============================================================
# LangChain Tool
# ============================================================
class InsectBlastTool(BaseTool):
    """在靶标昆虫基因组中运行 BLASTN 同源搜索。

    给定一条查询序列和物种名，返回基因组匹配结果
    （scaffold 位置、相似度、E-value 等）。

    对应原始服务: tools/insectbase_blast/executor.py
    """

    name: str = "insect_blast"
    description: str = (
        "在靶标昆虫基因组数据库中运行 BLASTN 同源搜索。"
        "输入查询 DNA 序列和昆虫物种名（拉丁名，下划线分隔），"
        "返回基因组匹配位置、序列相似度和 E-value。"
        "用于：验证靶标基因在目标昆虫基因组中是否存在同源序列。"
    )
    args_schema: type = InsectBlastInput

    # ---- 运行时配置 ----
    timeout: int = 120  # BLAST 子进程超时秒数

    def _run(
        self,
        sequence: str,
        species: str,
        word_size: int = 7,
        evalue: float = 10.0,
        gapopen: int = 10,
        gapextend: int = 4,
        penalty: int = -1,
        reward: int = 1,
    ) -> str:
        """执行 BLAST 搜索（同步）。"""

        # 1. 定位数据库文件
        db_path = os.path.join(INSECT_BLAST_DB, species, species)
        if not os.path.exists(f"{db_path}.nsq"):
            return json.dumps({
                "status": "error",
                "error": f"BLAST database not found for species '{species}'",
                "expected_path": f"{db_path}.nsq",
            }, ensure_ascii=False, indent=2)

        # 2. 写临时 FASTA 文件
        tmp_in_path = None
        tmp_out_path = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".fasta", delete=False
            ) as tmp_in:
                tmp_in.write(f">query\n{sequence}\n")
                tmp_in_path = tmp_in.name

            tmp_out_path = tmp_in_path + ".out"

            # 3. 执行 blastn
            cmd = [
                "blastn",
                "-query", tmp_in_path,
                "-db", db_path,
                "-out", tmp_out_path,
                "-outfmt", "6",
                "-task", "blastn-short",
                "-word_size", str(word_size),
                "-evalue", str(evalue),
                "-gapopen", str(gapopen),
                "-gapextend", str(gapextend),
                "-penalty", str(penalty),
                "-reward", str(reward),
            ]

            logger.info("Executing BLAST: %s", " ".join(cmd))
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True,
                timeout=self.timeout,
            )

            # 4. 读取并解析结果
            with open(tmp_out_path, "r") as f:
                raw_results = f.read()

            hits = parse_blast_tabular(raw_results)

            return json.dumps({
                "status": "success",
                "species": species,
                "query_length": len(sequence),
                "hits_count": len(hits),
                "params": {
                    "word_size": word_size,
                    "evalue": evalue,
                    "task": "blastn-short",
                },
                "results": hits,
            }, ensure_ascii=False, indent=2)

        except subprocess.TimeoutExpired:
            return json.dumps({
                "status": "error",
                "error": f"BLAST search timed out after {self.timeout}s",
            }, ensure_ascii=False, indent=2)

        except subprocess.CalledProcessError as e:
            logger.error("BLAST Error: %s", e.stderr)
            return json.dumps({
                "status": "error",
                "error": "BLAST execution failed",
                "details": e.stderr,
            }, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.exception("Unexpected error in insect_blast")
            return json.dumps({
                "status": "error",
                "error": "Internal error",
                "details": str(e),
            }, ensure_ascii=False, indent=2)

        finally:
            # 5. 清理临时文件
            for p in [tmp_in_path, tmp_out_path]:
                if p and os.path.exists(p):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass


# ============================================================
# 单例导出
# ============================================================
# LangChain 的 BaseTool 需要实例化（不是类）
insect_blast_tool = InsectBlastTool()

"""
nto_blast.py —— 非靶标生物（NTO）基因组 BLAST 脱靶风险检查工具

与 insect_blast 几乎同构，唯一差异：
- 数据库路径指向非靶标生物基因组
- 工具命名与描述侧重「脱靶风险」语义

调用链路：
    LLM → NTOBlastTool._run() → subprocess blastn → 解析 tabular → JSON
"""

import os
import json
import subprocess
import tempfile
import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from tool_config import NTO_BLAST_DB

logger = logging.getLogger("RPA_Tools.NTOBlast")


# ============================================================
# Pydantic 输入 Schema
# ============================================================
class NTOBlastInput(BaseModel):
    """非靶标生物 BLAST 搜索参数。"""

    sequence: str = Field(
        description="待检测的 dsRNA/siRNA 序列 (5'→3')"
    )
    species: str = Field(
        description="非靶标物种名（拉丁名，下划线分隔），如 Apis_mellifera"
    )
    word_size: int = Field(
        default=7, description="BLAST word size，短序列(<50bp)建议 7-9"
    )
    evalue: float = Field(
        default=10.0, description="期望值阈值"
    )
    gapopen: int = Field(default=10, description="空位罚分（开 gap）")
    gapextend: int = Field(default=4, description="空位罚分（延伸）")
    penalty: int = Field(default=-1, description="碱基错配罚分")
    reward: int = Field(default=1, description="碱基匹配得分")


# ============================================================
# 核心解析函数（与 insect_blast 共用逻辑）
# ============================================================
def parse_blast_tabular(raw_text: str) -> List[dict]:
    """解析 blastn -outfmt 6 tabular 输出（标准 12 列）"""
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
class NTOBlastTool(BaseTool):
    """在非靶标生物基因组中运行 BLASTN，检查脱靶风险。

    给定一条 dsRNA/siRNA 序列和非靶标物种名，
    返回基因组匹配结果，判断是否存在脱靶风险。

    对应原始服务: tools/ntos_blast/executor.py
    """

    name: str = "nto_blast"
    description: str = (
        "在非靶标生物（NTO）基因组数据库中运行 BLASTN 搜索，"
        "检查 dsRNA/siRNA 是否存在脱靶风险。"
        "输入查询序列和非靶标物种名（拉丁名，下划线分隔），"
        "返回基因组匹配位置、序列相似度和 E-value。"
        "高相似度 + 低 E-value = 存在脱靶风险。"
    )
    args_schema: type = NTOBlastInput

    timeout: int = 120

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
        """执行 NTO BLAST 脱靶检查（同步）。"""

        # 1. 定位数据库
        db_path = os.path.join(NTO_BLAST_DB, species, species)
        if not os.path.exists(f"{db_path}.nsq"):
            return json.dumps({
                "status": "error",
                "error": f"NTO BLAST database not found for species '{species}'",
                "expected_path": f"{db_path}.nsq",
            }, ensure_ascii=False, indent=2)

        tmp_in_path = None
        tmp_out_path = None

        try:
            # 2. 写临时 FASTA
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

            logger.info("Executing NTO BLAST: %s", " ".join(cmd))
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True,
                timeout=self.timeout,
            )

            # 4. 解析结果
            with open(tmp_out_path, "r") as f:
                raw_results = f.read()

            hits = parse_blast_tabular(raw_results)

            # 5. 脱靶风险判定
            risky_hits = [h for h in hits if h["identity_pct"] >= 80]
            off_target_risk = "high" if risky_hits else (
                "medium" if any(h["identity_pct"] >= 60 for h in hits)
                else "low"
            )

            return json.dumps({
                "status": "success",
                "species": species,
                "query_length": len(sequence),
                "hits_count": len(hits),
                "risky_hits_count": len(risky_hits),
                "off_target_risk": off_target_risk,
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
            logger.error("NTO BLAST Error: %s", e.stderr)
            return json.dumps({
                "status": "error",
                "error": "BLAST execution failed",
                "details": e.stderr,
            }, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.exception("Unexpected error in nto_blast")
            return json.dumps({
                "status": "error",
                "error": "Internal error",
                "details": str(e),
            }, ensure_ascii=False, indent=2)

        finally:
            for p in [tmp_in_path, tmp_out_path]:
                if p and os.path.exists(p):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass


# ============================================================
# 单例导出
# ============================================================
nto_blast_tool = NTOBlastTool()

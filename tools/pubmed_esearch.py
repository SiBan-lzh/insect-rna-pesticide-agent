"""
pubmed_esearch.py —— PubMed 文献检索工具 (ESearch)

通过 NCBI E-utilities API 搜索 PubMed 数据库，返回匹配的 PMID 列表。
支持年份过滤和 API Key 认证。

对应 Dify 工作流: old_tools/文献检索/pubmed_Esearch.yml

调用链路：
    LLM → PubmedEsearchTool._run() → requests.get(esearch.fcgi) → JSON
"""

import json
import logging
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
import requests

from tool_config import PUBMED_BASE_URL, PUBMED_EMAIL, PUBMED_API_KEY

logger = logging.getLogger("RPA_Tools.PubmedEsearch")


# ============================================================
# Pydantic 输入 Schema
# ============================================================
class PubmedEsearchInput(BaseModel):
    """PubMed ESearch 检索参数。"""

    query: str = Field(
        description="PubMed 搜索关键词，如 'RNAi pesticide Bombyx mori'"
    )
    max_results: int = Field(
        default=10, ge=1, le=100,
        description="返回的最大 PMID 数量"
    )
    min_year: Optional[str] = Field(
        default=None,
        description="发表年份下限，如 '2018'"
    )
    max_year: Optional[str] = Field(
        default=None,
        description="发表年份上限，如 '2024'"
    )
    email: str = Field(
        default=PUBMED_EMAIL,
        description="NCBI 要求的用户邮箱，用于标识请求者"
    )
    api_key: str = Field(
        default=PUBMED_API_KEY,
        description="NCBI API Key，用于提高访问频率限制（可选）"
    )


# ============================================================
# 核心函数（从 Dify 工作流移植，保留原有逻辑）
# ============================================================
def build_pubmed_query(query: str, min_year: Optional[str], max_year: Optional[str]) -> str:
    """构建 PubMed 查询语法，支持年份范围过滤。

    PubMed 年份过滤使用 [pdat] 字段 + 冒号范围语法。
    例如: (RNAi) AND ("2018"[pdat] : "2024"[pdat])
    """
    query_parts = [query]

    year_filter_parts = []
    if min_year and min_year.strip():
        year_filter_parts.append(f'"{min_year.strip()}"[pdat]')
    if max_year and max_year.strip():
        year_filter_parts.append(f'"{max_year.strip()}"[pdat]')

    # 组装年份区间: "YYYY"[pdat] : "YYYY"[pdat]
    if len(year_filter_parts) == 1:
        # 单边过滤：直接作为 AND 条件
        query_parts.append(year_filter_parts[0])
    elif len(year_filter_parts) == 2:
        year_query = " : ".join(year_filter_parts)
        query_parts.append(year_query)

    # 用 AND 连接并用括号包裹确保优先级
    return " AND ".join([f"({part})" for part in query_parts])


def search_pubmed(
    query: str,
    max_results: int = 10,
    min_year: Optional[str] = None,
    max_year: Optional[str] = None,
    email: str = "",
    api_key: str = "",
) -> dict:
    """调用 PubMed ESearch API 执行文献检索。"""
    if not email or not email.strip():
        return {"success": False, "message": "Email is required for PubMed API.", "pmids": []}

    search_term = build_pubmed_query(query, min_year, max_year)

    params = {
        "db": "pubmed",
        "term": search_term,
        "retmax": max_results,
        "retmode": "json",
        "email": email.strip(),
        "usehistory": "n",
    }
    if api_key and api_key.strip():
        params["api_key"] = api_key.strip()

    try:
        resp = requests.get(f"{PUBMED_BASE_URL}/esearch.fcgi", params=params, timeout=15)
    except Exception as e:
        return {"success": False, "message": f"Network error: {e}", "pmids": []}

    if resp.status_code != 200:
        return {"success": False, "message": f"HTTP {resp.status_code}", "pmids": []}

    try:
        data = resp.json()
        id_list = data.get("esearchresult", {}).get("idlist", [])
        total_count = data.get("esearchresult", {}).get("count", "0")
    except Exception as e:
        return {"success": False, "message": f"Failed to parse response: {e}", "pmids": []}

    return {
        "success": True,
        "message": f"Found {len(id_list)} PMID(s) out of {total_count} total results.",
        "total_count": total_count,
        "pmids": id_list,
    }


# ============================================================
# LangChain Tool
# ============================================================
class PubmedEsearchTool(BaseTool):
    """PubMed 文献检索工具 (ESearch)。

    通过 NCBI E-utilities API 搜索 PubMed 数据库，
    根据关键词和年份范围查找相关文献，返回 PMID 列表。
    返回的 PMID 可直接传给 pubmed_Efetch 工具获取文献详情。

    典型用途：
    1. 搜索特定昆虫的 RNAi 相关研究文献
    2. 按年份过滤，获取最新的研究进展
    3. 与 kinship 配合：先找到近缘物种，再用物种名+RNAi 搜索文献

    对应 Dify 工作流: old_tools/文献检索/pubmed_Esearch.yml
    """

    name: str = "pubmed_Esearch"
    description: str = (
        "PubMed 文献检索工具。输入搜索关键词（如 'RNAi Bombyx mori'），"
        "返回匹配的 PMID 列表。支持年份范围过滤。"
        "PMID 可传给 pubmed_Efetch 工具获取文献标题、摘要等详细信息。"
    )
    args_schema: type = PubmedEsearchInput

    def _run(
        self,
        query: str,
        max_results: int = 10,
        min_year: Optional[str] = None,
        max_year: Optional[str] = None,
        email: str = "",
        api_key: str = "",
    ) -> str:
        """执行 PubMed 文献检索（同步）。"""

        if not query or not query.strip():
            return json.dumps({
                "status": "error",
                "error": "Query word is empty.",
                "pmids": [],
            }, ensure_ascii=False, indent=2)

        try:
            result = search_pubmed(query, max_results, min_year, max_year, email, api_key)
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.exception("PubMed ESearch failed")
            return json.dumps({
                "success": False,
                "message": f"Internal error: {e}",
                "pmids": [],
            }, ensure_ascii=False, indent=2)


# ============================================================
# 单例导出
# ============================================================
pubmed_esearch_tool = PubmedEsearchTool()

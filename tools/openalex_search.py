"""
openalex_search.py —— OpenAlex 开放学术文献检索工具

通过 OpenAlex REST API 搜索学术文献，支持关键词、领域过滤和年份范围。
OpenAlex 是免费开放的学术知识图谱，无需 API Key。

功能亮点:
- 自动将领域名称解析为 OpenAlex concept ID（如 "insect" → 昆虫学概念）
- 支持年份范围过滤
- 重建 inverted-index 摘要为可读文本

对应 Dify 工作流: old_tools/文献检索/openalex.yml

调用链路：
    LLM → OpenAlexSearchTool._run() → requests.get(works) → JSON
"""

import json
import logging
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
import requests

from tool_config import OPENALEX_BASE_URL

logger = logging.getLogger("RPA_Tools.OpenAlexSearch")


# ============================================================
# Pydantic 输入 Schema
# ============================================================
class OpenAlexSearchInput(BaseModel):
    """OpenAlex 文献检索参数。"""

    keyword: str = Field(
        description="搜索关键词，如 'RNA interference Bombyx mori'"
    )
    max_results: int = Field(
        default=10, ge=1, le=50,
        description="返回的最大文献数量"
    )
    field: str = Field(
        default="",
        description="研究领域过滤，如 'insect'、'agriculture'、'molecular biology'。留空则不按领域过滤"
    )
    min_year: Optional[str] = Field(
        default=None,
        description="发表年份下限，如 '2018'"
    )
    max_year: Optional[str] = Field(
        default=None,
        description="发表年份上限，如 '2024'"
    )


# ============================================================
# 核心函数（从 Dify 工作流移植，保留原有逻辑）
# ============================================================
def resolve_concept(field: str) -> dict:
    """将领域名称解析为 OpenAlex concept ID。

    返回: {"id": "C...", "name": "..."} 或空 dict
    """
    if not field or not field.strip():
        return {}

    try:
        resp = requests.get(
            f"{OPENALEX_BASE_URL}/concepts",
            params={"search": field.strip(), "per-page": 1},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            hits = data.get("results", [])
            if hits:
                top = hits[0]
                return {
                    "id": top.get("id"),
                    "name": top.get("display_name"),
                }
    except Exception:
        pass

    return {}


def rebuild_abstract(abstract_inverted_index: dict) -> str:
    """从 OpenAlex 的 inverted-index 格式重建摘要文本。

    OpenAlex 将摘要存储为 {word: [positions]} 的倒排索引，
    需要按位置排序后拼接为完整文本。
    """
    if not abstract_inverted_index:
        return ""

    expanded = []
    for word, positions in abstract_inverted_index.items():
        for pos in positions:
            expanded.append((pos, word))

    return " ".join(w for _, w in sorted(expanded))


def search_openalex(
    keyword: str,
    max_results: int = 10,
    field: str = "",
    min_year: Optional[str] = None,
    max_year: Optional[str] = None,
) -> dict:
    """调用 OpenAlex Works API 执行学术文献检索。"""
    filters = []
    applied_filters = {
        "min_year": min_year or "",
        "max_year": max_year or "",
        "field": field,
        "concept_resolved": False,
        "concept_id": None,
        "concept_name": None,
    }

    # ---- 年份过滤 ----
    try:
        if min_year and min_year.strip():
            y1 = int(min_year.strip())
            filters.append(f"from_publication_date:{y1}-01-01")
        if max_year and max_year.strip():
            y2 = int(max_year.strip())
            filters.append(f"to_publication_date:{y2}-12-31")
    except ValueError:
        pass

    # ---- 领域 → Concept ID 解析 ----
    if field and field.strip():
        concept = resolve_concept(field.strip())
        if concept.get("id"):
            filters.append(f"concepts.id:{concept['id']}")
            applied_filters["concept_resolved"] = True
            applied_filters["concept_id"] = concept["id"]
            applied_filters["concept_name"] = concept.get("name")

    # ---- API 请求 ----
    params = {
        "search": keyword.strip(),
        "per-page": max_results,
    }
    if filters:
        params["filter"] = ",".join(filters)

    try:
        resp = requests.get(f"{OPENALEX_BASE_URL}/works", params=params, timeout=20)
    except Exception as e:
        return {
            "success": False,
            "articles": [],
            "message": f"Network error: {e}",
            "status": "error",
            "applied_filters": applied_filters,
        }

    if resp.status_code != 200:
        return {
            "success": False,
            "articles": [],
            "message": f"HTTP {resp.status_code}: {resp.text[:500]}",
            "status": "error",
            "applied_filters": applied_filters,
        }

    data = resp.json()
    results = data.get("results", [])

    if not results:
        return {
            "success": True,
            "articles": [],
            "message": "No results with given filters.",
            "status": "completed",
            "applied_filters": applied_filters,
        }

    # ---- 解析文章 ----
    articles = []
    for item in results:
        # 从 inverted-index 重建摘要
        abstract_inv = item.get("abstract_inverted_index")
        if abstract_inv:
            abstract = rebuild_abstract(abstract_inv)
        else:
            abstract = item.get("abstract") or "No abstract"

        # 提取概念标签
        concepts = [
            c.get("display_name", "")
            for c in item.get("concepts", [])
            if c.get("display_name")
        ][:6]

        articles.append({
            "title": item.get("display_name", "No title"),
            "abstract": abstract,
            "openalex_id": item.get("id", "").replace("https://openalex.org/", ""),
            "doi": item.get("doi"),
            "publication_year": item.get("publication_year"),
            "cited_by_count": item.get("cited_by_count", 0),
            "concepts": concepts,
        })

    return {
        "success": True,
        "message": f"Found {len(articles)} article(s).",
        "status": "completed",
        "applied_filters": applied_filters,
        "articles": articles,
    }


# ============================================================
# LangChain Tool
# ============================================================
class OpenAlexSearchTool(BaseTool):
    """OpenAlex 开放学术文献检索工具。

    通过 OpenAlex REST API 免费检索学术文献，无需 API Key。
    支持关键词搜索、领域过滤（自动解析为 concept ID）和年份范围。
    返回文献标题、重建摘要、DOI、引用次数和概念标签。

    典型用途：
    1. 补充 PubMed 未覆盖的开放获取文献
    2. 按研究领域（如 insect biology）过滤文献
    3. 搜索跨学科的 RNAi 研究

    对应 Dify 工作流: old_tools/文献检索/openalex.yml
    """

    name: str = "openalex_search"
    description: str = (
        "OpenAlex 开放学术文献检索工具。输入搜索关键词，"
        "返回匹配的学术文献列表，包含标题、摘要、DOI 和引用次数。"
        "支持按研究领域过滤（如 'insect'、'molecular biology'）和年份范围。"
        "免费开放 API，无需认证。可与 PubMed 检索互补使用。"
    )
    args_schema: type = OpenAlexSearchInput

    def _run(
        self,
        keyword: str,
        max_results: int = 10,
        field: str = "",
        min_year: Optional[str] = None,
        max_year: Optional[str] = None,
    ) -> str:
        """执行 OpenAlex 文献检索（同步）。"""

        if not keyword or not keyword.strip():
            return json.dumps({
                "success": False,
                "articles": [],
                "message": "Keyword is empty.",
                "status": "error",
            }, ensure_ascii=False, indent=2)

        try:
            result = search_openalex(keyword, max_results, field, min_year, max_year)
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.exception("OpenAlex search failed")
            return json.dumps({
                "success": False,
                "articles": [],
                "message": f"Internal error: {e}",
                "status": "error",
            }, ensure_ascii=False, indent=2)


# ============================================================
# 单例导出
# ============================================================
openalex_search_tool = OpenAlexSearchTool()

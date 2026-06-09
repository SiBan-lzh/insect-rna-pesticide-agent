"""
pubmed_efetch.py —— PubMed 文献详情获取工具 (EFetch)

通过 NCBI E-utilities API，根据 PMID 批量获取文献的标题、摘要、
DOI、出版年份、MeSH 关键词等详细信息。

对应 Dify 工作流: old_tools/文献检索/pubmed_Efetch.yml

调用链路：
    LLM → PubmedEfetchTool._run() → requests.get(efetch.fcgi) → XML → JSON
"""

import json
import logging
import re
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
import requests

from tool_config import PUBMED_BASE_URL, PUBMED_EMAIL, PUBMED_API_KEY

logger = logging.getLogger("RPA_Tools.PubmedEfetch")


# ============================================================
# Pydantic 输入 Schema
# ============================================================
class PubmedEfetchInput(BaseModel):
    """PubMed EFetch 获取参数。"""

    pmids: str = Field(
        description="一个或多个 PMID，用逗号分隔，如 '12345678,23456789'"
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
def parse_pubmed_xml(content: str, requested_pmids: list) -> list:
    """解析 PubMed EFetch 返回的 XML，提取文章信息。

    使用正则表达式从 PubmedArticle XML 块中提取：
    - PMID, 标题, 摘要, DOI, 出版年份, MeSH 术语
    """
    articles = []

    # 按 <PubmedArticle>...</PubmedArticle> 切分
    article_blocks = re.findall(
        r'<PubmedArticle>.*?</PubmedArticle>', content, re.DOTALL
    )

    if not article_blocks:
        return articles

    for block in article_blocks:
        # PMID
        pmid_match = re.search(r'<PMID[^>]*>(\d+)</PMID>', block)
        if not pmid_match:
            continue
        current_pmid = pmid_match.group(1)

        # 确保这个 PMID 在请求列表中
        if current_pmid not in requested_pmids:
            continue

        # 标题
        title_match = re.search(
            r'<ArticleTitle[^>]*>(.*?)</ArticleTitle>', block, re.DOTALL
        )
        title = title_match.group(1).strip() if title_match else "No title"

        # 摘要（支持多个 <AbstractText> 段落）
        abstract_parts = re.findall(
            r'<AbstractText[^>]*>(.*?)</AbstractText>', block, re.DOTALL
        )
        if abstract_parts:
            abstract = " ".join([part.strip() for part in abstract_parts])
        else:
            abstract = "No abstract"

        # DOI
        doi = ""
        doi_match = re.search(r'<ArticleId IdType="doi">(.*?)</ArticleId>', block)
        if doi_match:
            doi = doi_match.group(1)

        # 出版年份
        pub_year = "Unknown"
        year_match = re.search(
            r'<PubDate>.*?<Year>(\d{4})</Year>.*?</PubDate>', block, re.DOTALL
        )
        if year_match:
            pub_year = year_match.group(1)
        else:
            medline_match = re.search(
                r'<MedlineDate>.*?(\d{4}).*?</MedlineDate>', block
            )
            if medline_match:
                pub_year = medline_match.group(1)

        # MeSH 关键词 / DescriptorName
        concepts = []
        mesh_terms = re.findall(
            r'<DescriptorName[^>]*>(.*?)</DescriptorName>', block
        )
        concepts.extend([term.strip() for term in mesh_terms[:6]])

        # 如果没有 MeSH，尝试提取 Keyword
        if not concepts:
            keywords = re.findall(
                r'<Keyword[^>]*>(.*?)</Keyword>', block
            )
            concepts.extend([kw.strip() for kw in keywords[:6]])

        articles.append({
            "pubmed_id": current_pmid,
            "title": title,
            "abstract": abstract,
            "doi": doi,
            "publication_year": pub_year,
            "concepts": concepts[:6],
        })

    return articles


def fetch_articles(pmids: str, email: str = "", api_key: str = "") -> dict:
    """调用 PubMed EFetch API 获取文献详细信息。"""
    if not email or not email.strip():
        return {
            "success": False,
            "message": "Email is required for PubMed API.",
            "articles": [],
        }

    params = {
        "db": "pubmed",
        "id": pmids.strip(),
        "retmode": "xml",
        "rettype": "abstract",
        "email": email.strip(),
    }
    if api_key and api_key.strip():
        params["api_key"] = api_key.strip()

    try:
        resp = requests.get(f"{PUBMED_BASE_URL}/efetch.fcgi", params=params, timeout=15)
    except Exception as e:
        return {"success": False, "message": f"Network error: {e}", "articles": []}

    if resp.status_code != 200:
        return {"success": False, "message": f"HTTP {resp.status_code}", "articles": []}

    requested_pmids = [p.strip() for p in pmids.split(",")]

    try:
        content = resp.content.decode("utf-8")
        articles = parse_pubmed_xml(content, requested_pmids)
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to parse XML response: {e}",
            "articles": [],
        }

    # 标记未找到的 PMID
    found_pmids = {article["pubmed_id"] for article in articles}
    for requested_id in requested_pmids:
        if requested_id not in found_pmids:
            articles.append({
                "pubmed_id": requested_id,
                "title": "Record not found in response",
                "abstract": "",
                "doi": "",
                "publication_year": "Unknown",
                "concepts": [],
            })

    return {
        "success": True,
        "message": f"Retrieved details for {len(articles)} article(s).",
        "articles": articles,
    }


# ============================================================
# LangChain Tool
# ============================================================
class PubmedEfetchTool(BaseTool):
    """PubMed 文献详情获取工具 (EFetch)。

    通过 NCBI E-utilities API，根据 PMID 获取文献的完整元数据，
    包括标题、摘要、DOI、出版年份和 MeSH 关键词。

    典型用途：
    1. 与 pubmed_Esearch 配合：先搜索 → 获取 PMID 列表 → 用此工具获取详情
    2. 根据已知 PMID 直接获取文献信息

    对应 Dify 工作流: old_tools/文献检索/pubmed_Efetch.yml
    """

    name: str = "pubmed_Efetch"
    description: str = (
        "PubMed 文献详情获取工具。输入一个或多个 PMID（逗号分隔），"
        "返回每篇文献的标题、摘要、DOI、出版年份和 MeSH 关键词。"
        "通常与 pubmed_Esearch 配合使用：先搜索获取 PMID，再批量获取详情。"
    )
    args_schema: type = PubmedEfetchInput

    def _run(
        self,
        pmids: str,
        email: str = "",
        api_key: str = "",
    ) -> str:
        """获取 PubMed 文献详情（同步）。"""

        if not pmids or not pmids.strip():
            return json.dumps({
                "success": False,
                "message": "PMID is empty.",
                "articles": [],
            }, ensure_ascii=False, indent=2)

        try:
            result = fetch_articles(pmids, email, api_key)
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.exception("PubMed EFetch failed")
            return json.dumps({
                "success": False,
                "message": f"Internal error: {e}",
                "articles": [],
            }, ensure_ascii=False, indent=2)


# ============================================================
# 单例导出
# ============================================================
pubmed_efetch_tool = PubmedEfetchTool()

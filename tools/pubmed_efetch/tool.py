"""
pubmed_efetch.py -- PubMed article detail fetcher (EFetch)

Fetches titles, abstracts, DOIs, publication years, and MeSH terms
for a batch of PMIDs via the NCBI E-utilities API.
"""

import json
import logging
import re
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
import requests

from .config import PUBMED_BASE_URL, PUBMED_EMAIL, PUBMED_API_KEY

logger = logging.getLogger("RPA_Tools.PubmedEfetch")


# ============================================================
# Pydantic input schema
# ============================================================
class PubmedEfetchInput(BaseModel):
    """Parameters for PubMed EFetch."""

    pmids: str = Field(
        description="One or more PMIDs separated by commas, e.g. '12345678,23456789'"
    )
    email: str = Field(
        default=PUBMED_EMAIL,
        description="Email required by NCBI to identify the requester"
    )
    api_key: str = Field(
        default=PUBMED_API_KEY,
        description="NCBI API key to increase rate limit (optional)"
    )


# ============================================================
# Core function
# ============================================================
def parse_pubmed_xml(content: str, requested_pmids: list) -> list:
    """Parse PubMed EFetch XML into article records.

    Extracts PMID, title, abstract, DOI, publication year, and MeSH terms
    from PubmedArticle XML blocks using regex.
    """
    articles = []

    # Split on <PubmedArticle>...</PubmedArticle>
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

        # Ensure this PMID was in the request list
        if current_pmid not in requested_pmids:
            continue

        # Title
        title_match = re.search(
            r'<ArticleTitle[^>]*>(.*?)</ArticleTitle>', block, re.DOTALL
        )
        title = title_match.group(1).strip() if title_match else "No title"

        # Abstract (supports multiple <AbstractText> paragraphs)
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

        # Publication year
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

        # MeSH terms / DescriptorName
        concepts = []
        mesh_terms = re.findall(
            r'<DescriptorName[^>]*>(.*?)</DescriptorName>', block
        )
        concepts.extend([term.strip() for term in mesh_terms[:6]])

        # Fallback to Keyword if no MeSH terms
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
    """Call PubMed EFetch API to retrieve article details."""
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

    # Mark PMIDs not found in the response
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
    """PubMed article detail fetcher (EFetch).

    Retrieves full metadata for a batch of PMIDs via the NCBI E-utilities API,
    including title, abstract, DOI, publication year, and MeSH keywords.
    """

    name: str = "pubmed_Efetch"
    description: str = (
        "PubMed article detail fetcher. Input one or more PMIDs (comma-separated), "
        "returns title, abstract, DOI, publication year, and MeSH keywords for each. "
        "Typically used with pubmed_Esearch: search first to get PMIDs, then fetch details."
    )
    args_schema: type = PubmedEfetchInput

    # ============================================================
    # Format helper: standardise to {status, articles, ...}
    # ============================================================
    @staticmethod
    def _format_result(result: dict) -> str:
        """Wrap fetch_articles() output into standard status/error format."""
        if result.get("success"):
            return json.dumps({
                "status": "success",
                "articles": result.get("articles", []),
                "message": result.get("message", ""),
            }, ensure_ascii=False, indent=2)
        else:
            return json.dumps({
                "status": "error",
                "error": result.get("message", "PubMed fetch failed"),
                "articles": [],
            }, ensure_ascii=False, indent=2)

    def _run(
        self,
        pmids: str,
        email: str = "",
        api_key: str = "",
    ) -> str:
        """Fetch PubMed article details (synchronous)."""

        if not pmids or not pmids.strip():
            return json.dumps({
                "status": "error",
                "error": "PMID is empty.",
                "articles": [],
            }, ensure_ascii=False, indent=2)

        try:
            result = fetch_articles(pmids, email, api_key)
            return self._format_result(result)
        except Exception as e:
            logger.exception("PubMed EFetch failed")
            return json.dumps({
                "status": "error",
                "error": f"Internal error: {e}",
                "articles": [],
            }, ensure_ascii=False, indent=2)


# ============================================================
# Singleton export
# ============================================================
pubmed_efetch_tool = PubmedEfetchTool()

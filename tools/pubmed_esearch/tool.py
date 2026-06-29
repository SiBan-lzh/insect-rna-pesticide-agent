"""
pubmed_esearch.py -- PubMed literature search tool (ESearch)

Searches PubMed via NCBI E-utilities API and returns matching PMIDs.
Supports year-range filtering and API Key authentication.
"""

import json
import logging
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
import requests

from .config import PUBMED_BASE_URL, PUBMED_EMAIL, PUBMED_API_KEY

logger = logging.getLogger("RPA_Tools.PubmedEsearch")


# ============================================================
# Pydantic input schema
# ============================================================
class PubmedEsearchInput(BaseModel):
    """PubMed ESearch query parameters."""

    query: str = Field(
        description="PubMed search keywords, e.g. 'RNAi pesticide Bombyx mori'"
    )
    max_results: int = Field(
        default=10, ge=1, le=100,
        description="Maximum number of PMIDs to return"
    )
    min_year: Optional[str] = Field(
        default=None,
        description="Earliest publication year, e.g. '2018'"
    )
    max_year: Optional[str] = Field(
        default=None,
        description="Latest publication year, e.g. '2024'"
    )
    email: str = Field(
        default=PUBMED_EMAIL,
        description="User email required by NCBI for request identification"
    )
    api_key: str = Field(
        default=PUBMED_API_KEY,
        description="NCBI API Key to increase rate limits (optional)"
    )


# ============================================================
# Core function
# ============================================================
def build_pubmed_query(query: str, min_year: Optional[str], max_year: Optional[str]) -> str:
    """Build a PubMed query string with optional year-range filtering.

    PubMed uses [pdat] field with colon-range syntax.
    Example: (RNAi) AND ("2018"[pdat] : "2024"[pdat])
    """
    query_parts = [query]

    year_filter_parts = []
    if min_year and min_year.strip():
        year_filter_parts.append(f'"{min_year.strip()}"[pdat]')
    if max_year and max_year.strip():
        year_filter_parts.append(f'"{max_year.strip()}"[pdat]')

    if len(year_filter_parts) == 1:
        query_parts.append(year_filter_parts[0])
    elif len(year_filter_parts) == 2:
        year_query = " : ".join(year_filter_parts)
        query_parts.append(year_query)

    return " AND ".join([f"({part})" for part in query_parts])


def search_pubmed(
    query: str,
    max_results: int = 10,
    min_year: Optional[str] = None,
    max_year: Optional[str] = None,
    email: str = "",
    api_key: str = "",
) -> dict:
    """Call the PubMed ESearch API to perform a literature search."""
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
    """PubMed literature search tool (ESearch).

    Searches PubMed via NCBI E-utilities API by keywords and year range,
    returns a list of PMIDs that can be passed to pubmed_Efetch for details.
    """

    name: str = "pubmed_Esearch"
    description: str = (
        "PubMed literature search tool. Input search keywords "
        "(e.g. 'RNAi Bombyx mori'), returns matching PMID list. "
        "Supports year-range filtering. "
        "PMIDs can be passed to pubmed_Efetch for titles, abstracts, etc."
    )
    args_schema: type = PubmedEsearchInput

    # ============================================================
    # Format helper: standardise to {status, pmids, ...}
    # ============================================================
    @staticmethod
    def _format_result(result: dict) -> str:
        """Wrap search_pubmed() output into standard status/error format."""
        if result.get("success"):
            return json.dumps({
                "status": "success",
                "pmids": result.get("pmids", []),
                "total_count": result.get("total_count", "0"),
                "message": result.get("message", ""),
            }, ensure_ascii=False, indent=2)
        else:
            return json.dumps({
                "status": "error",
                "error": result.get("message", "PubMed search failed"),
                "pmids": [],
            }, ensure_ascii=False, indent=2)

    def _run(
        self,
        query: str,
        max_results: int = 10,
        min_year: Optional[str] = None,
        max_year: Optional[str] = None,
        email: str = "",
        api_key: str = "",
    ) -> str:
        """Execute PubMed literature search (synchronous)."""

        if not query or not query.strip():
            return json.dumps({
                "status": "error",
                "error": "Query word is empty.",
                "pmids": [],
            }, ensure_ascii=False, indent=2)

        try:
            result = search_pubmed(query, max_results, min_year, max_year, email, api_key)
            return self._format_result(result)
        except Exception as e:
            logger.exception("PubMed ESearch failed")
            return json.dumps({
                "status": "error",
                "error": f"Internal error: {e}",
                "pmids": [],
            }, ensure_ascii=False, indent=2)


# ============================================================
# Singleton export
# ============================================================
pubmed_esearch_tool = PubmedEsearchTool()

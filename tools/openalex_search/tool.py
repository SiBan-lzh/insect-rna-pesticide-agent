"""
openalex_search.py -- OpenAlex open academic literature search tool

Search academic literature via the OpenAlex REST API with keyword,
field filtering, and year range. Free and open, no API key required.

Features:
- Auto-resolve field names to OpenAlex concept IDs (e.g. "insect" -> entomology concept)
- Year range filtering
- Rebuild inverted-index abstracts into readable text
"""

import json
import logging
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
import requests

from .config import OPENALEX_BASE_URL

logger = logging.getLogger("RPA_Tools.OpenAlexSearch")


# ============================================================
# Pydantic input schema
# ============================================================
class OpenAlexSearchInput(BaseModel):
    """Parameters for OpenAlex literature search."""

    keyword: str = Field(
        description="Search keyword, e.g. 'RNA interference Bombyx mori'"
    )
    max_results: int = Field(
        default=10, ge=1, le=50,
        description="Maximum number of results to return"
    )
    field: str = Field(
        default="",
        description="Research field filter, e.g. 'insect', 'agriculture', 'molecular biology'. Leave empty for no filter."
    )
    min_year: Optional[str] = Field(
        default=None,
        description="Earliest publication year, e.g. '2018'"
    )
    max_year: Optional[str] = Field(
        default=None,
        description="Latest publication year, e.g. '2024'"
    )


# ============================================================
# Core functions
# ============================================================
def resolve_concept(field: str) -> dict:
    """Resolve a field name to an OpenAlex concept ID.

    Returns: {"id": "C...", "name": "..."} or empty dict
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
    """Rebuild abstract text from OpenAlex inverted-index format.

    OpenAlex stores abstracts as {word: [positions]} inverted indices;
    words are sorted by position and concatenated into full text.
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
    """Call the OpenAlex Works API to search academic literature."""
    filters = []
    applied_filters = {
        "min_year": min_year or "",
        "max_year": max_year or "",
        "field": field,
        "concept_resolved": False,
        "concept_id": None,
        "concept_name": None,
    }

    # ---- Year range ----
    try:
        if min_year and min_year.strip():
            y1 = int(min_year.strip())
            filters.append(f"from_publication_date:{y1}-01-01")
        if max_year and max_year.strip():
            y2 = int(max_year.strip())
            filters.append(f"to_publication_date:{y2}-12-31")
    except ValueError:
        pass

    # ---- Field -> Concept ID ----
    if field and field.strip():
        concept = resolve_concept(field.strip())
        if concept.get("id"):
            filters.append(f"concepts.id:{concept['id']}")
            applied_filters["concept_resolved"] = True
            applied_filters["concept_id"] = concept["id"]
            applied_filters["concept_name"] = concept.get("name")

    # ---- API request ----
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

    # ---- Parse articles ----
    articles = []
    for item in results:
        abstract_inv = item.get("abstract_inverted_index")
        if abstract_inv:
            abstract = rebuild_abstract(abstract_inv)
        else:
            abstract = item.get("abstract") or "No abstract"

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
    """OpenAlex open academic literature search tool.

    Free academic literature search via the OpenAlex REST API, no API key required.
    Supports keyword search, field filtering (auto-resolved to concept ID),
    and year range. Returns titles, rebuilt abstracts, DOIs, citation counts,
    and concept tags.
    """

    name: str = "openalex_search"
    description: str = (
        "OpenAlex open academic literature search. Input a search keyword, "
        "returns a list of matching articles with title, abstract, DOI, and citation count. "
        "Supports research field filters (e.g. 'insect', 'molecular biology') and year ranges. "
        "Free open API, no authentication required. Complements PubMed search."
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
        """Execute OpenAlex literature search (synchronous)."""

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
# Singleton export
# ============================================================
openalex_search_tool = OpenAlexSearchTool()

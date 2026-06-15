"""
search_skills.py — Tool to retrieve domain-specific .skill files on demand.

The LLM calls this tool when it needs domain knowledge (e.g., dsRNA design guide,
safety assessment protocol). This avoids bloating the system prompt with every
skill file — only behavioral skills are always injected.

Usage:
    from tools.search_skills import search_skills_tool

    result = search_skills_tool.invoke({"query": "dsRNA design primer"})
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from skill.skill_loader import SKILL_DIR

# ============================================================
# Pydantic input schema
# ============================================================
class SearchSkillsInput(BaseModel):
    """Skill search parameters."""

    query: str = Field(
        description="Natural language description of the domain knowledge needed, "
        "e.g. 'dsRNA design primer parameters', 'safety assessment protocol'"
    )

# ============================================================
# Skill index — maps keywords to skill file names
# Add new skills here as the project grows.
# ============================================================
SKILL_INDEX: dict[str, list[str]] = {
    "dsrna_design": [
        "dsrna",
        "sirna",
        "oligowalk",
        "primer3",
        "primer design",
        "fragment design",
        "pcr",
        "interference",
        "hotspot",
    ],
    # Future skills:
    # "safety_assessment": [
    #     "safety", "off-target", "non-target", "nto", "ecosystem",
    #     "risk assessment", "blast",
    # ],
    # "target_discovery": [
    #     "target", "blast", "homolog", "gene discovery", "cds",
    # ],
    # "literature": [
    #     "literature", "pubmed", "paper", "article", "reference",
    # ],
    # "kinship": [
    #     "kinship", "phylogeny", "evolution", "divergence", "taxonomy",
    # ],
}

# Skills that are always injected (never searched)
BEHAVIORAL_SKILLS = {"principles", "evidence", "tool", "recommendation"}


def _match_skills(query: str) -> list[str]:
    """Return skill names whose keywords match the query."""
    query_lower = query.lower()
    matched: list[str] = []
    for skill_name, keywords in SKILL_INDEX.items():
        if any(kw in query_lower for kw in keywords):
            matched.append(skill_name)
    return matched


def _load_skill_content(name: str) -> str:
    """Load a .skill file by name (without extension)."""
    path = SKILL_DIR / f"{name}.skill"
    if not path.exists():
        raise FileNotFoundError(f"Skill '{name}' not found at {path}")
    return path.read_text(encoding="utf-8")


def search_skills(query: str) -> str:
    """Search domain-specific skill documents by query.

    Args:
        query: Natural language description of the domain knowledge needed
               (e.g., "dsRNA design primer", "safety assessment").

    Returns:
        The full content of matching skill document(s), or a message if none match.
    """
    matched = _match_skills(query)
    if not matched:
        return (
            f"No domain skill found matching query: '{query}'.\n"
            f"Available domain skills: {', '.join(SKILL_INDEX.keys())}.\n"
            f"Behavioral skills (always available, no need to search): "
            f"{', '.join(sorted(BEHAVIORAL_SKILLS))}."
        )

    sections: list[str] = []
    for name in matched:
        try:
            content = _load_skill_content(name)
            sections.append(content)
        except FileNotFoundError as e:
            sections.append(f"[Skill '{name}' not found on disk]")

    separator = "\n\n---\n\n"
    return separator.join(sections)


class SearchSkillsTool(BaseTool):
    """Tool for retrieving domain-specific .skill documents on demand."""

    name: str = "search_skills"
    description: str = (
        "Retrieve domain-specific design guides and protocols by keyword. "
        "Call this when you need detailed domain knowledge for a task, such as "
        "dsRNA design guidelines, primer design parameters, or safety assessment "
        "protocols. Returns the full content of the matching skill document(s). "
        "Behavioral rules (principles, evidence, tool, recommendation) are always "
        "available in your system prompt — no need to search for them."
    )
    args_schema: type[BaseModel] = SearchSkillsInput

    def _run(self, query: str) -> str:
        """Run the skill search."""
        return search_skills(query)

    async def _arun(self, query: str) -> str:
        """Async variant."""
        return self._run(query)


search_skills_tool = SearchSkillsTool()

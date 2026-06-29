"""
list_skills.py — Tool to browse available skill document metadata.

Usage:
    from tools.list_skills import list_skills_tool

    result = list_skills_tool.invoke({"category": "all"})
    result = list_skills_tool.invoke({"category": "rnai"})
"""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


# ============================================================
# Pydantic input schema
# ============================================================
class ListSkillsInput(BaseModel):
    """Skill listing parameters."""

    category: str = Field(
        default="all",
        description="Skill category filter. Categories:\n"
        "- 'all' — every registered skill\n"
        "- 'behavior' — behavior standards (analysis, shell, memory, file management, git rollback)\n"
        "- 'rnai' — RNAi workflow guides (e.g., dsrna_design)",
    )


def list_skills(category: str = "all") -> str:
    """List available skill documents with metadata.

    Args:
        category: Skill category filter. 'all' returns every category.

    Returns:
        Formatted metadata listing from SKILL_REGISTRY.
    """
    # Lazy import to avoid circular dependency
    from skills import SKILL_CATEGORIES, SKILL_REGISTRY

    if category != "all" and category not in SKILL_CATEGORIES:
        valid = ", ".join(sorted(SKILL_CATEGORIES.keys()))
        return (
            f"Unknown category: '{category}'.\n"
            f"Valid categories: {valid}."
        )

    # Determine which skill names to show
    if category == "all":
        # Collect all names from all categories
        names: set[str] = set()
        for cat in SKILL_CATEGORIES.values():
            names.update(cat["skills"])
        # Also include any skill not in a category
        names.update(SKILL_REGISTRY.keys())
        names = sorted(names)
    else:
        names = sorted(SKILL_CATEGORIES[category]["skills"])

    if not names:
        return f"No skills in category '{category}'."

    lines: list[str] = []
    for name in names:
        doc = SKILL_REGISTRY.get(name)
        if not doc:
            # Behavioral skills have no frontmatter, skip gracefully
            continue
        lines.append(f"  • {doc.name}")
        lines.append(f"    Description: {doc.description}")
        lines.append(f"    Category: {doc.category}")
        if doc.required_tools_tag:
            lines.append(f"    Required tools tag: {', '.join(doc.required_tools_tag)}")
        if doc.keywords:
            lines.append(f"    Keywords: {', '.join(doc.keywords)}")
        lines.append("")

    if not lines:
        return (
            f"Category '{category}' contains only behavioral skills "
            f"(no frontmatter — always available in system prompt)."
        )

    header = "All skills" if category == "all" else SKILL_CATEGORIES[category]["label"]
    result = f"{header} ({len(lines) // 5}):\n\n" + "\n".join(lines)
    result += "Use `read_skill` to load the full content of a skill document."
    return result


class ListSkillsTool(BaseTool):
    """Tool for listing available skill document metadata."""

    name: str = "list_skills"
    description: str = (
        "List available skill documents with descriptions, categories, "
        "required tools, and tags. Skills are domain-specific design guides "
        "and protocols. Call this BEFORE using `read_skill` to discover "
        "which skill you need."
    )
    args_schema: type[BaseModel] = ListSkillsInput

    def _run(self, category: str = "all") -> str:
        """Run the skill listing."""
        return list_skills(category)

list_skills_tool = ListSkillsTool()
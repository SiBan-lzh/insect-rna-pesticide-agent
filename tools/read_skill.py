"""
read_skill.py — Tool to load the full content of a skill document by name.

Usage:
    from tools.read_skill import read_skill_tool

    result = read_skill_tool.invoke({"skill_name": "dsrna_design"})
"""

from __future__ import annotations

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


# ============================================================
# Pydantic input schema
# ============================================================
class ReadSkillInput(BaseModel):
    """Skill reading parameters."""

    skill_name: str = Field(
        description="Name of the skill document to load (e.g., 'dsrna_design'). "
        "Use `list_skills` to discover available skill names.",
    )


def read_skill(skill_name: str) -> str:
    """Load the full content of a skill document by name.

    Args:
        skill_name: Name of the skill (without .skill extension).

    Returns:
        Full document content without frontmatter.
    """
    # Lazy import to avoid circular dependency
    from skills import SKILL_REGISTRY

    skill_name = skill_name.strip().lower()

    doc = SKILL_REGISTRY.get(skill_name)
    if doc is None:
        available = ", ".join(sorted(SKILL_REGISTRY.keys()))
        return (
            f"Skill '{skill_name}' not found.\n"
            f"Available skills: {available}.\n"
            f"Use `list_skills` to see all available skills."
        )

    return doc.content


class ReadSkillTool(BaseTool):
    """Tool for loading full skill document content."""

    name: str = "read_skill"
    description: str = (
        "Load the full content of a skill document by name. "
        "Use `list_skills` first to discover available skill names. "
        "Returns the complete design guide or protocol for the requested skill."
    )
    args_schema: type[BaseModel] = ReadSkillInput

    def _run(self, skill_name: str) -> str:
        """Run the skill reading."""
        return read_skill(skill_name)

    async def _arun(self, skill_name: str) -> str:
        """Async variant."""
        return self._run(skill_name)


read_skill_tool = ReadSkillTool()

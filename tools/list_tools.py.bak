"""
list_tools.py — Tool for the LLM to discover available tools and their schemas.

The LLM calls this tool when it needs to know what tools are available,
what each tool does, or what parameters a specific tool expects.

Usage:
    from tools.list_tools import list_tools_tool

    # List all tools
    result = list_tools_tool.invoke({"category": "all"})

    # List tools in a specific category
    result = list_tools_tool.invoke({"category": "safety"})

    # Get detailed schema for a specific tool
    result = list_tools_tool.invoke({"tool_name": "oligowalk"})
"""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function
from pydantic import BaseModel, Field


# ============================================================
# Pydantic input schema
# ============================================================
class ListToolsInput(BaseModel):
    """Tool listing parameters."""

    category: str = Field(
        default="all",
        description="Tool category filter. Categories:\n"
        "- 'all' — every registered tool\n"
        "- 'target' — BLAST, annotation, CDS extraction\n"
        "- 'sequence' — sequence cleaning (clean_seq) and clipping (clip_seq)\n"
        "- 'dsrna' — OligoWalk, Primer3\n"
        "- 'safety' — NTO BLAST, Clustal, sequence fetch\n"
        "- 'literature' — PubMed, OpenAlex\n"
        "- 'kinship' — species phylogeny\n"
        "- 'discovery' — list_tools, list_skills, read_skill, list_ragbase, list_database, search_knowledge",
    )
    tool_name: Optional[str] = Field(
        default=None,
        description="Exact tool name to get detailed schema (e.g., 'oligowalk', 'primer3'). "
        "When set, returns the full parameter schema for that single tool. "
        "Overrides the 'category' filter.",
    )


# ============================================================
# Category map (lazy import from tools.__init__ to avoid circular dep)
# ============================================================
def _get_categories():
    """Return dict: category_key -> list_of_tools_in_that_category."""
    from tools import TOOL_CATEGORIES, TOOL_REGISTRY

    cat_map = {}
    for key, cat in TOOL_CATEGORIES.items():
        cat_map[key] = list(cat["tools"])
    cat_map["all"] = list(TOOL_REGISTRY)
    return cat_map


def _get_category_labels():
    """Return dict: category_key -> human-readable label."""
    from tools import TOOL_CATEGORIES

    labels = {"all": "📋 All tools"}
    for key, cat in TOOL_CATEGORIES.items():
        labels[key] = cat["label"]
    return labels


def _get_all_tools():
    from tools import TOOL_REGISTRY
    return TOOL_REGISTRY


def _get_tool_by_name(name: str):
    """Find a tool by its exact name."""
    all_tools = _get_all_tools()
    for t in all_tools:
        if t.name == name:
            return t
    return None


def list_tools(category: str = "all", tool_name: Optional[str] = None) -> str:
    """List available tools and their descriptions or detailed schemas.

    Args:
        category: Tool category filter. 'all' returns every registered tool.
        tool_name: Exact tool name for detailed schema. Overrides category.

    Returns:
        Formatted tool listing or detailed schema for a single tool.
    """
    categories = _get_categories()
    category_labels = _get_category_labels()
    all_tools = _get_all_tools()

    # Mode 1: detailed schema for a specific tool
    if tool_name:
        tool = _get_tool_by_name(tool_name)
        if not tool:
            available = ", ".join(sorted(t.name for t in all_tools))
            return (
                f"Tool '{tool_name}' not found.\n"
                f"Available tools: {available}.\n"
                f"Use list_tools(category='all') to see all tools."
            )
        schema = convert_to_openai_function(tool)
        lines = [f"=== {tool_name} ==="]
        lines.append(f"Description: {schema.get('description', 'N/A')}")
        params = schema.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])
        if props:
            lines.append("Parameters:")
            for pname, pinfo in props.items():
                is_req = pname in required
                ptype = pinfo.get("type", "any")
                pdesc = pinfo.get("description", "")
                lines.append(f"  [{pname}] type={ptype}, required={is_req}")
                if pdesc:
                    lines.append(f"    {pdesc}")
        else:
            lines.append("Parameters: none")
        return "\n".join(lines)

    # Mode 2: category listing
    tools_in_category = categories.get(category)
    if tools_in_category is None:
        valid_cats = ", ".join(sorted(categories.keys()))
        return (
            f"Unknown category: '{category}'.\n"
            f"Valid categories: {valid_cats}."
        )

    if not tools_in_category:
        return f"No tools in category '{category}'."

    label = category_labels.get(category, category)
    lines = [f"{label} ({len(tools_in_category)}):"]
    lines.append("")
    for t in tools_in_category:
        desc_short = t.description.split(".")[0].strip()
        lines.append(f"  {t.name} — {desc_short}.")
    lines.append("")
    lines.append("Use list_tools(tool_name='<name>') for detailed parameters.")
    return "\n".join(lines)


class ListToolsTool(BaseTool):
    """Tool for listing available tools and their schemas."""

    name: str = "list_tools"
    description: str = (
        "List available bioinformatics tools and their parameter schemas. "
        "Call this when you need to discover what tools are available, check "
        "which tools exist in a category, or inspect the parameters of a "
        "specific tool. Use category='all' for a complete overview, or "
        "set tool_name to a specific tool name (e.g., 'oligowalk') for "
        "detailed parameter documentation."
    )
    args_schema: type[BaseModel] = ListToolsInput

    def _run(self, category: str = "all", tool_name: Optional[str] = None) -> str:
        """Run the tool listing."""
        return list_tools(category, tool_name)

    async def _arun(self, category: str = "all", tool_name: Optional[str] = None) -> str:
        """Async variant."""
        return self._run(category, tool_name)


list_tools_tool = ListToolsTool()

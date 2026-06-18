"""
tool_loader.py — Auto-discover tools from subdirectory structure.

Scans tools/ for subdirectories, reads metadata.md from each,
dynamically imports the tool instance from tool.py, and registers
all discovered tools with auto-generated categories.

Directory convention:
    tools/<tool_name>/
        ├── metadata.md   — YAML frontmatter (name, tag, description, parameters)
        ├── config.py     — Tool-specific configuration (optional)
        ├── tool.py       — Exports <tool_name>_tool instance (BaseTool subclass)
        └── __init__.py   — Re-exports from .tool for backward compatibility

Usage:
    from tools.tool_loader import TOOL_REGISTRY, TOOL_CATEGORIES
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import List, Tuple

from langchain_core.tools import BaseTool

logger = logging.getLogger("RPA_Tools.ToolLoader")

# ============================================================
# Tools directory — loader lives in tools/, so parent is the dir
# ============================================================
TOOLS_DIR = Path(__file__).parent.resolve()


# ============================================================
# Metadata parsing
# ============================================================
def _parse_metadata(md_path: Path) -> dict | None:
    """Parse YAML frontmatter from metadata.md.

    Returns dict with keys: name, tag, description, parameters.
    Returns None if file missing or frontmatter invalid.
    """
    if not md_path.exists():
        return None

    raw = md_path.read_text(encoding="utf-8")
    if not raw.startswith("---"):
        return None

    parts = raw.split("---", 2)
    if len(parts) < 3:
        return None

    frontmatter_text = parts[1].strip()
    if not frontmatter_text:
        return None

    try:
        import yaml
        meta = yaml.safe_load(frontmatter_text)
    except Exception:
        return None

    if not isinstance(meta, dict) or "name" not in meta:
        return None

    return meta


# ============================================================
# Tool discovery
# ============================================================
def _discover_tools(tools_dir: Path = TOOLS_DIR) -> Tuple[List[BaseTool], dict[str, str]]:
    """Scan tools/ subdirectories and discover tools.

    For each subdirectory containing both metadata.md and tool.py:
    1. Read metadata.md for validation and tag extraction
    2. Dynamically import tools.<dirname>.tool
    3. Find the variable ending with '_tool' that is a BaseTool instance

    Returns:
        (registry, name_to_tag) — tool instances and their tag from metadata.
    """
    registry: list[BaseTool] = []
    name_to_tag: dict[str, str] = {}

    for entry in sorted(tools_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("_") or entry.name.startswith("."):
            continue
        if entry.name == "__pycache__":
            continue

        metadata_md = entry / "metadata.md"
        tool_py = entry / "tool.py"

        if not metadata_md.exists() or not tool_py.exists():
            continue

        # Parse metadata
        meta = _parse_metadata(metadata_md)
        if meta is None:
            logger.warning("Skipping '%s': invalid or missing metadata.md", entry.name)
            continue

        tool_name = meta["name"]
        tag = meta.get("tag", "other")

        # Dynamic import
        module_name = f"tools.{entry.name}.tool"
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            logger.warning("Skipping '%s': import failed: %s", entry.name, e)
            continue

        # Find tool instance (variable ending with '_tool')
        found = False
        for var_name in dir(module):
            if not var_name.endswith("_tool"):
                continue
            instance = getattr(module, var_name, None)
            if isinstance(instance, BaseTool):
                registry.append(instance)
                name_to_tag[tool_name] = tag
                found = True
                break

        if not found:
            logger.warning(
                "Skipping '%s': no BaseTool instance found (expected '*_tool' variable)",
                entry.name,
            )

    # Deduplicate by tool name
    seen: set[str] = set()
    unique_registry: list[BaseTool] = []
    for t in registry:
        if t.name not in seen:
            seen.add(t.name)
            unique_registry.append(t)

    return unique_registry, name_to_tag


# ============================================================
# Categorization
# ============================================================
def _categorize(
    tools: list[BaseTool],
    name_to_tag: dict[str, str],
) -> dict[str, dict]:
    """Group tools by their tag from metadata.md.

    The tag value from metadata is used directly as the display label — no mapping.

    Returns:
        dict like {"target_discovery": {"label": "target discovery", "tools": [...]}}
    """
    tag_to_tools: dict[str, list[BaseTool]] = {}
    for t in tools:
        tag = name_to_tag.get(t.name, "other")
        if tag not in tag_to_tools:
            tag_to_tools[tag] = []
        tag_to_tools[tag].append(t)

    categories: dict[str, dict] = {}
    for tag, tools_in_tag in tag_to_tools.items():
        key = tag.lower().replace(" ", "_")
        categories[key] = {
            "label": tag,
            "tools": tools_in_tag,
        }

    return categories


# ============================================================
# Module-level registries
# ============================================================
_TOOL_REGISTRY_TUPLE = _discover_tools()
TOOL_REGISTRY: list[BaseTool] = _TOOL_REGISTRY_TUPLE[0]
_NAME_TO_TAG: dict[str, str] = _TOOL_REGISTRY_TUPLE[1]
TOOL_CATEGORIES: dict[str, dict] = _categorize(TOOL_REGISTRY, _NAME_TO_TAG)

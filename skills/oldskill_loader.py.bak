"""
skill_loader.py — Skill document loader with frontmatter parsing.

Two independent paths:
  1. SKILL_REGISTRY — auto-discovered from .skill files with YAML frontmatter
  2. load_skill() / build_skills() — raw file reading, no frontmatter parsing

Usage:
    from skills.skill_loader import SKILL_REGISTRY, SKILL_CATEGORIES, build_skills
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from tool_config import SKILLS_DIR


# ============================================================
# Skill document data model
# ============================================================
@dataclass
class SkillDoc:
    """Structured representation of a .skill file."""

    name: str
    description: str
    category: str = "domain"
    required_tools: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    content: str = ""


# ============================================================
# Frontmatter parsing
# ============================================================
def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter (--- ... ---) from skill file text.

    Args:
        text: Raw file content.

    Returns:
        (metadata_dict, body_without_frontmatter).
        If no frontmatter found, returns ({}, text).
    """
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    frontmatter_text = parts[1].strip()
    body = parts[2].strip()

    if not frontmatter_text:
        return {}, body

    try:
        import yaml
        meta = yaml.safe_load(frontmatter_text)
    except Exception:
        return {}, text

    if not isinstance(meta, dict):
        return {}, body

    return meta, body


# ============================================================
# Skill discovery
# ============================================================
def _discover_skills(skill_dir: Path = SKILLS_DIR) -> dict[str, SkillDoc]:
    """Scan skill_dir and build SKILL_REGISTRY from frontmatter.

    Files without YAML frontmatter are skipped (behavioral skills).
    """
    registry: dict[str, SkillDoc] = {}

    for path in sorted(skill_dir.glob("*.skill")):
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw)

        if not meta or "name" not in meta:
            continue

        name = meta["name"]
        registry[name] = SkillDoc(
            name=name,
            description=meta.get("description", ""),
            category=meta.get("category", "domain"),
            required_tools=meta.get("required_tools", []),
            tags=meta.get("tags", []),
            content=body,
        )

    return registry


# ============================================================
# Module-level registries
# ============================================================
SKILL_REGISTRY: dict[str, SkillDoc] = _discover_skills()

SKILL_CATEGORIES: dict[str, dict] = {
    "behavior": {
        "label": "Behavior standards",
        "skills": ["principles", "evidence", "tool", "recommendation"],
    },
    "rnai": {
        "label": "RNAi workflow",
        "skills": ["dsrna_design"],
    },
}


# ============================================================
# Raw file reading (independent path — no frontmatter parsing)
# ============================================================
@lru_cache(maxsize=None)
def load_skill(name: str, skill_dir: Path = SKILLS_DIR) -> str:
    """Load a single .skill file as raw text.

    Reads the full file content as-is. No frontmatter parsing.
    Used for behavioral skills that have no frontmatter.

    Args:
        name: Skill name (with or without .skill extension).
        skill_dir: Directory containing .skill files.

    Returns:
        Full file content as string.
    """
    path = skill_dir / (name if name.endswith(".skill") else f"{name}.skill")
    return path.read_text(encoding="utf-8")


def build_skills(
    *names: str,
    skill_dir: Path = SKILLS_DIR,
    separator: str = "\n\n---\n\n",
) -> str:
    """Load and concatenate multiple skill files.

    Used for behavioral skills injected into system prompt.

    Args:
        *names: Skill names to load.
        skill_dir: Directory containing .skill files.
        separator: Separator between concatenated skills.

    Returns:
        Concatenated skill content.
    """
    return separator.join(load_skill(n, skill_dir) for n in names)

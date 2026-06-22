"""
skill_loader.py — Skill document loader with frontmatter parsing.

Single path: every .skill file carries YAML frontmatter and is
auto-discovered into SKILL_REGISTRY via recursive directory scan.
load_skill() / build_skills() read content out of SKILL_REGISTRY.

Usage:
    from skills.skill_loader import SKILL_REGISTRY, SKILL_CATEGORIES, build_skills
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Skills directory — loader lives in skills/, so parent is the dir
SKILLS_DIR = Path(__file__).parent.resolve()


# ============================================================
# Skill document data model
# ============================================================
@dataclass
class SkillDoc:
    """Structured representation of a .skill file."""

    name: str
    description: str
    category: str = "domain"
    keywords: list[str] = field(default_factory=list)
    required_tools_tag: list[str] = field(default_factory=list)
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
        If no frontmatter found, or parsing fails, returns ({}, text).
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
# Skill discovery — recursive scan of all subdirectories
# ============================================================
def _discover_skills(skill_dir: Path = SKILLS_DIR) -> dict[str, SkillDoc]:
    """Recursively scan skill_dir for .skill files and build registry.

    Every .skill file is expected to carry YAML frontmatter.
    Files without valid frontmatter are skipped with a warning.
    """
    registry: dict[str, SkillDoc] = {}

    for path in sorted(skill_dir.glob("**/*.skill")):
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw)

        if not meta or "name" not in meta:
            print(f"[skill_loader] WARNING: {path.relative_to(skill_dir)} has no valid frontmatter, skipped.")
            continue

        name = meta["name"]
        registry[name] = SkillDoc(
            name=name,
            description=meta.get("description", ""),
            category=meta.get("category", "domain"),
            keywords=meta.get("keywords", meta.get("tags", [])),
            required_tools_tag=meta.get("required_tools_tag", []),
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
        "skills": ["behavior1"],
    },
    "rnai": {
        "label": "RNAi workflow",
        "skills": ["dsrna_design", "safety_inspection"],
    },
}


# ============================================================
# Skill content access — reads from SKILL_REGISTRY, not disk
# ============================================================
def load_skill(name: str) -> str:
    """Return the body content of a single skill by name.

    Looks up SKILL_REGISTRY — the skill must have been discovered via
    frontmatter at import time. Raises KeyError if not found, with a
    hint listing available names.

    Args:
        name: Skill name as declared in its frontmatter (no .skill extension).

    Returns:
        The skill's content (frontmatter stripped).
    """
    doc = SKILL_REGISTRY.get(name)
    if doc is None:
        available = ", ".join(sorted(SKILL_REGISTRY.keys()))
        raise KeyError(f"Skill '{name}' not found. Available: {available}")
    return doc.content


def build_skills(*names: str, separator: str = "\n\n---\n\n") -> str:
    """Load and concatenate multiple skills' content by name.

    Used to assemble the behavioral-skill block injected into the
    agent's system prompt (multiple skills are concatenated together).

    Args:
        *names: Skill names to load, in order.
        separator: Separator between concatenated skills.

    Returns:
        Concatenated skill content.
    """
    return separator.join(load_skill(n) for n in names)

"""
skill_loader.py — Lightweight .skill file loader.

Usage:
    from skill.skill_loader import load_skill, build_skills

    protocol = build_skills("principles", "evidence", "tool")
    # => concatenated content of principles.skill + evidence.skill + tool.skill
"""

from functools import lru_cache
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def load_skill(name: str) -> str:
    """Load a single .skill file by name (omit the .skill extension)."""
    path = SKILL_DIR / (name if name.endswith(".skill") else f"{name}.skill")
    return path.read_text(encoding="utf-8")


def build_skills(*names: str, separator: str = "\n\n---\n\n") -> str:
    """Load and concatenate multiple skill files."""
    return separator.join(load_skill(n) for n in names)

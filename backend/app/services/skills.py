"""Agent Skills 支持：发现、解析与加载。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..core.config import settings
from ..core.logging import app_logger

logger = app_logger

SKILLS_DIR = settings.base_dir / "skills"
_skills_cache: Dict[str, "SkillRecord"] = {}
_skills_signature: Optional[float] = None


@dataclass(frozen=True)
class SkillRecord:
    name: str
    description: str
    path: Path


def _skills_mtime_signature() -> Optional[float]:
    if not SKILLS_DIR.exists():
        return None
    latest = 0.0
    found = False
    for skill_file in SKILLS_DIR.glob("*/SKILL.md"):
        try:
            latest = max(latest, skill_file.stat().st_mtime)
            found = True
        except OSError:
            continue
    return latest if found else None


def _parse_frontmatter(content: str) -> dict[str, str]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    end_idx = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_idx = idx
            break
    if end_idx is None:
        return {}
    fields: dict[str, str] = {}
    for line in lines[1:end_idx]:
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        fields[key.strip()] = value.strip()
    return fields


def _build_skill_registry() -> Dict[str, SkillRecord]:
    registry: Dict[str, SkillRecord] = {}
    if not SKILLS_DIR.exists():
        return registry
    for skill_file in SKILLS_DIR.glob("*/SKILL.md"):
        try:
            content = skill_file.read_text(encoding="utf-8")
        except OSError:
            continue
        meta = _parse_frontmatter(content)
        name = (meta.get("name") or "").strip()
        description = (meta.get("description") or "").strip()
        if not name or not description:
            continue
        dir_name = skill_file.parent.name
        if name != dir_name:
            logger.warning(
                "Skill name mismatch",
                extra={"skill_name": name, "dir_name": dir_name, "path": str(skill_file)},
            )
            continue
        registry[name] = SkillRecord(name=name, description=description, path=skill_file)
    return registry


def list_skills() -> List[SkillRecord]:
    global _skills_cache, _skills_signature
    signature = _skills_mtime_signature()
    if signature is None:
        _skills_cache = {}
        _skills_signature = None
        return []
    if _skills_signature == signature and _skills_cache:
        return list(_skills_cache.values())
    registry = _build_skill_registry()
    _skills_cache = registry
    _skills_signature = signature
    return list(registry.values())


def load_skill_content(skill_name: str) -> str:
    if not skill_name:
        raise ValueError("skill_name must be provided")
    registry = {record.name: record for record in list_skills()}
    record = registry.get(skill_name)
    if record is None:
        raise ValueError(f"skill not found: {skill_name}")
    return record.path.read_text(encoding="utf-8")


def build_skills_prompt() -> str:
    skills = list_skills()
    if not skills:
        return ""
    lines = [
        "【可用技能】",
        "以下技能可按需使用，请先调用 load_skill 加载完整说明：",
    ]
    for record in sorted(skills, key=lambda item: item.name):
        lines.append(f"- {record.name}: {record.description}")
    return "\n".join(lines)


__all__ = ["SkillRecord", "list_skills", "load_skill_content", "build_skills_prompt"]

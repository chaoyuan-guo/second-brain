"""工具函数实现。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.config import settings
from ..core.logging import app_logger
from .exceptions import ToolExecutionError

logger = app_logger


def read_note_file(
    path: str,
    *,
    offset: int = 0,
    limit_chars: int = 60000,
) -> dict[str, Any]:
    """读取指定笔记文件的文本片段（按字符偏移）。"""

    if not path:
        raise ValueError("Path must be provided")

    base_dir = (settings.base_dir / "data" / "notes" / "my_markdowns").resolve()
    raw_path = Path(path)
    if raw_path.is_absolute():
        resolved = raw_path.resolve()
    else:
        candidate = raw_path
        if str(candidate).startswith("data/notes/my_markdowns"):
            candidate = settings.base_dir / candidate
        else:
            candidate = base_dir / candidate
        resolved = candidate.resolve()

    if resolved != base_dir and base_dir not in resolved.parents:
        raise ToolExecutionError("只允许读取 data/notes/my_markdowns/ 下的文件。")
    if not resolved.exists():
        raise ToolExecutionError(f"文件不存在: {resolved}")
    if resolved.is_dir():
        raise ToolExecutionError(f"路径是目录，无法读取: {resolved}")

    try:
        offset_value = int(offset)
    except (TypeError, ValueError) as exc:
        raise ValueError("offset must be an integer") from exc
    if offset_value < 0:
        raise ValueError("offset must be >= 0")

    try:
        limit_value = int(limit_chars)
    except (TypeError, ValueError) as exc:
        raise ValueError("limit_chars must be an integer") from exc
    if limit_value <= 0:
        raise ValueError("limit_chars must be > 0")
    if limit_value > 60000:
        limit_value = 60000

    text = resolved.read_text(encoding="utf-8")
    total_chars = len(text)
    start = min(offset_value, total_chars)
    end = min(start + limit_value, total_chars)
    content = text[start:end]
    next_offset = end if end < total_chars else None

    logger.info(
        "Note file read",
        extra={
            "path": str(resolved),
            "offset": start,
            "limit_chars": limit_value,
            "total_chars": total_chars,
            "returned_chars": len(content),
        },
    )

    result = {
        "source_file": str(resolved),
        "offset": start,
        "limit_chars": limit_value,
        "total_chars": total_chars,
        "next_offset": next_offset,
        "done": end >= total_chars,
        "content": content,
    }

    if end >= total_chars:
        result["read_progress"] = {
            "status": "complete",
            "message": "已读完整文件",
        }
    else:
        percent = int((end / total_chars) * 100) if total_chars > 0 else 0
        result["read_progress"] = {
            "status": "incomplete",
            "percent": percent,
            "read_chars": end,
            "total_chars": total_chars,
            "next_call": {"path": path, "offset": next_offset},
            "message": f"仅读取 {percent}%，需继续读取",
        }

    return result


__all__ = [
    "read_note_file",
]

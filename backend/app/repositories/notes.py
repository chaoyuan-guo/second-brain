"""笔记索引仓储。"""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any, List, Optional

import faiss
import json

from ..core.config import settings
from ..services.exceptions import ToolExecutionError

_faiss_index: Optional[faiss.Index] = None
_metadata: Optional[List[dict[str, Any]]] = None
_lock = Lock()


def load_index() -> faiss.Index:
    """加载 FAISS 索引。"""

    index_path = settings.faiss_index_path
    if not index_path.exists():
        raise ToolExecutionError("my_notes.index 不存在，请先运行索引构建脚本。")

    global _faiss_index
    with _lock:
        if _faiss_index is None:
            _faiss_index = faiss.read_index(str(index_path))
    return _faiss_index


def load_metadata() -> List[dict[str, Any]]:
    """加载向量对应的元数据。"""

    metadata_path = settings.faiss_metadata_path
    if not metadata_path.exists():
        raise ToolExecutionError("my_notes_metadata.json 不存在，无法映射检索结果。")

    global _metadata
    with _lock:
        if _metadata is None:
            _metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return _metadata


__all__ = ["load_index", "load_metadata"]


"""Pydantic 数据模型定义."""

from __future__ import annotations

from pydantic import BaseModel


class NoteUploadResponse(BaseModel):
    """上传笔记后的响应."""

    message: str
    file_name: str
    chunks_added: int
    replaced: bool
    removed_vectors: int
    total_vectors: int


__all__ = [
    "NoteUploadResponse",
]

"""FastAPI 路由定义。"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from ..models.schemas import NoteUploadResponse
from ..services.exceptions import ToolExecutionError
from ..services.notes_index import update_notes_index_from_upload

router = APIRouter()


@router.post("/notes/upload")
async def upload_note(file: UploadFile = File(...)) -> NoteUploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    if not file.filename.lower().endswith(".md"):
        raise HTTPException(status_code=400, detail="Only .md files are supported.")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = await asyncio.to_thread(update_notes_index_from_upload, file.filename, raw_bytes)
    except ToolExecutionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected upload failure
        raise HTTPException(status_code=500, detail="Failed to process upload.") from exc

    return NoteUploadResponse(**result)


@router.get("/notes/content")
async def get_note_content(
    path: str,
    offset: int = Query(default=0, ge=0),
    limit_chars: int = Query(default=12000, gt=0, le=60000),
) -> dict[str, object]:
    """读取指定笔记文件的内容，供前端预览使用。"""

    from ..services.tools import read_note_file

    try:
        result = await asyncio.to_thread(read_note_file, path, offset=offset, limit_chars=limit_chars)
    except ToolExecutionError as exc:
        detail = str(exc)
        if "只允许读取" in detail:
            status_code = 403
        elif "文件不存在" in detail:
            status_code = 404
        else:
            status_code = 400
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="读取文件失败") from exc

    return {
        "content": result.get("content", ""),
        "done": result.get("done", False),
        "next_offset": result.get("next_offset"),
        "total_chars": result.get("total_chars"),
        "offset": result.get("offset"),
        "limit_chars": result.get("limit_chars"),
    }
__all__ = ["router"]

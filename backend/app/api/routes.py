"""FastAPI 路由定义。"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, File, HTTPException, UploadFile

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
__all__ = ["router"]

"""FastAPI 路由定义。"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from urllib.parse import unquote
from fastapi.responses import FileResponse, StreamingResponse

from ..core.config import settings
from ..models.schemas import ChatRequest, ChatResponse, ChatTitleResponse, NoteUploadResponse
from ..services.notes_index import update_notes_index_from_upload
from ..services.exceptions import ToolExecutionError
from ..services.chat import execute_chat, generate_title, stream_chat_response

router = APIRouter()


@router.get("/hello")
def read_hello(input: str) -> dict[str, str]:
    """Echo 输入。"""

    return {"message": f"Hello, World {input}"}


@router.post("/chat")
async def chat_completion(request: Request, payload: ChatRequest) -> ChatResponse:
    eval_context = _extract_eval_context(request)
    return await execute_chat(payload, eval_context=eval_context)


@router.post("/chat/stream", response_class=StreamingResponse)
async def chat_completion_stream(request: Request, payload: ChatRequest) -> StreamingResponse:
    stream_format = request.headers.get("x-stream-format")
    accept = request.headers.get("accept", "")
    want_ndjson = stream_format == "ndjson" or "application/x-ndjson" in accept
    eval_context = _extract_eval_context(request)
    return await stream_chat_response(payload, want_ndjson=want_ndjson, eval_context=eval_context)


@router.post("/chat/title")
async def chat_title(payload: ChatRequest) -> ChatTitleResponse:
    return await generate_title(payload)


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


@router.get("/", include_in_schema=False)
async def serve_root() -> FileResponse:
    if not settings.frontend_index_file.exists():
        raise HTTPException(status_code=404, detail="前端静态资源尚未构建")
    return FileResponse(settings.frontend_index_file)


@router.get("/{resource_path:path}", include_in_schema=False)
async def serve_frontend(resource_path: str) -> FileResponse:
    if not settings.frontend_build_dir.exists():
        raise HTTPException(status_code=404, detail="前端静态资源尚未构建")

    candidate = (settings.frontend_build_dir / resource_path).resolve()
    try:
        candidate.relative_to(settings.frontend_build_dir)
    except ValueError:
        candidate = None

    if candidate and candidate.is_file():
        return FileResponse(candidate)

    if settings.frontend_index_file.exists():
        return FileResponse(settings.frontend_index_file)

    raise HTTPException(status_code=404, detail="前端静态资源缺失")


__all__ = ["router"]


def _extract_eval_context(request: Request) -> dict[str, object]:
    strict = request.headers.get("x-eval-strict", "").strip().lower() in {"1", "true", "yes", "on"}
    question_id = request.headers.get("x-eval-question-id", "").strip()
    expected_sources_raw = request.headers.get("x-eval-expected-sources", "")
    expected_sources = []
    for item in expected_sources_raw.split(","):
        item = item.strip()
        if not item:
            continue
        expected_sources.append(unquote(item))
    return {
        "strict": strict,
        "question_id": question_id,
        "expected_sources": expected_sources,
    }

"""FastAPI 路由定义。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from ..core.config import settings
from ..models.schemas import ChatRequest, ChatResponse, ChatTitleResponse
from ..services.chat import execute_chat, generate_title, stream_chat_response

router = APIRouter()


@router.get("/hello")
def read_hello(input: str) -> dict[str, str]:
    """Echo 输入。"""

    return {"message": f"Hello, World {input}"}


@router.post("/chat")
async def chat_completion(payload: ChatRequest) -> ChatResponse:
    return await execute_chat(payload)


@router.post("/chat/stream", response_class=StreamingResponse)
async def chat_completion_stream(payload: ChatRequest) -> StreamingResponse:
    return await stream_chat_response(payload)


@router.post("/chat/title")
async def chat_title(payload: ChatRequest) -> ChatTitleResponse:
    return await generate_title(payload)


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

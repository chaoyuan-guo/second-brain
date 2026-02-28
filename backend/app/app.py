"""FastAPI 应用工厂。"""

from __future__ import annotations

import time
import uuid

from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .core.config import settings
from .core.logging import app_logger


def create_app() -> FastAPI:
    app = FastAPI(title="Second Brain API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def trace_requests(request: Request, call_next):  # type: ignore[override]
        trace_id = request.headers.get("x-request-id", "").strip() or uuid.uuid4().hex
        start = time.perf_counter()
        app_logger.info(
            "HTTP request start %s %s",
            request.method,
            request.url.path,
            extra={"trace_id": trace_id},
        )
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            app_logger.exception(
                "HTTP request failed %s %s duration_ms=%s",
                request.method,
                request.url.path,
                duration_ms,
                extra={"trace_id": trace_id},
            )
            raise

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["x-request-id"] = trace_id
        app_logger.info(
            "HTTP request done %s %s status=%s duration_ms=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            extra={"trace_id": trace_id},
        )
        return response

    app.include_router(router)
    return app


app = create_app()


__all__ = ["app", "create_app"]

"""FastAPI 入口，适配 PORT 环境变量。"""

from __future__ import annotations

import os
import uvicorn

from .app import app


def run() -> None:
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    run()


__all__ = ["app", "run"]

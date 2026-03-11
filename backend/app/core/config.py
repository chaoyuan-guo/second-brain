"""全局配置与常量。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field


def is_truthy(value: str | None) -> bool:
    """将环境变量式的字符串转换为布尔值。"""

    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
NOTES_DIR = DATA_DIR / "notes"
INDEX_DIR = DATA_DIR / "indexes"
RUNTIME_DIR = PROJECT_ROOT / "runtime"
LOGS_DIR = RUNTIME_DIR / "logs"

for directory in (DATA_DIR, NOTES_DIR, INDEX_DIR, RUNTIME_DIR, LOGS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DEFAULT_API_BASE_URL = "https://space.ai-builders.com/backend/v1"
DEFAULT_ALLOWED_ORIGINS = "http://localhost:9080,http://127.0.0.1:9080"


def running_in_container() -> bool:
    """检测是否运行在 Docker/Koyeb 容器内。"""

    return Path("/.dockerenv").exists()

# OpenAI / Azure OpenAI 请求超时（秒）。
OPENAI_DEFAULT_TIMEOUT_SECONDS = 120.0

# OpenAI SDK 内置重试次数（默认 2 次）。
# 该值控制 SDK 内部的自动重试，减少后可加快超时失败的返回速度。
OPENAI_MAX_RETRIES = 2


def _parse_allowed_origins(raw: str | None) -> List[str]:
    origins = raw or DEFAULT_ALLOWED_ORIGINS
    return [origin.strip() for origin in origins.split(",") if origin.strip()]


class Settings(BaseModel):
    """集中管理运行配置。"""

    base_dir: Path = Field(default=PROJECT_ROOT, frozen=True)
    log_path: Path = Field(default=LOGS_DIR / "backend.log")
    tool_log_path: Path = Field(default=LOGS_DIR / "tool_output.log")
    faiss_index_path: Path = Field(default=INDEX_DIR / "my_notes.index")
    faiss_metadata_path: Path = Field(default=NOTES_DIR / "my_notes_metadata.json")
    api_key: str
    api_base_url: str = Field(default=DEFAULT_API_BASE_URL)
    allowed_origins: List[str] = Field(default_factory=list)


def load_settings() -> Settings:
    """构建 Settings 并做必要校验。"""

    api_key = os.getenv("SUPER_MIND_API_KEY") or os.getenv("AI_BUILDER_TOKEN")
    if not api_key:
        raise RuntimeError(
            "Missing API token: set SUPER_MIND_API_KEY or rely on AI_BUILDER_TOKEN."
        )

    return Settings(
        api_key=api_key,
        api_base_url=os.getenv("SUPER_MIND_API_BASE_URL", DEFAULT_API_BASE_URL),
        allowed_origins=_parse_allowed_origins(os.getenv("CHAT_ALLOWED_ORIGINS")),
    )


settings = load_settings()


__all__ = [
    "settings",
    "OPENAI_DEFAULT_TIMEOUT_SECONDS",
    "OPENAI_MAX_RETRIES",
    "is_truthy",
]

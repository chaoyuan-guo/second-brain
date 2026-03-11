"""OpenAI 客户端初始化。"""

from __future__ import annotations

from openai import OpenAI

from ..core.config import settings
from ..core.config import OPENAI_DEFAULT_TIMEOUT_SECONDS, OPENAI_MAX_RETRIES


def _create_client() -> OpenAI:
    return OpenAI(
        api_key=settings.api_key,
        base_url=settings.api_base_url,
        timeout=OPENAI_DEFAULT_TIMEOUT_SECONDS,
        max_retries=OPENAI_MAX_RETRIES,
    )


client = _create_client()


__all__ = ["client"]

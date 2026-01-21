"""OpenAI 客户端初始化。"""

from __future__ import annotations

from openai import AsyncOpenAI, OpenAI

from ..core.config import settings
from ..core.config import OPENAI_DEFAULT_TIMEOUT_SECONDS
from ..core.logging import app_logger


def _create_clients() -> tuple[OpenAI, AsyncOpenAI, str]:
    client = OpenAI(
        api_key=settings.api_key,
        base_url=settings.api_base_url,
        timeout=OPENAI_DEFAULT_TIMEOUT_SECONDS,
    )

    if settings.use_azure_chat:
        if not all(
            [
                settings.azure_base_url,
                settings.azure_api_key,
                settings.azure_api_version,
                settings.azure_chat_model_name,
            ]
        ):
            raise RuntimeError("Azure OpenAI settings are incomplete.")
        normalized_base = settings.azure_base_url.rstrip("/")
        lowered = normalized_base.lower()
        if lowered.endswith("/openai/v1"):
            normalized_base = normalized_base[: -len("/v1")]
        elif not lowered.endswith("/openai"):
            normalized_base = f"{normalized_base}/openai"
        azure_v1_base = f"{normalized_base.rstrip('/')}/v1/"
        chat_client = AsyncOpenAI(
            api_key=settings.azure_api_key,
            base_url=azure_v1_base,
            timeout=OPENAI_DEFAULT_TIMEOUT_SECONDS,
        )
        chat_model_name = settings.azure_chat_model_name
        app_logger.info(
            "Azure chat endpoint configured",
            extra={"azure_base": azure_v1_base, "azure_api_version": settings.azure_api_version},
        )
    else:
        chat_client = AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.api_base_url,
            timeout=OPENAI_DEFAULT_TIMEOUT_SECONDS,
        )
        chat_model_name = settings.chat_model_name

    return client, chat_client, chat_model_name


client, chat_client, chat_model_name = _create_clients()


__all__ = ["client", "chat_client", "chat_model_name"]

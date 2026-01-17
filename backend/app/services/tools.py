"""工具函数实现。"""

from __future__ import annotations

import json
import subprocess
import time
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar

import httpx
import numpy as np
from bs4 import BeautifulSoup
from openai import (
    APITimeoutError,
    APIConnectionError,
    APIStatusError,
    RateLimitError,
)

from ..core.config import (
    CHAT_API_MAX_RETRIES,
    CHAT_API_RETRY_BACKOFF_SECONDS,
    RETRYABLE_STATUS_CODES,
    settings,
)
from ..core.logging import app_logger
from ..repositories.notes import load_index, load_metadata
from .clients import chat_client, client, chat_model_name
from .exceptions import ToolExecutionError

logger = app_logger


API_BASE_URL = settings.api_base_url
SEARCH_API_URL = f"{API_BASE_URL}/search/"
MCP_BRIDGE_SCRIPT = settings.mcp_bridge_script
DEFAULT_MCP_DRIVER = settings.mcp_driver_path
DEFAULT_MCP_COMMAND = settings.mcp_command_path
DEFAULT_MCP_WORKDIR = settings.mcp_workdir
DEFAULT_MCP_ENDPOINT = settings.mcp_endpoint
_mcp_python_path = settings.mcp_python_path

_mcp_command = Path(DEFAULT_MCP_COMMAND)
_mcp_workdir = Path(DEFAULT_MCP_WORKDIR)
_mcp_endpoint = DEFAULT_MCP_ENDPOINT

T = TypeVar("T")


def _sync_mcp_workspace() -> None:
    """在 MCP 工作目录下准备笔记目录。

    本地开发通常由 start_services.sh 负责同步；容器环境下没有该脚本，
    因此在首次调用解释器前做一次轻量同步，确保解释器能按相对路径
    访问 data/notes/my_markdowns。
    """

    source_dir = settings.base_dir / "data" / "notes" / "my_markdowns"
    if not source_dir.exists():
        return

    dest_dir = _mcp_workdir / "data" / "notes" / "my_markdowns"
    if dest_dir.exists():
        return

    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        dest_dir.symlink_to(source_dir, target_is_directory=True)
    except OSError:
        shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)


def looks_like_raw_markdown(text: str | None) -> bool:
    """粗略判断输出是否只是 Markdown 原文，没有结构化数据。"""

    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    md_signals = ("# ", "## ", "```", "- ", "* ")
    has_md = any(token in stripped for token in md_signals)
    has_json = stripped.startswith("{") or stripped.startswith("[")
    has_table = "|" in stripped.splitlines()[0] if stripped else False
    return has_md and not (has_json or has_table)


def looks_like_interpreter_error(text: str | None) -> bool:
    """判断代码解释器输出是否包含典型错误栈或 Traceback。"""

    if not text:
        return False
    lowered = text.lower()
    error_keywords: Iterable[str] = (
        "traceback (most recent call last)",
        "error:",
        "exception",
        "stack trace",
        "file \"",
        "line ",
    )
    return any(keyword in lowered for keyword in error_keywords)


def looks_like_incomplete_insight(text: str | None) -> bool:
    """检测输出是否缺少指标或只包含模板提示。"""

    if not text:
        return True
    lowered = text.lower()
    incomplete_markers = (
        "todo",
        "tbd",
        "placeholder",
        "no data",
        "not found",
        "无法回答",
    )
    if any(marker in lowered for marker in incomplete_markers):
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <= 2 and not looks_like_raw_markdown(text):
        return True
    return False


def web_search(query: str) -> dict | list:
    """内部搜索 API."""

    if not query:
        raise ValueError("Query must be a non-empty string")

    payload = {"keywords": [query], "max_results": 3}
    headers = {"Authorization": f"Bearer {settings.api_key}"}

    try:
        response = httpx.post(
            SEARCH_API_URL,
            json=payload,
            headers=headers,
            timeout=httpx.Timeout(20.0, read=20.0),
        )
        response.raise_for_status()
    except httpx.ReadTimeout as exc:
        logger.exception("Web search request timed out")
        raise ToolExecutionError("Web search request timed out") from exc
    except httpx.HTTPError as exc:
        logger.exception("Web search request failed")
        raise ToolExecutionError("Web search request failed") from exc

    data = response.json()
    logger.info(
        "Web search completed",
        extra={"query": query, "status_code": response.status_code},
    )
    return data


def read_page(url: str) -> str:
    """拉取网页并清洗文本。"""

    if not url:
        raise ValueError("URL must be provided")

    headers = {"User-Agent": "super-mind-agent/1.0"}

    try:
        response = httpx.get(url, headers=headers, timeout=httpx.Timeout(20.0, read=20.0))
        response.raise_for_status()
    except httpx.ReadTimeout as exc:
        logger.exception("Read page request timed out", extra={"url": url})
        raise ToolExecutionError("Read page request timed out") from exc
    except httpx.HTTPError as exc:
        logger.exception("Read page request failed", extra={"url": url})
        raise ToolExecutionError("Read page request failed") from exc

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    logger.info("Page content extracted", extra={"url": url, "length": len(cleaned)})
    return cleaned


def ensure_mcp_ready() -> None:
    if not _mcp_command.exists():
        raise ToolExecutionError(
            "未找到 mcp-python-interpreter，请确认已在 .mcp_env 中安装。"
        )
    _mcp_workdir.mkdir(parents=True, exist_ok=True)
    _sync_mcp_workspace()


def call_mcp_python_interpreter(payload: dict[str, Any]) -> dict[str, Any]:
    ensure_mcp_ready()

    requested_timeout: int
    try:
        requested_timeout = int(payload.get("timeout") or 300)
    except (TypeError, ValueError):
        requested_timeout = 300
    requested_timeout = max(requested_timeout, 1)
    process_timeout_seconds = max(420, requested_timeout + 60)

    cmd: list[str] = [
        str(settings.mcp_driver_path),
        str(MCP_BRIDGE_SCRIPT),
        "--process-timeout",
        str(process_timeout_seconds),
    ]
    if _mcp_endpoint:
        cmd += ["--endpoint", _mcp_endpoint]
    else:
        cmd += [
            "--workdir",
            str(_mcp_workdir),
            "--server-command",
            str(_mcp_command),
            "--server-python-path",
            _mcp_python_path,
        ]
    try:
        process = subprocess.run(
            cmd,
            input=json.dumps(payload).encode("utf-8"),
            capture_output=True,
            check=False,
            timeout=process_timeout_seconds + 30,
        )
    except FileNotFoundError as exc:
        raise ToolExecutionError("无法执行 MCP 解释器脚本") from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolExecutionError(
            f"MCP 调用超时（{process_timeout_seconds}s），请检查脚本是否卡住或 MCP 服务是否可用。"
        ) from exc

    if process.returncode != 0:
        stderr_output = process.stderr.decode("utf-8", errors="ignore")
        stdout_output = process.stdout.decode("utf-8", errors="ignore")
        raise ToolExecutionError(
            f"MCP 进程退出异常: {stderr_output or stdout_output or 'unknown error'}"
        )

    stdout_text = process.stdout.decode("utf-8", errors="ignore")
    try:
        response = json.loads(stdout_text)
    except json.JSONDecodeError as exc:
        raise ToolExecutionError(f"MCP 返回异常输出: {stdout_text[:200]}...") from exc

    if not response.get("ok"):
        raise ToolExecutionError(response.get("error", "MCP 执行失败"))
    return response


def build_embedding(text: str) -> List[float]:
    try:
        response = call_with_retries(
            lambda: client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )
        )
    except Exception as exc:
        logger.error("Embedding request failed", extra={"error": str(exc)})
        raise ToolExecutionError("Embedding API 请求失败") from exc

    if not response.data:
        raise ToolExecutionError("Embedding API 返回空结果")
    return response.data[0].embedding


def query_my_notes(query: str, top_k: int = 5) -> dict[str, Any]:
    if not query:
        raise ValueError("query 不能为空")

    index = load_index()
    metadata = load_metadata()

    embedding = np.array(build_embedding(query), dtype="float32").reshape(1, -1)
    distances, indices = index.search(embedding, min(top_k, index.ntotal))

    results: List[dict[str, Any]] = []
    for score, idx in zip(distances[0], indices[0]):
        if idx == -1 or idx >= len(metadata):
            continue
        record = metadata[idx]
        results.append(
            {
                "score": float(score),
                "source_path": record.get("source_path"),
                "chunk_index": record.get("chunk_index"),
                "heading_path": record.get("heading_path"),
                "document_title": record.get("document_title"),
                "chunk_type": record.get("chunk_type"),
                "text": record.get("text"),
            }
        )

    logger.info(
        "query_my_notes completed",
        extra={"query": query, "results": len(results)},
    )
    return {"query": query, "results": results}


def call_with_retries(operation: Callable[[], T]) -> T:
    attempts = max(1, CHAT_API_MAX_RETRIES)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except (APITimeoutError, APIConnectionError) as exc:
            last_error = exc
            logger.warning(
                "Chat API connection error",
                extra={"attempt": attempt, "error": str(exc)},
            )
        except RateLimitError as exc:
            last_error = exc
            logger.warning(
                "Chat API rate limited",
                extra={"attempt": attempt, "error": str(exc)},
            )
        except APIStatusError as exc:
            status_code = getattr(exc, "status_code", None)
            if not is_retryable_status(status_code):
                raise
            last_error = exc
            logger.warning(
                "Chat API status error",
                extra={"attempt": attempt, "status_code": status_code},
            )

        if attempt < attempts:
            backoff = CHAT_API_RETRY_BACKOFF_SECONDS * attempt
            time.sleep(backoff)
            continue
        break

    if last_error is not None:
        raise last_error
    raise RuntimeError("Chat API call failed without exception")


def is_retryable_status(status_code: int | None) -> bool:
    if status_code is None:
        return False
    if status_code in RETRYABLE_STATUS_CODES:
        return True
    return 500 <= status_code < 600


__all__ = [
    "web_search",
    "read_page",
    "ensure_mcp_ready",
    "call_mcp_python_interpreter",
    "build_embedding",
    "query_my_notes",
    "call_with_retries",
    "is_retryable_status",
    "looks_like_raw_markdown",
    "looks_like_interpreter_error",
    "looks_like_incomplete_insight",
]

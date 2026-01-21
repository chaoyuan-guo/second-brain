"""工具函数实现。"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
import shutil
import os
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, TypeVar

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
    is_truthy,
    running_in_container,
    settings,
)
from ..core.logging import app_logger
from ..repositories.notes import load_index, load_metadata
from .clients import chat_client, client, chat_model_name
from .exceptions import ToolExecutionError
from .embedded_interpreter import embedded_python_interpreter

logger = app_logger

_mcp_config_logged = False
_mcp_stdio_warning_logged = False
_embedded_interpreter_logged = False


def _mcp_mode() -> str:
    return "sse" if _mcp_endpoint else "stdio"


def _use_embedded_interpreter() -> bool:
    """判断是否走进程内解释器。

    - 容器部署默认启用（满足单端口/单进程且高频调用更快）。
    - 可通过 MCP_INTERPRETER_BACKEND 覆盖：
      - embedded: 强制进程内
      - mcp/sse/bridge: 强制沿用 MCP bridge（stdio 或 SSE）
    """

    forced = os.getenv("MCP_INTERPRETER_BACKEND")
    if forced:
        lowered = forced.strip().lower()
        if lowered in {"embedded", "inprocess", "in-process"}:
            return True
        if lowered in {"mcp", "sse", "bridge", "stdio"}:
            return False

    return running_in_container()


def _wrap_code_with_system_exit_guard(code: str) -> str:
    """为解释器代码包裹 SystemExit 捕获，避免脚本调用 exit() 杀掉 MCP 服务。"""

    stripped = code.strip("\n")
    if not stripped:
        return code
    if stripped.lstrip().startswith("from __future__"):
        return code

    indented = "\n".join(f"    {line}" for line in stripped.splitlines())
    return (
        "try:\n"
        f"{indented}\n"
        "except SystemExit as exc:\n"
        "    print(\"[SystemExit suppressed]\", exc)\n"
    )


def _summarize_code(code: str, max_chars: int = 160) -> str:
    cleaned = " ".join(code.strip().split())
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[:max_chars]}…"


def _looks_like_sse_disconnect(detail: str) -> bool:
    lowered = detail.lower()
    markers = (
        "error in sse_reader",
        "remoteprotocolerror",
        "incomplete chunked read",
        "peer closed connection",
        "connection refused",
        "connecterror",
    )
    return any(marker in lowered for marker in markers)


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


def read_note_file(
    path: str,
    *,
    offset: int = 0,
    limit_chars: int = 60000,
) -> dict[str, Any]:
    """读取指定笔记文件的文本片段（按字符偏移）。"""

    if not path:
        raise ValueError("Path must be provided")

    base_dir = (settings.base_dir / "data" / "notes" / "my_markdowns").resolve()
    raw_path = Path(path)
    if raw_path.is_absolute():
        resolved = raw_path.resolve()
    else:
        candidate = raw_path
        if str(candidate).startswith("data/notes/my_markdowns"):
            candidate = settings.base_dir / candidate
        else:
            candidate = base_dir / candidate
        resolved = candidate.resolve()

    if resolved != base_dir and base_dir not in resolved.parents:
        raise ToolExecutionError("只允许读取 data/notes/my_markdowns/ 下的文件。")
    if not resolved.exists():
        raise ToolExecutionError(f"文件不存在: {resolved}")
    if resolved.is_dir():
        raise ToolExecutionError(f"路径是目录，无法读取: {resolved}")

    try:
        offset_value = int(offset)
    except (TypeError, ValueError) as exc:
        raise ValueError("offset must be an integer") from exc
    if offset_value < 0:
        raise ValueError("offset must be >= 0")

    try:
        limit_value = int(limit_chars)
    except (TypeError, ValueError) as exc:
        raise ValueError("limit_chars must be an integer") from exc
    if limit_value <= 0:
        raise ValueError("limit_chars must be > 0")
    if limit_value > 60000:
        limit_value = 60000

    text = resolved.read_text(encoding="utf-8")
    total_chars = len(text)
    start = min(offset_value, total_chars)
    end = min(start + limit_value, total_chars)
    content = text[start:end]
    next_offset = end if end < total_chars else None

    logger.info(
        "Note file read",
        extra={
            "path": str(resolved),
            "offset": start,
            "limit_chars": limit_value,
            "total_chars": total_chars,
            "returned_chars": len(content),
        },
    )

    return {
        "source_file": str(resolved),
        "offset": start,
        "limit_chars": limit_value,
        "total_chars": total_chars,
        "next_offset": next_offset,
        "done": end >= total_chars,
        "content": content,
    }


def ensure_mcp_ready() -> None:
    global _mcp_config_logged, _mcp_stdio_warning_logged, _embedded_interpreter_logged

    using_embedded = _use_embedded_interpreter()

    if not _mcp_config_logged:
        logger.info(
            "MCP config loaded",
            extra={
                "mcp_mode": _mcp_mode(),
                "mcp_endpoint": _mcp_endpoint,
                "mcp_command": str(_mcp_command),
                "mcp_driver": str(settings.mcp_driver_path),
                "mcp_workdir": str(_mcp_workdir),
            },
        )
        _mcp_config_logged = True

    if using_embedded and not _embedded_interpreter_logged:
        logger.info(
            "Embedded interpreter enabled",
            extra={"mcp_workdir": str(_mcp_workdir)},
        )
        _embedded_interpreter_logged = True

    if not _mcp_stdio_warning_logged and not _mcp_endpoint and not using_embedded:
        logger.warning(
            "MCP is running in stdio mode; prefer setting MCP_SSE_ENDPOINT to use the persistent server",
            extra={"mcp_workdir": str(_mcp_workdir), "mcp_command": str(_mcp_command)},
        )
        _mcp_stdio_warning_logged = True

    if not using_embedded and not _mcp_command.exists():
        raise ToolExecutionError(
            "未找到 mcp-python-interpreter，请确认已在 .mcp_env 中安装。"
        )
    _mcp_workdir.mkdir(parents=True, exist_ok=True)
    _sync_mcp_workspace()


def call_mcp_python_interpreter(payload: dict[str, Any]) -> dict[str, Any]:
    ensure_mcp_ready()

    original_code = str(payload.get("code") or "")
    payload = dict(payload)
    payload["code"] = _wrap_code_with_system_exit_guard(original_code)

    if _use_embedded_interpreter():
        requested_timeout: int
        try:
            requested_timeout = int(payload.get("timeout") or 300)
        except (TypeError, ValueError):
            requested_timeout = 300
        requested_timeout = max(requested_timeout, 1)

        execution_mode = str(payload.get("execution_mode") or "inline")
        session_id = str(payload.get("session_id") or "default")
        allow_system_access = is_truthy(os.getenv("MCP_ALLOW_SYSTEM_ACCESS"))

        logger.info(
            "Invoking embedded interpreter (timeout=%ss session=%s code=%s)",
            requested_timeout,
            session_id,
            _summarize_code(original_code),
            extra={
                "mcp_mode": "embedded",
                "timeout": requested_timeout,
                "execution_mode": execution_mode,
                "session_id": session_id,
            },
        )

        response = embedded_python_interpreter.run(
            code=str(payload["code"]),
            session_id=session_id,
            execution_mode=execution_mode,
            timeout=requested_timeout,
            workdir=_mcp_workdir,
            allow_system_access=allow_system_access,
        )
        if not response.get("ok"):
            raise ToolExecutionError(response.get("error", "MCP 执行失败"))
        return response

    requested_timeout: int
    try:
        requested_timeout = int(payload.get("timeout") or 300)
    except (TypeError, ValueError):
        requested_timeout = 300
    requested_timeout = max(requested_timeout, 1)
    # Bridge 脚本的 --process-timeout 控制整个 MCP 调用的等待上限。
    # 这里按 payload.timeout + buffer 对齐，避免默认 420s 导致前端长时间卡住。
    process_timeout_seconds = max(60, requested_timeout + 30)

    logger.info(
        "Invoking MCP Python Interpreter (mode=%s endpoint=%s timeout=%ss process_timeout=%ss session=%s code=%s)",
        _mcp_mode(),
        _mcp_endpoint or "",
        requested_timeout,
        process_timeout_seconds,
        payload.get("session_id") or "",
        _summarize_code(original_code),
        extra={
            "mcp_mode": _mcp_mode(),
            "timeout": requested_timeout,
            "process_timeout": process_timeout_seconds,
            "execution_mode": payload.get("execution_mode"),
            "session_id": payload.get("session_id"),
        },
    )

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
        detail = stderr_output or stdout_output or "unknown error"
        detail = detail.strip()
        max_chars = 4000
        if len(detail) > max_chars:
            detail = f"{detail[:max_chars]}\n...[truncated {len(detail) - max_chars} chars]"

        if _mcp_endpoint and _looks_like_sse_disconnect(detail):
            logger.warning(
                "MCP SSE disconnected; the interpreter service may have crashed or restarted",
                extra={"mcp_endpoint": _mcp_endpoint},
            )
        raise ToolExecutionError(f"MCP 进程退出异常: {detail}")

    stdout_text = process.stdout.decode("utf-8", errors="ignore")
    try:
        response = json.loads(stdout_text)
    except json.JSONDecodeError as exc:
        raise ToolExecutionError(f"MCP 返回异常输出: {stdout_text[:200]}...") from exc

    if not response.get("ok"):
        raise ToolExecutionError(response.get("error", "MCP 执行失败"))

    logger.info(
        "MCP call succeeded",
        extra={
            "mcp_mode": _mcp_mode(),
            "timeout": requested_timeout,
            "process_timeout": process_timeout_seconds,
            "content_length": len(str(response.get("content", ""))),
        },
    )
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


async def async_call_with_retries(operation: Callable[[], Awaitable[T]]) -> T:
    attempts = max(1, CHAT_API_MAX_RETRIES)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
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
            await asyncio.sleep(backoff)
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
    "read_note_file",
    "ensure_mcp_ready",
    "call_mcp_python_interpreter",
    "build_embedding",
    "query_my_notes",
    "call_with_retries",
    "async_call_with_retries",
    "is_retryable_status",
    "looks_like_raw_markdown",
    "looks_like_interpreter_error",
    "looks_like_incomplete_insight",
]

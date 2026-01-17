"""聊天与工具调用核心逻辑。"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from openai import APIError

from ..core.config import (
    CHAT_API_MAX_RETRIES,
    CHAT_API_RETRY_BACKOFF_SECONDS,
    MAX_TOOL_TURNS,
    MAX_TOOL_OUTPUT_CHARS,
    SYSTEM_PROMPT,
    settings,
)
from ..core.logging import app_logger, tool_output_logger
from ..models.schemas import (
    ChatRequest,
    ChatResponse,
    ChatTitleResponse,
    ConversationMessage,
    ToolCall,
    ToolCallFunction,
)
from .clients import chat_client, chat_model_name
from .exceptions import ToolExecutionError
from .tools import (
    call_mcp_python_interpreter,
    call_with_retries,
    ensure_mcp_ready,
    is_retryable_status,
    looks_like_incomplete_insight,
    looks_like_interpreter_error,
    looks_like_raw_markdown,
    query_my_notes,
    read_page,
    web_search,
)

logger = app_logger


def _format_tool_status_message(
    *,
    tool_name: str,
    tool_index: int,
    tool_count: int,
    arguments: dict[str, Any],
    stage: str,
) -> str:
    """为前端流式状态栏生成可解释的工具调用文案。"""

    tool_label = tool_name
    if tool_name == "query_my_notes":
        tool_label = "笔记检索"
    elif tool_name == "web_search":
        tool_label = "联网搜索"
    elif tool_name == "read_page":
        tool_label = "网页读取"
    elif tool_name == "run_code_interpreter":
        tool_label = "代码执行"

    prefix = f"[{tool_index}] {tool_label}（第 {tool_count} 次）"

    if tool_name in {"query_my_notes", "web_search"}:
        query = str(arguments.get("query") or "").strip()
        query_hint = query[:60] + ("…" if len(query) > 60 else "")
        if query_hint:
            prefix += f"：{query_hint}"

    if tool_name == "read_page":
        url = str(arguments.get("url") or "").strip()
        if url:
            prefix += f"：{url}"

    if tool_name == "run_code_interpreter":
        mode = str(arguments.get("execution_mode") or "inline")
        timeout = arguments.get("timeout")
        details: list[str] = []
        if mode:
            details.append(f"mode={mode}")
        if isinstance(timeout, int):
            details.append(f"timeout={timeout}s")
        if details:
            prefix += f"（{' '.join(details)}）"

    if stage == "start":
        return f"正在调用 {prefix}…"
    if stage == "end":
        return f"已完成 {prefix}"
    if stage == "error":
        return f"调用失败 {prefix}"
    return prefix


def _is_tool_error_output(output: str) -> bool:
    """判断工具输出是否为 error JSON。"""

    cleaned = output.strip()
    if not cleaned:
        return False
    if not cleaned.startswith("{"):
        return False
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict) and "error" in payload


def _truncate_tool_text(text: str, *, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> tuple[str, bool]:
    """截断过长的工具输出，避免阻塞后续模型调用。"""

    if len(text) <= max_chars:
        return text, False
    truncated = text[:max_chars]
    return f"{truncated}\n\n...[truncated {len(text) - max_chars} chars]", True


def _dumps_tool_payload(payload: dict[str, Any]) -> str:
    """将工具结果序列化为受控大小的 JSON 字符串。"""

    content = json.dumps(payload, ensure_ascii=False)
    if len(content) <= MAX_TOOL_OUTPUT_CHARS:
        return content

    results = payload.get("results")
    if isinstance(results, list):
        payload = dict(payload)
        slimmed: list[dict[str, Any]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            copied = dict(item)
            text_value = copied.get("text")
            if isinstance(text_value, str):
                copied["text"], _ = _truncate_tool_text(text_value, max_chars=1200)
            slimmed.append(copied)
            if len(slimmed) >= 5:
                break
        payload["results"] = slimmed
        payload["truncated"] = True
        content = json.dumps(payload, ensure_ascii=False)

    if len(content) <= MAX_TOOL_OUTPUT_CHARS:
        return content

    content, _ = _truncate_tool_text(content)
    return content


def _shrink_mcp_response_for_model(response: dict[str, Any]) -> dict[str, Any]:
    """压缩 MCP 解释器响应，避免将超大 stdout 送入模型上下文。

    - 完整响应仍会写入 tool_output.log。
    - 送入模型的 tool role 消息仅保留必要字段与截断后的 content。
    """

    slim: dict[str, Any] = {
        "ok": response.get("ok"),
    }

    content = response.get("content")
    if isinstance(content, str):
        max_chars = 6000
        if len(content) <= max_chars:
            slim["content"] = content
            slim["content_truncated"] = False
        else:
            head = content[:1000]
            tail = content[-4000:]
            slim["content"] = (
                f"{head}\n\n...[truncated {len(content) - (len(head) + len(tail))} chars]...\n\n{tail}"
            )
            slim["content_truncated"] = True
        slim["content_length"] = len(content)

    # raw 字段通常非常大（包含重复的 result），不发送给模型。
    if "raw" in response:
        slim["raw_omitted"] = True
    return slim


WEB_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Use this tool to search internal knowledge for up-to-date information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language question or keywords to search for.",
                }
            },
            "required": ["query"],
        },
    },
}


READ_PAGE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_page",
        "description": "Fetch the textual content of a web page to extract detailed information.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Fully qualified URL to retrieve and summarize.",
                }
            },
            "required": ["url"],
        },
    },
}


QUERY_MY_NOTES_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_my_notes",
        "description": "在个人 Markdown 向量索引中执行语义搜索，返回最相关的笔记片段。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "针对笔记内容的自然语言查询或关键词。",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                    "description": "返回的匹配数量。",
                },
            },
            "required": ["query"],
        },
    },
}


MCP_CODE_INTERPRETER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_code_interpreter",
        "description": "在 MCP Python Interpreter 沙盒中运行 Python 代码。",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "要执行的 Python 代码"},
                "execution_mode": {
                    "type": "string",
                    "enum": ["inline", "subprocess"],
                    "default": "inline",
                },
                "environment": {
                    "type": "string",
                    "description": "subprocess 模式下的环境名称",
                },
                "session_id": {"type": "string", "description": "复用的 session ID"},
                "timeout": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 600,
                    "default": 300,
                },
            },
            "required": ["code"],
        },
    },
}


AVAILABLE_TOOLS = [
    QUERY_MY_NOTES_TOOL_SCHEMA,
    WEB_SEARCH_TOOL_SCHEMA,
    READ_PAGE_TOOL_SCHEMA,
    MCP_CODE_INTERPRETER_SCHEMA,
]


VALID_CHAT_ROLES = {"system", "user", "assistant", "tool", "developer"}
TITLE_PROMPT = (
    "你是会话命名助手。阅读最近一次对话的用户与助手消息，"
    "生成 4-16 个字符的中文标题，突出主要意图或主题，避免口头禅与标点，"
    "不要包含引号或编号，只返回标题本身。"
)
TITLE_SOURCE_ROLES = {"user": "用户", "assistant": "助手"}
TITLE_MESSAGE_LIMIT = 8


def _normalize_title(raw: str) -> str:
    """清洗模型返回的标题，避免空值或冗余标点。"""

    if not raw:
        return "新的对话"
    candidate = raw.strip().strip('"“”')
    first_line = candidate.splitlines()[0]
    if ":" in first_line:
        _, remainder = first_line.split(":", 1)
        if remainder.strip():
            first_line = remainder.strip()
    normalized = first_line.strip()
    if not normalized:
        return "新的对话"
    return normalized[:32]


def _build_title_context(messages: List[dict[str, Any]]) -> str:
    """将最近的用户/助手消息整理为提示文本。"""

    relevant: List[str] = []
    for message in messages[-(TITLE_MESSAGE_LIMIT * 2) :]:
        role = message.get("role")
        content = message.get("content")
        if role not in TITLE_SOURCE_ROLES or not isinstance(content, str):
            continue
        sanitized = " ".join(content.split())
        if not sanitized:
            continue
        relevant.append(f"{TITLE_SOURCE_ROLES[role]}: {sanitized}")
        if len(relevant) >= TITLE_MESSAGE_LIMIT:
            break
    return "\n".join(relevant)


def _resolve_messages(payload: ChatRequest) -> List[dict[str, Any]]:
    prepared: List[dict[str, Any]] = []
    if payload.messages:
        for message in payload.messages:
            prepared_message: dict[str, Any] = {
                "role": message.role,
                "content": message.content,
            }
            if message.name:
                prepared_message["name"] = message.name
            if message.tool_call_id:
                prepared_message["tool_call_id"] = message.tool_call_id
            prepared.append(prepared_message)
    if payload.user_message:
        prepared.append({"role": "user", "content": payload.user_message})
    if not prepared:
        raise HTTPException(status_code=400, detail="缺少用户消息")
    return prepared


def _tool_calls_to_models(tool_calls: Optional[List[dict[str, Any]]]) -> Optional[List[ToolCall]]:
    if not tool_calls:
        return None

    formatted: List[ToolCall] = []
    for call in tool_calls:
        function = call.get("function") or {}
        formatted.append(
            ToolCall(
                id=call.get("id") or "",
                type=call.get("type") or "function",
                function=ToolCallFunction(
                    name=function.get("name") or "",
                    arguments=function.get("arguments") or "",
                ),
            )
        )
    return formatted


def _normalize_client_messages(
    payload: ChatRequest, *, require_last_user: bool = True
) -> List[dict[str, Any]]:
    messages = _resolve_messages(payload)
    for message in messages:
        if message.get("role") not in VALID_CHAT_ROLES:
            raise HTTPException(status_code=400, detail=f"无效角色: {message.get('role')}")
    if require_last_user and messages[-1]["role"] != "user":
        raise HTTPException(status_code=400, detail="最后一条消息必须来自用户")
    return messages


async def generate_title(payload: ChatRequest) -> ChatTitleResponse:
    messages = _normalize_client_messages(payload, require_last_user=False)
    context = _build_title_context(messages)
    if not context:
        return ChatTitleResponse(title="新的对话")

    user_prompt = f"基于以下对话片段生成标题：\n{context}"

    try:
        completion_kwargs = {
            "model": chat_model_name,
            "messages": [
                {"role": "system", "content": TITLE_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        if settings.use_azure_chat:
            completion_kwargs["max_completion_tokens"] = 32
        else:
            completion_kwargs["max_tokens"] = 32

        completion = await asyncio.to_thread(
            call_with_retries,
            lambda: chat_client.chat.completions.create(**completion_kwargs),
        )
    except Exception:
        logger.exception("Failed to generate session title")
        return ChatTitleResponse(title="新的对话")

    if not completion.choices:
        return ChatTitleResponse(title="新的对话")

    choice = completion.choices[0]
    raw_title = choice.message.content if getattr(choice, "message", None) else ""
    return ChatTitleResponse(title=_normalize_title(raw_title or ""))


def _normalize_tool_calls(tool_calls_raw: Optional[List[Any]]) -> Optional[List[dict[str, Any]]]:
    if not tool_calls_raw:
        return None

    normalized: List[dict[str, Any]] = []
    for call in tool_calls_raw:
        function = getattr(call, "function", None)
        normalized.append(
            {
                "id": getattr(call, "id", "") or "",
                "type": getattr(call, "type", "function") or "function",
                "function": {
                    "name": getattr(function, "name", "") if function else "",
                    "arguments": getattr(function, "arguments", "") if function else "",
                },
            }
        )
    return normalized


def _safe_load_tool_arguments(arguments_raw: str) -> dict[str, Any]:
    if not arguments_raw:
        raise ValueError("tool arguments empty")
    cleaned = arguments_raw.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        parsed, offset = decoder.raw_decode(cleaned)
        remainder = cleaned[offset:].strip()
        logger.warning(
            "Tool arguments contained trailing data; truncated",
            extra={
                "arguments_prefix": cleaned[:200],
                "trailing_prefix": remainder[:200],
            },
        )
        return parsed


def _apply_system_prompt(messages: List[dict[str, Any]]) -> List[dict[str, Any]]:
    for message in messages:
        if message.get("role") == "system":
            message["content"] = SYSTEM_PROMPT
            break
    else:
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return messages


def _request_completion(
    messages: List[dict[str, Any]],
    stream_callback: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    messages = _apply_system_prompt(messages)

    started_at = time.perf_counter()
    logger.info(
        "Calling chat completion",
        extra={"stream": bool(stream_callback), "message_count": len(messages)},
    )

    completion = call_with_retries(
        lambda: chat_client.chat.completions.create(
            model=chat_model_name,
            messages=messages,
            tools=AVAILABLE_TOOLS,
        )
    )

    logger.info(
        "Chat completion returned",
        extra={
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "stream": bool(stream_callback),
        },
    )

    if not completion.choices:
        logger.error("Chat completion returned no choices")
        raise HTTPException(status_code=502, detail="No response from assistant")

    message = completion.choices[0].message
    role = getattr(message, "role", "assistant") or "assistant"
    content = message.content or ""
    tool_calls = _normalize_tool_calls(getattr(message, "tool_calls", None))

    if stream_callback and content:
        # 避免上游 stream=True 迭代器偶发卡死：改为本地切片推送。
        # 这仍然能让前端收到 delta 更新，同时所有网络超时由非流式请求兜底。
        chunk_size = 400
        for start in range(0, len(content), chunk_size):
            stream_callback(content[start : start + chunk_size])

    return {
        "role": role,
        "content": content,
        "tool_calls": tool_calls,
        # 非流式 completion 返回的 tool_calls 已经完整，可以直接执行。
        "tool_call_finished": bool(tool_calls),
    }


def _request_streaming_completion(
    messages: List[dict[str, Any]], stream_callback: Callable[[str], None]
) -> dict[str, Any]:
    last_exception: Optional[Exception] = None
    fallback_due_to_tool = False
    stream_warning_emitted = False
    for attempt in range(1, CHAT_API_MAX_RETRIES + 1):
        tool_call_in_progress = False
        try:
            stream = call_with_retries(
                lambda: chat_client.chat.completions.create(
                    model=chat_model_name,
                    messages=messages,
                    tools=AVAILABLE_TOOLS,
                    stream=True,
                )
            )

            content_parts: List[str] = []
            tool_calls: Dict[int, dict[str, Any]] = {}
            role = "assistant"
            tool_call_finished = False

            pending_tool_call_ids: List[str] = []
            for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta

                if delta.role:
                    role = delta.role

                if delta.content:
                    stream_callback(delta.content)
                    content_parts.append(delta.content)

                if delta.tool_calls:
                    tool_call_in_progress = True
                    for tool_delta in delta.tool_calls:
                        entry = tool_calls.setdefault(
                            tool_delta.index,
                            {
                                "id": tool_delta.id or "",
                                "type": tool_delta.type or "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        if tool_delta.id:
                            entry["id"] = tool_delta.id
                            if tool_delta.id not in pending_tool_call_ids:
                                pending_tool_call_ids.append(tool_delta.id)
                        if tool_delta.type:
                            entry["type"] = tool_delta.type
                        if tool_delta.function:
                            if tool_delta.function.name:
                                entry["function"]["name"] = tool_delta.function.name
                            if tool_delta.function.arguments:
                                entry["function"]["arguments"] += tool_delta.function.arguments

                if choice.finish_reason == "tool_calls":
                    tool_call_finished = True
                    pending_tool_call_ids.clear()
                    break

            if pending_tool_call_ids and not tool_call_finished:
                raise APIError(
                    "Missing tool call responses",
                    None,
                    body={"missing_tool_calls": pending_tool_call_ids},
                )

            tool_calls_list = (
                [tool_calls[index] for index in sorted(tool_calls)] if tool_calls else None
            )

            content = "".join(content_parts)
            return {
                "role": role,
                "content": content,
                "tool_calls": tool_calls_list,
                "tool_call_finished": tool_call_finished,
            }
        except (httpx.HTTPError, httpx.TimeoutException) as exc:  # type: ignore[attr-defined]
            last_exception = exc
            if tool_call_in_progress:
                fallback_due_to_tool = True
                break
            log_kwargs = {"extra": {"attempt": attempt}}
            if not stream_warning_emitted:
                logger.warning(
                    "Streaming response interrupted, retrying (attempt=%s)",
                    attempt,
                    exc_info=True,
                    **log_kwargs,
                )
                stream_warning_emitted = True
            else:
                logger.debug(
                    "Streaming response interrupted, retrying (attempt=%s)",
                    attempt,
                    **log_kwargs,
                )
            time.sleep(CHAT_API_RETRY_BACKOFF_SECONDS * attempt)
        except APIError as exc:
            last_exception = exc
            if tool_call_in_progress:
                fallback_due_to_tool = True
                break
            log_kwargs = {"extra": {"attempt": attempt}}
            if not stream_warning_emitted:
                logger.warning(
                    "Streaming API error, retrying (attempt=%s)",
                    attempt,
                    exc_info=True,
                    **log_kwargs,
                )
                stream_warning_emitted = True
            else:
                logger.debug(
                    "Streaming API error, retrying (attempt=%s)",
                    attempt,
                    **log_kwargs,
                )
            time.sleep(CHAT_API_RETRY_BACKOFF_SECONDS * attempt)

    if fallback_due_to_tool:
        logger.warning(
            "Streaming aborted mid tool call, falling back to non-streaming",
            extra={"error": str(last_exception) if last_exception else None},
        )
    else:
        logger.error(
            "Streaming failed after retries, falling back to non-streaming",
            extra={"error": str(last_exception) if last_exception else None},
        )
    completion = call_with_retries(
        lambda: chat_client.chat.completions.create(
            model=chat_model_name,
            messages=messages,
            tools=AVAILABLE_TOOLS,
        )
    )

    if not completion.choices:
        raise HTTPException(status_code=502, detail="No response from assistant")

    message = completion.choices[0].message
    fallback_content = message.content or ""
    if fallback_content:
        stream_callback(fallback_content)
    return {
        "role": getattr(message, "role", "assistant") or "assistant",
        "content": fallback_content,
        "tool_calls": _normalize_tool_calls(getattr(message, "tool_calls", None)),
        "tool_call_finished": False,
    }


def run_chat_conversation(
    payload: ChatRequest,
    stream_callback: Optional[Callable[[str], None]] = None,
    event_callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> ChatResponse:
    messages = _normalize_client_messages(payload)
    tool_invocations = 0
    query_tool_attempts = 0
    code_interpreter_used = False
    agentic_hint_inserted = False

    if event_callback:
        try:
            event_callback(
                {
                    "type": "status",
                    "phase": "thinking",
                    "message": "正在思考…",
                    "ts": time.time(),
                }
            )
        except Exception:
            logger.debug("Failed to emit status event", exc_info=True)

    for turn in range(MAX_TOOL_TURNS + 1):
        assistant_message = _request_completion(messages, stream_callback=stream_callback)
        tool_calls = assistant_message.get("tool_calls")
        tool_call_finished = assistant_message.get("tool_call_finished")

        if not tool_calls:
            content = assistant_message.get("content")
            if content:
                logger.info("[Agent] 最终答案：'%s'", content)
                return ChatResponse(
                    response=content,
                    tool_calls=_tool_calls_to_models(tool_calls),
                )
            logger.warning(
                "Assistant response without content or tool calls",
                extra={"turn": turn + 1},
            )
            continue

        messages.append(
            {
                "role": assistant_message.get("role", "assistant"),
                "content": assistant_message.get("content", ""),
                "tool_calls": tool_calls,
            }
        )

        if not tool_call_finished:
            continue

        for call in tool_calls:
            tool_call_id = call.get("id")
            if not tool_call_id:
                logger.error(
                    "Assistant tool call missing id",
                    extra={"tool_call": call},
                )
                continue

            function_block = call.get("function") or {}
            tool_name = function_block.get("name")
            arguments_raw = function_block.get("arguments", "")
            try:
                arguments = _safe_load_tool_arguments(arguments_raw)
            except ValueError:
                logger.exception("Invalid tool arguments", extra={"tool": tool_name})
                continue

            tool_count = tool_invocations + 1
            if event_callback and tool_name:
                try:
                    event_callback(
                        {
                            "type": "tool",
                            "stage": "start",
                            "tool_name": tool_name,
                            "tool_call_id": tool_call_id,
                            "tool_count": tool_count,
                            "message": _format_tool_status_message(
                                tool_name=tool_name,
                                tool_index=tool_count,
                                tool_count=tool_count,
                                arguments=arguments,
                                stage="start",
                            ),
                            "ts": time.time(),
                        }
                    )
                except Exception:
                    logger.debug("Failed to emit tool start event", exc_info=True)

            post_tool_messages: List[dict[str, Any]] = []

            if tool_name == "query_my_notes":
                try:
                    result = query_my_notes(
                        query=arguments.get("query", ""),
                        top_k=int(arguments.get("top_k", 5)),
                    )
                except Exception as exc:
                    result = {"error": str(exc)}
                    if event_callback:
                        try:
                            event_callback(
                                {
                                    "type": "tool",
                                    "stage": "error",
                                    "tool_name": tool_name,
                                    "tool_call_id": tool_call_id,
                                    "tool_count": tool_count,
                                    "message": _format_tool_status_message(
                                        tool_name=tool_name,
                                        tool_index=tool_count,
                                        tool_count=tool_count,
                                        arguments=arguments,
                                        stage="error",
                                    ),
                                    "error": str(exc),
                                    "ts": time.time(),
                                }
                            )
                        except Exception:
                            logger.debug("Failed to emit tool error event", exc_info=True)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": _dumps_tool_payload(result),
                    }
                )
                query_tool_attempts += 1
                tool_invocations += 1
                if event_callback:
                    try:
                        event_callback(
                            {
                                "type": "tool",
                                "stage": "end",
                                "tool_name": tool_name,
                                "tool_call_id": tool_call_id,
                                "tool_count": tool_count,
                                "message": _format_tool_status_message(
                                    tool_name=tool_name,
                                    tool_index=tool_count,
                                    tool_count=tool_count,
                                    arguments=arguments,
                                    stage="end",
                                ),
                                "ts": time.time(),
                            }
                        )
                    except Exception:
                        logger.debug("Failed to emit tool end event", exc_info=True)
                continue

            if tool_name == "web_search":
                query = arguments.get("query")
                if not query:
                    logger.warning("web_search called without query")
                    continue
                try:
                    result = web_search(query)
                except Exception as exc:
                    result = {"error": str(exc)}
                    if event_callback:
                        try:
                            event_callback(
                                {
                                    "type": "tool",
                                    "stage": "error",
                                    "tool_name": tool_name,
                                    "tool_call_id": tool_call_id,
                                    "tool_count": tool_count,
                                    "message": _format_tool_status_message(
                                        tool_name=tool_name,
                                        tool_index=tool_count,
                                        tool_count=tool_count,
                                        arguments=arguments,
                                        stage="error",
                                    ),
                                    "error": str(exc),
                                    "ts": time.time(),
                                }
                            )
                        except Exception:
                            logger.debug("Failed to emit tool error event", exc_info=True)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
                tool_invocations += 1
                if event_callback:
                    try:
                        event_callback(
                            {
                                "type": "tool",
                                "stage": "end",
                                "tool_name": tool_name,
                                "tool_call_id": tool_call_id,
                                "tool_count": tool_count,
                                "message": _format_tool_status_message(
                                    tool_name=tool_name,
                                    tool_index=tool_count,
                                    tool_count=tool_count,
                                    arguments=arguments,
                                    stage="end",
                                ),
                                "ts": time.time(),
                            }
                        )
                    except Exception:
                        logger.debug("Failed to emit tool end event", exc_info=True)
                continue

            if tool_name == "read_page":
                url = arguments.get("url")
                tool_started_at = time.perf_counter()
                try:
                    content = read_page(url)
                    snippet = content[:200]
                    logger.info(
                        "[System] read_page 完成",
                        extra={
                            "tool_call_id": tool_call_id,
                            "tool_latency_ms": round(
                                (time.perf_counter() - tool_started_at) * 1000, 2
                            ),
                            "snippet": snippet,
                        },
                    )
                    tool_output = content
                except ToolExecutionError as exc:
                    tool_output = json.dumps({"error": str(exc)}, ensure_ascii=False)
                    logger.error(
                        "[System] read_page 异常",
                        extra={
                            "tool_call_id": tool_call_id,
                            "error": tool_output,
                            "tool_latency_ms": round(
                                (time.perf_counter() - tool_started_at) * 1000, 2
                            ),
                        },
                    )
                else:
                    tool_output_logger.info(
                        json.dumps(
                            {
                                "tool_call_id": tool_call_id,
                                "tool": tool_name,
                                "output": content,
                            },
                            ensure_ascii=False,
                        )
                    )
                if event_callback:
                    try:
                        stage = "end" if not _is_tool_error_output(tool_output) else "error"
                        event_callback(
                            {
                                "type": "tool",
                                "stage": stage,
                                "tool_name": tool_name,
                                "tool_call_id": tool_call_id,
                                "tool_count": tool_count,
                                "latency_ms": round(
                                    (time.perf_counter() - tool_started_at) * 1000, 2
                                ),
                                "message": _format_tool_status_message(
                                    tool_name=tool_name,
                                    tool_index=tool_count,
                                    tool_count=tool_count,
                                    arguments=arguments,
                                    stage=stage,
                                ),
                                "ts": time.time(),
                            }
                        )
                    except Exception:
                        logger.debug("Failed to emit tool end event", exc_info=True)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_output,
                    }
                )
                tool_invocations += 1
                continue

            if tool_name == "run_code_interpreter":
                code = arguments.get("code")
                execution_mode = arguments.get("execution_mode", "inline")
                environment = arguments.get("environment")
                session_id = arguments.get("session_id")
                timeout = arguments.get("timeout", 300)
                tool_started_at = time.perf_counter()
                tool_error_detail: str | None = None
                logger.info(
                    "[Agent] 决定调用工具：'run_code_interpreter'",
                    extra={
                        "mode": execution_mode,
                        "session": session_id,
                        "tool_call_id": tool_call_id,
                    },
                )
                try:
                    payload = {
                        "code": code,
                        "execution_mode": execution_mode,
                        "environment": environment,
                        "session_id": session_id,
                        "timeout": timeout,
                    }
                    cleaned_payload = {k: v for k, v in payload.items() if v is not None}
                    response = call_mcp_python_interpreter(cleaned_payload)
                    model_payload = _shrink_mcp_response_for_model(response)
                    tool_output = json.dumps(model_payload, ensure_ascii=False)
                    interpreter_output = str(model_payload.get("content", ""))
                    if looks_like_raw_markdown(interpreter_output):
                        warning_message = (
                            "系统提示：你刚才的 run_code_interpreter 输出只有 Markdown 原文，"
                            "没有任何结构化的统计结果。请重新运行代码解释器，"
                            "编写脚本读取 my_markdowns/ 中的原始数据并输出 JSON/表格等结构化结果，"
                            "再根据输出撰写答案。"
                        )
                        post_tool_messages.append(
                            {"role": "system", "content": warning_message}
                        )
                    elif not interpreter_output.strip() or looks_like_interpreter_error(
                        interpreter_output
                    ):
                        warning_message = (
                            "系统提示：run_code_interpreter 刚才返回了空输出或 Python 错误。"
                            "请先在回复中列出修复计划，再提供新的脚本，并务必打印 JSON/表格等结构化结果"
                            "（包含字段名、统计值、来源文件），确保异常原因被说明。"
                        )
                        post_tool_messages.append(
                            {"role": "system", "content": warning_message}
                        )
                    elif looks_like_incomplete_insight(interpreter_output):
                        warning_message = (
                            "系统提示：代码解释器的输出缺少回答 Query 所需的关键字段或失败信息模糊。"
                            "请回到原始文件片段，总结你观察到的结构，并在计划里写清“如何根据该结构获取答案”，再改写脚本。"
                        )
                        post_tool_messages.append(
                            {"role": "system", "content": warning_message}
                        )
                except ToolExecutionError as exc:
                    tool_error_detail = str(exc)
                    tool_output = json.dumps({"error": str(exc)}, ensure_ascii=False)
                    logger.error(
                        "[System] MCP 工具异常: %s",
                        tool_error_detail,
                        extra={
                            "tool_call_id": tool_call_id,
                            "error": tool_output,
                            "tool_latency_ms": round(
                                (time.perf_counter() - tool_started_at) * 1000, 2
                            ),
                        },
                    )
                    tool_output_logger.info(
                        json.dumps(
                            {
                                "tool_call_id": tool_call_id,
                                "tool": tool_name,
                                "error": tool_error_detail,
                            },
                            ensure_ascii=False,
                        )
                    )
                else:
                    tool_output_logger.info(
                        json.dumps(
                            {
                                "tool_call_id": tool_call_id,
                                "tool": tool_name,
                                "output": response,
                            },
                            ensure_ascii=False,
                        )
                    )
                    logger.info(
                        "[System] run_code_interpreter 完成",
                        extra={
                            "tool_call_id": tool_call_id,
                            "tool_latency_ms": round(
                                (time.perf_counter() - tool_started_at) * 1000, 2
                            ),
                        },
                    )
                code_interpreter_used = True

                if event_callback:
                    try:
                        stage = "end" if not _is_tool_error_output(tool_output) else "error"
                        event_callback(
                            {
                                "type": "tool",
                                "stage": stage,
                                "tool_name": tool_name,
                                "tool_call_id": tool_call_id,
                                "tool_count": tool_count,
                                "latency_ms": round(
                                    (time.perf_counter() - tool_started_at) * 1000, 2
                                ),
                                "message": _format_tool_status_message(
                                    tool_name=tool_name,
                                    tool_index=tool_count,
                                    tool_count=tool_count,
                                    arguments=arguments,
                                    stage=stage,
                                ),
                                "error": tool_error_detail if stage == "error" else None,
                                "ts": time.time(),
                            }
                        )
                    except Exception:
                        logger.debug("Failed to emit tool end event", exc_info=True)

                tool_output, was_truncated = _truncate_tool_text(tool_output)
                if was_truncated:
                    logger.warning(
                        "Truncated tool output before sending to model",
                        extra={"tool": tool_name, "tool_call_id": tool_call_id},
                    )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": tool_output,
                    }
                )
                tool_invocations += 1
                if post_tool_messages:
                    messages.extend(post_tool_messages)
                continue

            logger.error("Unknown tool requested", extra={"tool_name": tool_name})
            tool_output = json.dumps(
                {"error": f"Unknown tool: {tool_name}"}, ensure_ascii=False
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": tool_output,
                }
            )
            tool_invocations += 1

            if event_callback and tool_name:
                try:
                    event_callback(
                        {
                            "type": "tool",
                            "stage": "error",
                            "tool_name": tool_name,
                            "tool_call_id": tool_call_id,
                            "tool_count": tool_count,
                            "message": _format_tool_status_message(
                                tool_name=tool_name,
                                tool_index=tool_count,
                                tool_count=tool_count,
                                arguments=arguments,
                                stage="error",
                            ),
                            "error": f"Unknown tool: {tool_name}",
                            "ts": time.time(),
                        }
                    )
                except Exception:
                    logger.debug("Failed to emit tool error event", exc_info=True)

        if (
            not agentic_hint_inserted
            and not code_interpreter_used
            and query_tool_attempts >= 2
        ):
            hint_message = (
                "系统提示：你已经多次调用 query_my_notes 仍然无法回答。"
                "请立即编写 Python 计划并调用 run_code_interpreter："
                "1) 用自然语言列出步骤；2) 给出可执行脚本，读取 my_markdowns/ 中的原始数据；"
                "3) 根据当前问题推导所需字段/统计指标并输出结构化结果（JSON/表格等）；"
                "4) 基于脚本输出撰写答案并引用来源，切勿只打印 Markdown。"
            )
            messages.append({"role": "system", "content": hint_message})
            agentic_hint_inserted = True
            logger.info(
                "Inserted agentic hint after repeated query_my_notes",
                extra={"query_attempts": query_tool_attempts},
            )

        if event_callback:
            try:
                event_callback(
                    {
                        "type": "status",
                        "phase": "synthesize",
                        "message": "正在整理工具结果并继续生成…",
                        "tool_invocations": tool_invocations,
                        "ts": time.time(),
                    }
                )
            except Exception:
                logger.debug("Failed to emit synthesize status", exc_info=True)

    logger.error("Agent loop terminated without final answer")
    return ChatResponse(response="抱歉，暂时无法得出答案，请稍后重试。")


async def execute_chat(
    payload: ChatRequest,
    stream_callback: Optional[Callable[[str], None]] = None,
    event_callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> ChatResponse:
    return await asyncio.to_thread(
        run_chat_conversation,
        payload,
        stream_callback,
        event_callback,
    )


async def stream_chat_response(payload: ChatRequest, *, want_ndjson: bool = False) -> StreamingResponse:
    async def streamer():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        chat_response: Optional[ChatResponse] = None

        def _queue_put(value: Optional[str]) -> None:
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None

            if running_loop is loop:
                queue.put_nowait(value)
                return

            asyncio.run_coroutine_threadsafe(queue.put(value), loop)

        def _enqueue_plain(chunk: str) -> None:
            _queue_put(chunk)

        def _enqueue_json(event: dict[str, Any]) -> None:
            try:
                payload_json = json.dumps(event, ensure_ascii=False)
            except TypeError:
                payload_json = json.dumps(
                    {"type": "status", "message": str(event)}, ensure_ascii=False
                )
            _queue_put(payload_json + "\n")

        def _enqueue_delta(chunk: str) -> None:
            _enqueue_json({"type": "delta", "delta": chunk, "ts": time.time()})

        async def _run_chat() -> None:
            nonlocal chat_response
            try:
                chat_response = await execute_chat(
                    payload,
                    stream_callback=_enqueue_delta if want_ndjson else _enqueue_plain,
                    event_callback=_enqueue_json if want_ndjson else None,
                )
            finally:
                if want_ndjson:
                    _enqueue_json({"type": "done", "ts": time.time()})
                await queue.put(None)

        task = asyncio.create_task(_run_chat())
        streamed = False
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            streamed = True
            yield chunk
        await task
        if not streamed and chat_response and chat_response.response:
            if want_ndjson:
                yield json.dumps(
                    {"type": "delta", "delta": chat_response.response, "ts": time.time()},
                    ensure_ascii=False,
                )
                yield "\n"
                yield json.dumps({"type": "done", "ts": time.time()}, ensure_ascii=False)
                yield "\n"
            else:
                yield chat_response.response

    media_type = "application/x-ndjson" if want_ndjson else "text/plain; charset=utf-8"
    return StreamingResponse(streamer(), media_type=media_type)


__all__ = [
    "AVAILABLE_TOOLS",
    "WEB_SEARCH_TOOL_SCHEMA",
    "READ_PAGE_TOOL_SCHEMA",
    "QUERY_MY_NOTES_TOOL_SCHEMA",
    "MCP_CODE_INTERPRETER_SCHEMA",
    "run_chat_conversation",
    "execute_chat",
    "stream_chat_response",
    "generate_title",
]

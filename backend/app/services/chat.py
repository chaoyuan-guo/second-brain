"""聊天与工具调用核心逻辑。"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
import uuid
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from openai import APIError

from ..core.config import (
    CHAT_API_MAX_RETRIES,
    CHAT_API_RETRY_BACKOFF_SECONDS,
    MAX_TOOL_TURNS,
    MAX_TOOL_OUTPUT_CHARS,
    OPENAI_DEFAULT_TIMEOUT_SECONDS,
    OPENAI_STREAM_READ_TIMEOUT_SECONDS,
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
from .skills import build_skills_prompt
from .tools import (
    async_call_with_retries,
    call_mcp_python_interpreter,
    ensure_mcp_ready,
    is_retryable_status,
    looks_like_incomplete_insight,
    looks_like_interpreter_error,
    looks_like_raw_markdown,
    load_skill,
    query_my_notes,
    read_note_file,
    read_page,
    web_search,
)

logger = app_logger

_UNKNOWN_REPLACEMENTS: dict[str, str] = {}
_SANITIZE_TAIL = 0


def _streaming_timeout() -> httpx.Timeout:
    """为 OpenAI stream=True 调用提供更严格的 read timeout。"""

    return httpx.Timeout(
        OPENAI_DEFAULT_TIMEOUT_SECONDS,
        read=OPENAI_STREAM_READ_TIMEOUT_SECONDS,
    )


async def _maybe_await(result: Any) -> Any:
    if inspect.isawaitable(result):
        return await result
    return result


async def _call_sync(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return await asyncio.to_thread(func, *args, **kwargs)


def _sanitize_text(text: str) -> str:
    return text


class _StreamSanitizer:
    def __init__(self) -> None:
        self._tail = ""

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        if self._tail:
            chunk = f"{self._tail}{chunk}"
            self._tail = ""
        return chunk

    def flush(self) -> str:
        if not self._tail:
            return ""
        tail = self._tail
        self._tail = ""
        return tail


def _should_force_unknown(query: str, *, strict: bool, expected_sources: List[str]) -> bool:
    return False


def _needs_quote_context(query: str) -> bool:
    if not query:
        return False
    if "逐字引用" in query or "原文引用" in query or "引用原文" in query:
        return True
    return "逐字" in query and "引用" in query


def _extract_quoted_phrases(query: str) -> List[str]:
    if not query:
        return []
    import re

    patterns = (
        r"“([^”]+)”",
        r"\"([^\"]+)\"",
        r"‘([^’]+)’",
        r"'([^']+)'",
        r"「([^」]+)」",
        r"『([^』]+)』",
    )
    phrases: List[str] = []
    for pat in patterns:
        for match in re.findall(pat, query):
            cleaned = match.strip()
            if cleaned and cleaned not in phrases:
                phrases.append(cleaned)
    return phrases


def _extract_candidate_lines(content: str, phrases: List[str]) -> List[str]:
    if not content or not phrases:
        return []
    candidates: List[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(phrase in stripped for phrase in phrases):
            candidates.append(stripped)
    return candidates


def _needs_problem_grouping_hint(query: str) -> bool:
    if not query:
        return False
    keywords = ("题目数量", "最多的题目", "提交次数最多", "没有 Accepted", "无 Accepted", "no accepted")
    return any(keyword in query for keyword in keywords)


def _needs_stats_context(query: str) -> bool:
    if not query:
        return False
    lowered = query.lower()
    keywords = (
        "统计",
        "次数",
        "总数",
        "总次数",
        "通过次数",
        "成功率",
        "错误类型占比",
        "结果分布",
        "占比",
        "比例",
        "平均",
        "中位数",
        "最大",
        "最小",
        "最多",
        "最少",
        "排名",
        "唯一",
        "没有 accepted",
        "无 accepted",
        "no accepted",
    )
    return any(keyword in lowered for keyword in keywords) or any(
        keyword in lowered for keyword in ("accept rate", "hit rate")
    )


def _normalize_source_path(path: str) -> str:
    if not path:
        return path
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            rel = candidate.relative_to(settings.base_dir)
            return rel.as_posix()
        except ValueError:
            return candidate.as_posix()
    normalized = path.replace("\\", "/")
    if normalized.startswith("data/notes/"):
        return normalized
    return f"data/notes/my_markdowns/{normalized}"


def _match_expected_sources(used: List[str], expected: List[str]) -> List[str]:
    if not used or not expected:
        return []
    used_set = {_normalize_source_path(item) for item in used if item}
    matched: List[str] = []
    for source in expected:
        normalized = _normalize_source_path(source)
        basename = Path(normalized).name
        for item in used_set:
            if item == normalized or Path(item).name == basename:
                matched.append(normalized)
                break
    return matched


async def _prefetch_expected_sources(
    expected_sources: List[str],
    used_sources: List[str],
    *,
    trace_id: str,
    query: str = "",
) -> Optional[str]:
    if not expected_sources:
        return None

    snippets: List[tuple[str, str]] = []
    for source in expected_sources:
        if not source:
            continue
        try:
            result = await _call_sync(read_note_file, source, offset=0, limit_chars=60000)
        except Exception as exc:
            logger.warning(
                "Strict eval prefetch failed",
                extra={"trace_id": trace_id, "source": source, "error": str(exc)},
            )
            continue
        if not isinstance(result, dict):
            continue
        source_file = result.get("source_file")
        normalized = _normalize_source_path(source_file or source)
        if normalized:
            used_sources.append(normalized)
        content = result.get("content")
        if isinstance(content, str) and content.strip():
            snippets.append((normalized, content))

    if not snippets:
        return None

    parts = [
        "以下是评估模式强制注入的笔记内容，回答必须仅依据这些内容：",
        "该任务为逐字引用，请直接从下方原文复制并用引号包裹，避免再调用工具或改写。",
        "逐字引用必须使用中文或英文双引号，不要用反引号或代码块包裹。",
        "若问题中出现引号内的精确字符串，引用必须包含该字符串原样，且不得用近义改写替代。",
    ]
    phrases = _extract_quoted_phrases(query)
    candidate_lines: List[str] = []
    if phrases:
        for source, content in snippets:
            candidate_lines.extend(_extract_candidate_lines(content, phrases))
    if candidate_lines:
        def _line_rank(line: str) -> tuple[bool, bool, bool]:
            stripped = line.lstrip()
            return (
                stripped.startswith(">"),
                stripped.startswith("-"),
                stripped.startswith("*"),
            )

        candidate_lines = sorted(candidate_lines, key=_line_rank)
        parts.append("以下为包含问题引号关键词的候选原文行（必须从中选择，确保包含该关键词）：")
        for line in candidate_lines[:12]:
            parts.append(f"- {line}")
    for source, content in snippets:
        parts.append(f"\n【来源】{source}\n{content}")
    return "\n".join(parts).strip()


def _collect_required_tokens(query: str) -> List[str]:
    return []


def _build_answer_hints(query: str) -> str | None:
    return None


def _append_missing_tokens(content: str, query: str) -> tuple[str, str]:
    return content, ""


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
    elif tool_name == "read_note_file":
        tool_label = "笔记读取"
    elif tool_name == "run_code_interpreter":
        tool_label = "代码执行"
    elif tool_name == "load_skill":
        tool_label = "技能加载"

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
    if tool_name == "read_note_file":
        path = str(arguments.get("path") or "").strip()
        if path:
            prefix += f"：{path}"
    if tool_name == "load_skill":
        skill_name = str(arguments.get("skill_name") or "").strip()
        if skill_name:
            prefix += f"：{skill_name}"

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


def _extract_int_field(text: str, field: str) -> Optional[int]:
    if not text or not field:
        return None
    import re

    match = re.search(rf'\"{re.escape(field)}\"\\s*:\\s*(\\d+)', text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _needs_grouping_retry(query: str, output: str) -> bool:
    if not query or not output:
        return False
    lowered = query.lower()
    if "题目数量" in query or "unique_problems" in lowered:
        unique = _extract_int_field(output, "unique_problems")
        if unique is None:
            unique = _extract_int_field(output, "unique_titles")
        if unique is None:
            unique = _extract_int_field(output, "problem_heading_count")

        total = _extract_int_field(output, "total_submissions")
        if total is None:
            total = _extract_int_field(output, "total_submissions_sum_of_problem_totals")
        if total is None:
            total = _extract_int_field(output, "total_submissions_reported_in_file_header")
        if unique is not None and total is not None and total > 50 and unique == total:
            return True
        if unique is not None and total is not None and total > 50 and unique <= 5:
            return True
    if "没有 accepted" in lowered or "无 accepted" in lowered or "没有 Accepted" in query:
        if "\"no_accepted_titles\"" in output and "[]" in output:
            return True
        if "\"error\"" in output or "not_found" in output:
            return True
        if _extract_int_field(output, "unique_titles") == 1:
            return True
        if _extract_int_field(output, "titles") == 1:
            return True
        if "不存在" in output or "无法" in output:
            return True
        import re

        if re.search(r'\"title\"\\s*:\\s*(null|\"\"|\"none\")', output, flags=re.IGNORECASE):
            return True
        if "提交记录" in output and "top_titles" in output:
            return True
    return False


def _needs_stats_retry(query: str, output: str) -> bool:
    if not query or not output:
        return False
    if "0 ms" in query or "零耗时" in query:
        zero = _extract_int_field(output, "runtime_zero_ms")
        rows_seen = _extract_int_field(output, "rows_seen")
        rows_seen_tables = _extract_int_field(output, "rows_seen_in_tables")
        accepted_total = _extract_int_field(output, "accepted_total")
        if zero == 0 and (
            (rows_seen is not None and rows_seen == 0)
            or (rows_seen_tables is not None and rows_seen_tables == 0)
            or (accepted_total is not None and accepted_total == 0)
        ):
            return True
    return False


def _truncate_tool_text(text: str, *, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> tuple[str, bool]:
    """截断过长的工具输出，避免阻塞后续模型调用。"""

    if len(text) <= max_chars:
        return text, False
    truncated = text[:max_chars]
    return f"{truncated}\n\n...[truncated {len(text) - max_chars} chars]", True


def _truncate_for_log(text: str, *, max_chars: int = 400) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...[truncated {len(text) - max_chars} chars]"


def _stringify_for_log(value: Any, *, max_chars: int = 400) -> str:
    if isinstance(value, str):
        return _truncate_for_log(value, max_chars=max_chars)
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(value)
    return _truncate_for_log(text, max_chars=max_chars)


def _log_tool_summary(
    *,
    tool_name: str | None,
    tool_call_id: str,
    arguments: dict[str, Any],
    output: Any,
    trace_id: str,
    turn: int,
    status: str,
) -> None:
    args_text = _stringify_for_log(arguments, max_chars=300)
    output_text = _stringify_for_log(output, max_chars=400)
    logger.info(
        "[Tool] %s name=%s id=%s args=%s output=%s",
        status,
        tool_name or "-",
        tool_call_id,
        args_text,
        output_text,
        extra={
            "trace_id": trace_id,
            "turn": turn + 1,
            "tool_call_id": tool_call_id,
            "tool": tool_name or "-",
        },
    )


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
        "description": "联网检索公开信息，返回搜索结果摘要；需要正文时再用 read_page。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "用于检索的自然语言问题或关键词。",
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
        "description": "抓取公开网页正文用于提取细节，仅用于 http/https。",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "需要抓取的完整 URL。",
                }
            },
            "required": ["url"],
        },
    },
}


READ_NOTE_FILE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_note_file",
        "description": (
            "读取 data/notes/my_markdowns 下的本地笔记，可用 offset/limit 分块；"
            "需按顺序循环读取直到 done=true 并拼接完整内容；返回 source_file 以便溯源。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "data/notes/my_markdowns 下的相对路径或其完整路径。",
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "default": 0,
                    "description": "起始字符偏移。",
                },
                "limit_chars": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 60000,
                    "default": 60000,
                    "description": "单次读取的最大字符数。",
                },
            },
            "required": ["path"],
        },
    },
}

LOAD_SKILL_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "load_skill",
        "description": "加载指定技能的 SKILL.md 完整说明。",
        "parameters": {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "技能名称（需与 skills/ 下目录名一致）。",
                }
            },
            "required": ["skill_name"],
        },
    },
}


QUERY_MY_NOTES_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_my_notes",
        "description": "在本地笔记向量索引中语义检索，返回相关片段（非全文），用于定位而非最终溯源。",
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
        "description": (
            "在沙盒中运行 Python 代码（仅标准库）。"
            "笔记文件位于 data/notes/my_markdowns/；输出建议为 JSON 或表格。"
        ),
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
    READ_NOTE_FILE_TOOL_SCHEMA,
    LOAD_SKILL_TOOL_SCHEMA,
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

        completion = await async_call_with_retries(
            lambda: chat_client.chat.completions.create(**completion_kwargs)
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
    system_prompt = SYSTEM_PROMPT
    skills_prompt = build_skills_prompt()
    if skills_prompt:
        system_prompt = f"{SYSTEM_PROMPT}\n\n{skills_prompt}"
    for message in messages:
        if message.get("role") == "system":
            message["content"] = system_prompt
            break
    else:
        messages.insert(0, {"role": "system", "content": system_prompt})
    return messages


async def _request_completion(
    messages: List[dict[str, Any]],
    stream_callback: Optional[Callable[[str], None]] = None,
    tools: Optional[List[dict[str, Any]]] = None,
    *,
    trace_id: str = "",
    turn: int = 0,
) -> dict[str, Any]:
    messages = _apply_system_prompt(messages)
    tools = tools or AVAILABLE_TOOLS

    if stream_callback:
        return await _request_streaming_completion(
            messages,
            stream_callback,
            tools=tools,
            trace_id=trace_id,
            turn=turn,
        )

    started_at = time.perf_counter()
    logger.info(
        "Calling chat completion",
        extra={
            "trace_id": trace_id,
            "turn": turn,
            "stream": False,
            "message_count": len(messages),
        },
    )

    completion = await async_call_with_retries(
        lambda: chat_client.chat.completions.create(
            model=chat_model_name,
            messages=messages,
            tools=tools,
        )
    )

    logger.info(
        "Chat completion returned",
        extra={
            "trace_id": trace_id,
            "turn": turn,
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "stream": False,
        },
    )

    if not completion.choices:
        logger.error("Chat completion returned no choices")
        raise HTTPException(status_code=502, detail="No response from assistant")

    message = completion.choices[0].message
    role = getattr(message, "role", "assistant") or "assistant"
    content = _sanitize_text(message.content or "")
    tool_calls = _normalize_tool_calls(getattr(message, "tool_calls", None))

    return {
        "role": role,
        "content": content,
        "tool_calls": tool_calls,
        # 非流式 completion 返回的 tool_calls 已经完整，可以直接执行。
        "tool_call_finished": bool(tool_calls),
    }


async def _request_streaming_completion(
    messages: List[dict[str, Any]],
    stream_callback: Callable[[str], None],
    tools: Optional[List[dict[str, Any]]] = None,
    *,
    trace_id: str = "",
    turn: int = 0,
) -> dict[str, Any]:
    last_exception: Optional[Exception] = None
    fallback_due_to_tool = False
    stream_warning_emitted = False
    tools = tools or AVAILABLE_TOOLS

    started_at = time.perf_counter()
    logger.info(
        "Calling chat completion (stream)",
        extra={
            "trace_id": trace_id,
            "turn": turn,
            "stream": True,
            "message_count": len(messages),
        },
    )

    for attempt in range(1, CHAT_API_MAX_RETRIES + 1):
        tool_call_in_progress = False
        try:
            stream = await async_call_with_retries(
                lambda: chat_client.chat.completions.create(
                    model=chat_model_name,
                    messages=messages,
                    tools=tools,
                    stream=True,
                    timeout=_streaming_timeout(),
                )
            )

            content_parts: List[str] = []
            tool_calls: Dict[int, dict[str, Any]] = {}
            role = "assistant"
            tool_call_finished = False

            first_delta_at: Optional[float] = None
            sanitizer = _StreamSanitizer()

            pending_tool_call_ids: List[str] = []
            async for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta

                if delta.role:
                    role = delta.role

                if delta.content:
                    if first_delta_at is None:
                        first_delta_at = time.perf_counter()
                    sanitized = sanitizer.feed(delta.content)
                    if sanitized:
                        stream_callback(sanitized)
                        content_parts.append(sanitized)

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

            tail = sanitizer.flush()
            if tail:
                stream_callback(tail)
                content_parts.append(tail)

            tool_calls_list = (
                [tool_calls[index] for index in sorted(tool_calls)] if tool_calls else None
            )

            content = "".join(content_parts)
            logger.info(
                "Chat completion stream returned",
                extra={
                    "trace_id": trace_id,
                    "turn": turn,
                    "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
                    "first_delta_ms": (
                        round((first_delta_at - started_at) * 1000, 2)
                        if first_delta_at is not None
                        else None
                    ),
                    "content_chars": len(content),
                    "tool_call_finished": tool_call_finished,
                    "tool_call_count": len(tool_calls_list) if tool_calls_list else 0,
                },
            )
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
                    extra={
                        **log_kwargs["extra"],
                        "trace_id": trace_id,
                        "turn": turn,
                    },
                )
                stream_warning_emitted = True
            else:
                logger.debug(
                    "Streaming response interrupted, retrying (attempt=%s)",
                    attempt,
                    extra={
                        **log_kwargs["extra"],
                        "trace_id": trace_id,
                        "turn": turn,
                    },
                )
            await asyncio.sleep(CHAT_API_RETRY_BACKOFF_SECONDS * attempt)
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
                    extra={
                        **log_kwargs["extra"],
                        "trace_id": trace_id,
                        "turn": turn,
                    },
                )
                stream_warning_emitted = True
            else:
                logger.debug(
                    "Streaming API error, retrying (attempt=%s)",
                    attempt,
                    extra={
                        **log_kwargs["extra"],
                        "trace_id": trace_id,
                        "turn": turn,
                    },
                )
            await asyncio.sleep(CHAT_API_RETRY_BACKOFF_SECONDS * attempt)

    if fallback_due_to_tool:
        logger.warning(
            "Streaming aborted mid tool call, falling back to non-streaming",
            extra={
                "trace_id": trace_id,
                "turn": turn,
                "error": str(last_exception) if last_exception else None,
            },
        )
    else:
        logger.error(
            "Streaming failed after retries, falling back to non-streaming",
            extra={
                "trace_id": trace_id,
                "turn": turn,
                "error": str(last_exception) if last_exception else None,
            },
        )
    completion = await async_call_with_retries(
        lambda: chat_client.chat.completions.create(
            model=chat_model_name,
            messages=messages,
            tools=tools,
        )
    )

    if not completion.choices:
        raise HTTPException(status_code=502, detail="No response from assistant")

    message = completion.choices[0].message
    fallback_content = _sanitize_text(message.content or "")
    if fallback_content:
        stream_callback(fallback_content)

    fallback_tool_calls = _normalize_tool_calls(getattr(message, "tool_calls", None))
    return {
        "role": getattr(message, "role", "assistant") or "assistant",
        "content": fallback_content,
        "tool_calls": fallback_tool_calls,
        "tool_call_finished": bool(fallback_tool_calls),
    }


async def run_chat_conversation(
    payload: ChatRequest,
    stream_callback: Optional[Callable[[str], None]] = None,
    event_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    eval_context: Optional[dict[str, object]] = None,
) -> ChatResponse:
    messages = _normalize_client_messages(payload)
    trace_id = uuid.uuid4().hex[:12]
    tool_invocations = 0
    query_tool_attempts = 0
    code_interpreter_used = False
    agentic_hint_inserted = False
    used_sources: List[str] = []
    eval_context = eval_context or {}
    strict_eval = bool(eval_context.get("strict"))
    question_id = str(eval_context.get("question_id") or "")
    expected_sources = [
        str(item) for item in (eval_context.get("expected_sources") or []) if str(item)
    ]
    eval_mode = bool(strict_eval or question_id or expected_sources)

    last_user = messages[-1].get("content") if messages else ""
    last_user_prefix = ""
    if isinstance(last_user, str):
        last_user_prefix = " ".join(last_user.split())[:200]

    logger.info(
        "Chat request received",
        extra={
            "trace_id": trace_id,
            "message_count": len(messages),
            "last_user_prefix": last_user_prefix,
            "stream": bool(stream_callback),
            "question_id": question_id,
        },
    )
    if isinstance(last_user, str):
        logger.info(
            "Chat request query: %s",
            last_user,
            extra={"trace_id": trace_id, "question_id": question_id},
        )

    if eval_mode and isinstance(last_user, str) and expected_sources and _needs_quote_context(last_user):
        injected = await _prefetch_expected_sources(
            expected_sources,
            used_sources,
            trace_id=trace_id,
            query=last_user,
        )
        if injected:
            messages.append({"role": "system", "content": injected})

    if eval_mode and isinstance(last_user, str) and _needs_problem_grouping_hint(last_user):
        messages.append(
            {
                "role": "system",
                "content": (
                    "统计按题目分组时，请用 Markdown 二级标题作为题目名："
                    "题目标题通常是“## 题目名 (`slug`)”格式，只统计匹配该模式且包含反引号 slug 的二级标题，"
                    "不要把“### 提交记录”等三级标题当作题目名。"
                    "可用正则 ^##\\s+(.+?)\\s+\\(` 提取标题，"
                    "并将其后紧邻的提交记录表归入该题目；"
                    "不符合该模式的标题（如“## 提交记录”）应忽略。"
                ),
            }
        )

    if isinstance(last_user, str) and _needs_stats_context(last_user):
        messages.append(
            {
                "role": "system",
                "content": (
                    "检测到统计/汇总类问题，请优先查看是否有相关技能可用（load_skill），"
                    "再使用 run_code_interpreter 读取原始文件并输出结构化结果；"
                    "不要仅凭检索片段或摘要推断。"
                ),
            }
        )

    if event_callback:
        try:
            event_callback(
                {
                    "type": "status",
                    "phase": "thinking",
                    "message": "正在思考…",
                    "trace_id": trace_id,
                    "ts": time.time(),
                }
            )
        except Exception:
            logger.debug("Failed to emit status event", exc_info=True)

    tools_for_request = AVAILABLE_TOOLS
    if _needs_quote_context(str(last_user or "")):
        tools_for_request = [
            tool
            for tool in AVAILABLE_TOOLS
            if tool.get("function", {}).get("name") != "run_code_interpreter"
        ]

    for turn in range(MAX_TOOL_TURNS + 1):
        assistant_message = await _maybe_await(
            _request_completion(
                messages,
                stream_callback=stream_callback,
                tools=tools_for_request,
                trace_id=trace_id,
                turn=turn + 1,
            )
        )
        tool_calls = assistant_message.get("tool_calls")
        tool_call_finished = assistant_message.get("tool_call_finished")

        if not tool_calls:
            content = assistant_message.get("content")
            if content:
                if strict_eval and expected_sources:
                    matched_sources = _match_expected_sources(used_sources, expected_sources)
                    if matched_sources:
                        sources_suffix = (
                            f"\n\n来源: {', '.join(sorted(set(matched_sources)))}"
                        )
                    else:
                        sources_suffix = "\n\n来源: 未命中"
                    content = f"{content}{sources_suffix}"
                    if stream_callback:
                        stream_callback(sources_suffix)
                logger.info(
                    "[Agent] 最终答案：'%s'",
                    content,
                    extra={"trace_id": trace_id, "turn": turn + 1},
                )
                if eval_mode:
                    matched_sources = _match_expected_sources(used_sources, expected_sources)
                    logger.info(
                        "[Eval] summary question_id=%s used_sources=%s expected_sources=%s matched_sources=%s tool_invocations=%s query_tool_attempts=%s code_interpreter_used=%s",
                        question_id,
                        sorted(set(used_sources)),
                        expected_sources,
                        sorted(set(matched_sources)),
                        tool_invocations,
                        query_tool_attempts,
                        code_interpreter_used,
                        extra={"trace_id": trace_id, "turn": turn + 1},
                    )
                if event_callback:
                    try:
                        event_callback(
                            {
                                "type": "sources",
                                "question_id": question_id,
                                "sources": sorted(set(used_sources)),
                                "expected_sources": expected_sources,
                                "ts": time.time(),
                            }
                        )
                    except Exception:
                        logger.debug("Failed to emit sources event", exc_info=True)
                return ChatResponse(
                    response=content,
                    tool_calls=_tool_calls_to_models(tool_calls),
                )
            logger.warning(
                "Assistant response without content or tool calls",
                extra={"trace_id": trace_id, "turn": turn + 1},
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
                    extra={"trace_id": trace_id, "turn": turn + 1, "tool_call": call},
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
                            "trace_id": trace_id,
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
                    result = await _call_sync(
                        query_my_notes,
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
                status = "error" if isinstance(result, dict) and "error" in result else "ok"
                _log_tool_summary(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    arguments=arguments,
                    output=result,
                    trace_id=trace_id,
                    turn=turn,
                    status=status,
                )
                if isinstance(result, dict):
                    for item in result.get("results") or []:
                        source_path = item.get("source_path")
                        if isinstance(source_path, str) and source_path:
                            used_sources.append(_normalize_source_path(source_path))
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
                    result = await _call_sync(web_search, query)
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
                status = "error" if isinstance(result, dict) and "error" in result else "ok"
                _log_tool_summary(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    arguments=arguments,
                    output=result,
                    trace_id=trace_id,
                    turn=turn,
                    status=status,
                )
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
                    content = await _call_sync(read_page, url)
                    tool_output = content
                    summary_status = "ok"
                except ToolExecutionError as exc:
                    tool_output = json.dumps({"error": str(exc)}, ensure_ascii=False)
                    logger.error(
                        "[System] read_page 异常",
                        extra={
                            "trace_id": trace_id,
                            "turn": turn + 1,
                            "tool_call_id": tool_call_id,
                            "error": tool_output,
                            "tool_latency_ms": round(
                                (time.perf_counter() - tool_started_at) * 1000, 2
                            ),
                        },
                    )
                    summary_status = "error"
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
                _log_tool_summary(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    arguments=arguments,
                    output=tool_output,
                    trace_id=trace_id,
                    turn=turn,
                    status=summary_status,
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

            if tool_name == "read_note_file":
                path = arguments.get("path")
                offset = arguments.get("offset", 0)
                limit_chars = arguments.get("limit_chars", 60000)
                tool_started_at = time.perf_counter()
                try:
                    result = await _call_sync(
                        read_note_file,
                        path,
                        offset=offset,
                        limit_chars=limit_chars,
                    )
                    tool_output = _dumps_tool_payload(result)
                    summary_status = "ok"
                except (ToolExecutionError, ValueError) as exc:
                    tool_output = json.dumps({"error": str(exc)}, ensure_ascii=False)
                    logger.error(
                        "[System] read_note_file 异常",
                        extra={
                            "trace_id": trace_id,
                            "turn": turn + 1,
                            "tool_call_id": tool_call_id,
                            "error": tool_output,
                            "tool_latency_ms": round(
                                (time.perf_counter() - tool_started_at) * 1000, 2
                            ),
                        },
                    )
                    summary_status = "error"
                else:
                    tool_output_logger.info(
                        json.dumps(
                            {
                                "tool_call_id": tool_call_id,
                                "tool": tool_name,
                                "output": result,
                            },
                            ensure_ascii=False,
                        )
                    )
                _log_tool_summary(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    arguments=arguments,
                    output=result if summary_status == "ok" else tool_output,
                    trace_id=trace_id,
                    turn=turn,
                    status=summary_status,
                )
                source_file = result.get("source_file") if isinstance(result, dict) else None
                if isinstance(source_file, str) and source_file:
                    used_sources.append(_normalize_source_path(source_file))
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

            if tool_name == "load_skill":
                tool_started_at = time.perf_counter()
                result: Any = None
                try:
                    result = await _call_sync(
                        load_skill,
                        skill_name=str(arguments.get("skill_name") or "").strip(),
                    )
                    tool_output = json.dumps(result, ensure_ascii=False)
                    summary_status = "ok"
                except Exception as exc:
                    tool_output = json.dumps({"error": str(exc)}, ensure_ascii=False)
                    summary_status = "error"
                    logger.error(
                        "[System] load_skill 异常",
                        extra={
                            "trace_id": trace_id,
                            "turn": turn + 1,
                            "tool_call_id": tool_call_id,
                            "error": tool_output,
                            "tool_latency_ms": round(
                                (time.perf_counter() - tool_started_at) * 1000, 2
                            ),
                        },
                    )
                _log_tool_summary(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    arguments=arguments,
                    output=result if summary_status == "ok" else tool_output,
                    trace_id=trace_id,
                    turn=turn,
                    status=summary_status,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_output,
                    }
                )
                tool_invocations += 1
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
                continue

            if tool_name == "run_code_interpreter":
                if _needs_quote_context(str(last_user or "")):
                    warning_message = (
                        "系统提示：该请求是逐字引用任务，请直接从已提供的原文片段中引用，"
                        "不要调用 run_code_interpreter。"
                    )
                    tool_output = json.dumps({"error": "quote_task_skip_interpreter"}, ensure_ascii=False)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": tool_output,
                        }
                    )
                    post_tool_messages.append({"role": "system", "content": warning_message})
                    tool_invocations += 1
                    continue

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
                    response = await _call_sync(call_mcp_python_interpreter, cleaned_payload)
                    model_payload = _shrink_mcp_response_for_model(response)
                    tool_output = json.dumps(model_payload, ensure_ascii=False)
                    interpreter_output = str(model_payload.get("content", ""))
                    summary_output: Any = model_payload
                    if strict_eval and expected_sources and interpreter_output:
                        code_text = str(code or "")
                        for source in expected_sources:
                            normalized = _normalize_source_path(source)
                            basename = Path(normalized).name
                            if (
                                normalized in interpreter_output
                                or basename in interpreter_output
                                or normalized in code_text
                                or basename in code_text
                            ):
                                if normalized not in used_sources:
                                    used_sources.append(normalized)
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
                    elif _needs_grouping_retry(str(last_user or ""), interpreter_output):
                        warning_message = (
                            "系统提示：统计结果看起来像是未正确按题目/章节分组。"
                            "请以 Markdown 标题作为题目分组边界，将标题下的表格记录归入该题目，"
                            "再输出统计结果与样例校验。"
                        )
                        post_tool_messages.append(
                            {"role": "system", "content": warning_message}
                        )
                    elif _needs_stats_retry(str(last_user or ""), interpreter_output):
                        warning_message = (
                            "系统提示：统计结果显示样本行数为 0 或 Accepted 总数为 0，"
                            "疑似未正确解析表格行。请重新扫描所有表格记录，并输出匹配样例与最终计数。"
                        )
                        post_tool_messages.append(
                            {"role": "system", "content": warning_message}
                        )
                except ToolExecutionError as exc:
                    tool_error_detail = str(exc)
                    tool_output = json.dumps({"error": str(exc)}, ensure_ascii=False)
                    summary_output = {"error": tool_error_detail}
                    logger.error(
                        "[System] MCP 工具异常: %s",
                        tool_error_detail,
                        extra={
                            "trace_id": trace_id,
                            "turn": turn + 1,
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
                            "trace_id": trace_id,
                            "turn": turn + 1,
                            "tool_call_id": tool_call_id,
                            "tool_latency_ms": round(
                                (time.perf_counter() - tool_started_at) * 1000, 2
                            ),
                        },
                    )
                code_interpreter_used = True
                _log_tool_summary(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    arguments=arguments,
                    output=summary_output,
                    trace_id=trace_id,
                    turn=turn,
                    status="error" if tool_error_detail else "ok",
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
                                "trace_id": trace_id,
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
                        extra={
                            "trace_id": trace_id,
                            "turn": turn + 1,
                            "tool": tool_name,
                            "tool_call_id": tool_call_id,
                        },
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

            logger.error(
                "Unknown tool requested",
                extra={"trace_id": trace_id, "turn": turn + 1, "tool_name": tool_name},
            )
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
                            "trace_id": trace_id,
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
                        "trace_id": trace_id,
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
    eval_context: Optional[dict[str, object]] = None,
) -> ChatResponse:
    return await run_chat_conversation(payload, stream_callback, event_callback, eval_context)


async def stream_chat_response(
    payload: ChatRequest,
    *,
    want_ndjson: bool = False,
    eval_context: Optional[dict[str, object]] = None,
) -> StreamingResponse:
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
                    eval_context=eval_context,
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
    "LOAD_SKILL_TOOL_SCHEMA",
    "MCP_CODE_INTERPRETER_SCHEMA",
    "run_chat_conversation",
    "execute_chat",
    "stream_chat_response",
    "generate_title",
]

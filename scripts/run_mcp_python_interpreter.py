#!/usr/bin/env python
"""Bridge script to call yzfly/mcp-python-interpreter via MCP stdio."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
import mcp.types as types


def _read_payload() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        raise ValueError("未提供 JSON 输入")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - invalid caller input
        raise ValueError("无法解析 JSON 输入") from exc


def _build_tool_params(payload: dict[str, Any]) -> dict[str, Any]:
    if "code" not in payload or not str(payload["code"]).strip():
        raise ValueError("缺少 code 内容")

    params: dict[str, Any] = {
        "code": str(payload["code"]),
        "execution_mode": payload.get("execution_mode", "inline"),
        "session_id": payload.get("session_id", "default"),
        "environment": payload.get("environment", "system"),
        "timeout": int(payload.get("timeout", 300)),
    }

    save_as = payload.get("save_as")
    if save_as:
        params["save_as"] = str(save_as)

    return params


def _render_content(result: types.CallToolResult) -> str:
    parts: list[str] = []
    for block in result.content or []:
        if isinstance(block, types.TextContent):
            parts.append(block.text)
        else:  # Other block types can be represented generically
            parts.append(f"[{block.type} block]")
    return "\n".join(parts).strip()


async def _call_tool_stdio(
    server_params: StdioServerParameters,
    tool_params: dict[str, Any],
    timeout_seconds: int,
) -> types.CallToolResult:
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            return await asyncio.wait_for(
                session.call_tool("run_python_code", tool_params),
                timeout=timeout_seconds,
            )


async def _call_tool_sse(
    endpoint: str,
    tool_params: dict[str, Any],
    timeout_seconds: int,
) -> types.CallToolResult:
    async with sse_client(endpoint, timeout=10) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            return await asyncio.wait_for(
                session.call_tool("run_python_code", tool_params),
                timeout=timeout_seconds,
            )


def _default_server_command(path_hint: str | None) -> str:
    return path_hint or "mcp-python-interpreter"


def main() -> None:
    parser = argparse.ArgumentParser(description="Invoke MCP Python Interpreter")
    parser.add_argument("--workdir", help="执行沙盒工作目录 (仅本地 stdio 模式需要)")
    parser.add_argument(
        "--server-command",
        default=None,
        help="mcp-python-interpreter 可执行文件路径",
    )
    parser.add_argument(
        "--server-python-path",
        default=None,
        help="传递给 mcp-python-interpreter 的 --python-path 参数",
    )
    parser.add_argument(
        "--process-timeout",
        type=int,
        default=420,
        help="整个工具调用的超时时间 (秒)",
    )
    parser.add_argument(
        "--allow-system-access",
        action="store_true",
        help="是否允许解释器访问系统目录",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="已常驻的 MCP Interpreter SSE 端点 (例如 http://127.0.0.1:9070/sse/)",
    )
    args = parser.parse_args()

    payload = _read_payload()
    tool_params = _build_tool_params(payload)

    try:
        timeout = max(args.process_timeout, 1)
        if args.endpoint:
            result = asyncio.run(
                _call_tool_sse(args.endpoint, tool_params, timeout)
            )
        else:
            if not args.workdir:
                raise ValueError("--workdir 参数在本地 stdio 模式下必填")
            workdir = Path(args.workdir).expanduser().resolve()
            workdir.mkdir(parents=True, exist_ok=True)

            server_args = ["--dir", str(workdir)]
            if args.server_python_path:
                server_args += ["--python-path", args.server_python_path]

            env = os.environ.copy()
            if not args.allow_system_access:
                env["MCP_ALLOW_SYSTEM_ACCESS"] = "0"

            server_params = StdioServerParameters(
                command=_default_server_command(args.server_command),
                args=server_args,
                env=env,
                cwd=str(workdir),
            )

            result = asyncio.run(
                _call_tool_stdio(server_params, tool_params, timeout)
            )
        content_text = _render_content(result)
        print(
            json.dumps(
                {
                    "ok": True,
                    "content": content_text,
                    "raw": result.model_dump(mode="json"),
                },
                ensure_ascii=False,
            )
        )
    except Exception as exc:  # pragma: no cover - surfaced to caller
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                },
                ensure_ascii=False,
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

import json

import asyncio


def test_format_tool_status_message_query_my_notes() -> None:
    from backend.app.services.chat import _format_tool_status_message

    message = _format_tool_status_message(
        tool_name="query_my_notes",
        tool_index=2,
        tool_count=2,
        arguments={"query": "如何配置 FastAPI 的 StreamingResponse?", "top_k": 5},
        stage="start",
    )

    assert "笔记检索" in message
    assert "第 2 次" in message
    assert "FastAPI" in message


def test_is_tool_error_output() -> None:
    from backend.app.services.chat import _is_tool_error_output

    assert _is_tool_error_output('{"error": "boom"}')
    assert not _is_tool_error_output('{"ok": true}')
    assert not _is_tool_error_output("plain text")


def test_stream_chat_response_ndjson_emits_done(monkeypatch) -> None:
    from backend.app.models.schemas import ChatRequest
    from backend.app.services import chat as chat_service

    async def fake_execute_chat(*args, **kwargs):
        stream_callback = kwargs.get("stream_callback")
        event_callback = kwargs.get("event_callback")
        if event_callback:
            event_callback({"type": "status", "message": "正在思考…"})
        if stream_callback:
            stream_callback("hello")
            stream_callback(" world")
        return chat_service.ChatResponse(response="hello world")

    monkeypatch.setattr(chat_service, "execute_chat", fake_execute_chat)

    async def _consume() -> list[dict]:
        response = await chat_service.stream_chat_response(
            ChatRequest(user_message="hi"),
            want_ndjson=True,
        )
        chunks: list[str] = []
        async for part in response.body_iterator:  # type: ignore[attr-defined]
            chunks.append(part)
        payload = "".join(chunks)
        return [json.loads(line) for line in payload.splitlines() if line.strip()]

    events = asyncio.run(_consume())
    assert any(event.get("type") == "delta" for event in events)
    assert events[-1].get("type") == "done"


def test_run_code_interpreter_error_emits_tool_error_event(monkeypatch) -> None:
    from backend.app.models.schemas import ChatRequest
    from backend.app.services import chat as chat_service
    from backend.app.services.exceptions import ToolExecutionError

    async def fake_request_completion(messages, stream_callback=None, **_kwargs):
        # First turn requests tool, second turn returns final answer.
        if not any(message.get("role") == "tool" for message in messages):
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "run_code_interpreter",
                            "arguments": '{"code": "print(1)", "timeout": 1}',
                        },
                    }
                ],
                "tool_call_finished": True,
            }
        return {
            "role": "assistant",
            "content": "done",
            "tool_calls": None,
            "tool_call_finished": False,
        }

    def fake_call_mcp_python_interpreter(payload):
        raise ToolExecutionError("boom")

    monkeypatch.setattr(chat_service, "_request_completion", fake_request_completion)
    monkeypatch.setattr(chat_service, "call_mcp_python_interpreter", fake_call_mcp_python_interpreter)

    async def _consume() -> list[dict]:
        response = await chat_service.stream_chat_response(
            ChatRequest(user_message="hi"),
            want_ndjson=True,
        )
        chunks: list[str] = []
        async for part in response.body_iterator:  # type: ignore[attr-defined]
            chunks.append(part)
        payload = "".join(chunks)
        return [json.loads(line) for line in payload.splitlines() if line.strip()]

    events = asyncio.run(_consume())
    tool_events = [event for event in events if event.get("type") == "tool"]
    assert tool_events, "expected tool event"
    assert any(event.get("stage") == "error" for event in tool_events)
    assert any((event.get("error") or "") == "boom" for event in tool_events)

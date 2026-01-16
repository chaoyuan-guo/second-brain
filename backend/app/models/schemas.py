"""Pydantic 数据模型定义."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class ConversationMessage(BaseModel):
    """代表一次对话消息."""

    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


class ChatRequest(BaseModel):
    """前端发起对话时的请求载荷."""

    user_message: Optional[str] = None
    messages: Optional[list[ConversationMessage]] = None


class ToolCallFunction(BaseModel):
    """工具调用函数信息."""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """工具调用结构化表示."""

    id: str
    type: str
    function: ToolCallFunction


class ChatResponse(BaseModel):
    """后端返回的聊天响应."""

    response: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None


class ChatTitleResponse(BaseModel):
    """模型生成标题的响应."""

    title: str


__all__ = [
    "ConversationMessage",
    "ChatRequest",
    "ToolCallFunction",
    "ToolCall",
    "ChatResponse",
    "ChatTitleResponse",
]

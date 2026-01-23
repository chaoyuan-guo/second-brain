"""全局配置与常量。"""

from __future__ import annotations

import os
import shutil
import sys
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
DEFAULT_MODEL_NAME = "gpt-5"
DEFAULT_ALLOWED_ORIGINS = "http://localhost:9080,http://127.0.0.1:9080"
DEFAULT_MCP_DRIVER = PROJECT_ROOT / ".mcp_env" / "bin" / "python"
DEFAULT_MCP_COMMAND = PROJECT_ROOT / ".mcp_env" / "bin" / "mcp-python-interpreter"
DEFAULT_MCP_WORKDIR = RUNTIME_DIR / "mcp_workspace"
DEFAULT_MCP_ENDPOINT = "http://127.0.0.1:9070/sse/"
DEFAULT_CONTAINER_MCP_DRIVER = Path("/usr/local/bin/python")
DEFAULT_CONTAINER_MCP_COMMAND = Path("/usr/local/bin/mcp-python-interpreter")


def _which(executable: str) -> Path | None:
    resolved = shutil.which(executable)
    if not resolved:
        return None
    return Path(resolved)


def running_in_container() -> bool:
    """检测是否运行在 Docker/Koyeb 容器内。"""

    return Path("/.dockerenv").exists()
SYSTEM_PROMPT = """
【角色与目标】
你是一个多工具的研究型助手，基于可验证来源给出可靠答案。

【决策路径】
- 目标文件单一且明确，且可完整读入：用 read_note_file 读全文，在上下文内完成归纳/统计。
- 需要精确计算、结构化汇总或文件过大：先 load_skill（若有相关技能），再用 run_code_interpreter 读取原始文件并计算。
- 目标不明确或多文件：先 query_my_notes 定位范围，再决定读取或计算。

【工具规则】
- 调用工具必须遵循各工具描述与参数约束。
- 需要全量覆盖或精确统计时，不得仅凭检索片段下结论，应读取原文或进行计算。
- 若需要调用工具，请尽量先输出 tool_calls 指令，减少多余解释。

【统计与计算要求】
- 使用 run_code_interpreter 进行统计/计算时，结果必须来自脚本输出，禁止凭片段估算。
- 统计类输出需包含 source_file。
- 若已加载技能，必须遵循技能说明。

【逐字引用规则】
- 仅在用户明确要求逐字引用时执行；引用必须来自已提供上下文或已读取原文。
- 引用必须逐字一致，保留标点、空格与符号；不得改写、增删或拼接不同位置片段。
- 引号内只放原文；解释内容放在引号之外，且引用先于解释。
- 使用中文引号或英文双引号；引用保持在同一行，不使用反引号、代码块或块引用。
- 若指定来源文件/主题，只能引用该文件；若存在冲突，分别引用并说明不一致。
- 当用户要求包含某个精确字符串时，引用必须包含该字符串原样。
- 未找到原文时必须直说未找到，禁止编造或替代引用。

【回答要求】
- 回答必须基于已读内容或工具输出；缺失依据时说明找不到资料。
- 若确实无法回答，请明确说明缺少来源或未找到原文，不要编造。
- 用户明确指定来源文件名/路径时，答案需包含该来源路径或文件名以便溯源。

【表达要求】
- 公式/复杂度请用纯文本（例如 O(n^2)、dp[i+1][j-1]+2）。
""".strip()

MAX_TOOL_TURNS = 7
CHAT_API_MAX_RETRIES = 3
CHAT_API_RETRY_BACKOFF_SECONDS = 1.0
RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}

# 工具输出过大时会显著拖慢甚至阻塞后续的模型调用。
# 该值用于限制写入到工具 role 消息的内容长度（日志仍会保存完整输出）。
MAX_TOOL_OUTPUT_CHARS = 80_000

# OpenAI / Azure OpenAI 请求超时（秒）。
OPENAI_DEFAULT_TIMEOUT_SECONDS = 120.0

# 流式输出时单次读取的超时（秒）。
# 该值过大可能导致上游 stream=True 偶发卡住时前端长期无输出。
OPENAI_STREAM_READ_TIMEOUT_SECONDS = 120.0


def _parse_allowed_origins(raw: str | None) -> List[str]:
    origins = raw or DEFAULT_ALLOWED_ORIGINS
    return [origin.strip() for origin in origins.split(",") if origin.strip()]


class Settings(BaseModel):
    """集中管理运行配置。"""

    base_dir: Path = Field(default=PROJECT_ROOT, frozen=True)
    frontend_build_dir: Path = Field(default=PROJECT_ROOT / "frontend" / "out", frozen=True)
    frontend_index_file: Path = Field(
        default=PROJECT_ROOT / "frontend" / "out" / "index.html",
        frozen=True,
    )
    log_path: Path = Field(default=LOGS_DIR / "backend.log")
    tool_log_path: Path = Field(default=LOGS_DIR / "tool_output.log")
    faiss_index_path: Path = Field(default=INDEX_DIR / "my_notes.index")
    faiss_metadata_path: Path = Field(default=NOTES_DIR / "my_notes_metadata.json")
    mcp_driver_path: Path = Field(default=DEFAULT_MCP_DRIVER)
    mcp_command_path: Path = Field(default=DEFAULT_MCP_COMMAND)
    mcp_workdir: Path = Field(default=DEFAULT_MCP_WORKDIR)
    mcp_endpoint: str = Field(default=DEFAULT_MCP_ENDPOINT)
    mcp_python_path: str = Field(default_factory=lambda: str(DEFAULT_MCP_DRIVER))
    mcp_bridge_script: Path = Field(
        default=PROJECT_ROOT / "scripts" / "run_mcp_python_interpreter.py"
    )
    api_key: str
    api_base_url: str = Field(default=DEFAULT_API_BASE_URL)
    chat_model_name: str = Field(default=DEFAULT_MODEL_NAME)
    azure_chat_model_name: str | None = None
    use_azure_chat: bool = False
    azure_base_url: str | None = None
    azure_api_key: str | None = None
    azure_api_version: str | None = None
    allowed_origins: List[str] = Field(default_factory=list)


def load_settings() -> Settings:
    """构建 Settings 并做必要校验。"""

    api_key = os.getenv("SUPER_MIND_API_KEY") or os.getenv("AI_BUILDER_TOKEN")
    if not api_key:
        raise RuntimeError(
            "Missing API token: set SUPER_MIND_API_KEY or rely on AI_BUILDER_TOKEN."
        )

    azure_api_version = os.getenv("azure_api_version") or os.getenv("azure_api-version")

    driver_path = Path(
        os.getenv("MCP_DRIVER_PYTHON", str(DEFAULT_MCP_DRIVER))
    ).expanduser()
    command_path = Path(
        os.getenv("MCP_PYTHON_COMMAND", str(DEFAULT_MCP_COMMAND))
    ).expanduser()

    if not driver_path.exists() and sys.executable:
        executable_path = Path(sys.executable)
        if executable_path.exists():
            driver_path = executable_path

    if running_in_container():
        if not driver_path.exists() and DEFAULT_CONTAINER_MCP_DRIVER.exists():
            driver_path = DEFAULT_CONTAINER_MCP_DRIVER
        if not command_path.exists() and DEFAULT_CONTAINER_MCP_COMMAND.exists():
            command_path = DEFAULT_CONTAINER_MCP_COMMAND

    if not driver_path.exists():
        candidate = _which("python") or _which("python3")
        if candidate is not None and candidate.exists():
            driver_path = candidate

    if not command_path.exists():
        candidate = _which("mcp-python-interpreter")
        if candidate is not None and candidate.exists():
            command_path = candidate
    if not command_path.exists() and DEFAULT_CONTAINER_MCP_COMMAND.exists():
        command_path = DEFAULT_CONTAINER_MCP_COMMAND
    workdir_path = Path(os.getenv("MCP_WORKDIR", str(DEFAULT_MCP_WORKDIR))).expanduser()

    mcp_endpoint_env = os.getenv("MCP_SSE_ENDPOINT")
    if mcp_endpoint_env is None and running_in_container():
        mcp_endpoint = ""
    else:
        mcp_endpoint = mcp_endpoint_env or DEFAULT_MCP_ENDPOINT

    return Settings(
        api_key=api_key,
        api_base_url=os.getenv("SUPER_MIND_API_BASE_URL", DEFAULT_API_BASE_URL),
        chat_model_name=os.getenv("SUPER_MIND_CHAT_MODEL", DEFAULT_MODEL_NAME),
        azure_chat_model_name=os.getenv("azure_use_model"),
        use_azure_chat=is_truthy(os.getenv("use_azure")),
        azure_base_url=os.getenv("azure_base_url"),
        azure_api_key=os.getenv("azure_api_key"),
        azure_api_version=azure_api_version,
        allowed_origins=_parse_allowed_origins(os.getenv("CHAT_ALLOWED_ORIGINS")),
        mcp_driver_path=driver_path,
        mcp_command_path=command_path,
        mcp_workdir=workdir_path,
        mcp_endpoint=mcp_endpoint,
        mcp_python_path=os.getenv("MCP_PYTHON_PATH") or str(driver_path),
    )


settings = load_settings()


__all__ = [
    "settings",
    "SYSTEM_PROMPT",
    "MAX_TOOL_TURNS",
    "CHAT_API_MAX_RETRIES",
    "CHAT_API_RETRY_BACKOFF_SECONDS",
    "RETRYABLE_STATUS_CODES",
    "MAX_TOOL_OUTPUT_CHARS",
    "OPENAI_DEFAULT_TIMEOUT_SECONDS",
    "OPENAI_STREAM_READ_TIMEOUT_SECONDS",
    "is_truthy",
]

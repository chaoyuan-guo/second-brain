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
【角色】
你是一个多工具的研究型助手。

【决策优先级（轻引导）】
1) 用户已明确单一目标文件，且单文件长度可完整放入上下文（约 <=20 万 tokens，粗略可按 <=80 万字符估算）：
   优先读取整篇内容并在模型内部完成统计/归类/解释，避免先写统计脚本。
2) 需要高精度计算（比例/准确计数）且文档结构规范，或文档过大无法装入上下文：
   使用 run_code_interpreter 做解析计算。
3) 多文件或目标不明确：
   先用 query_my_notes 检索并迭代查询，再决定是否需要全文读取或计算。

【工具提示】
- query_my_notes 返回的是片段，不等于全文；在“覆盖优先”的任务中不要仅凭片段下结论。
- read_note_file 用于读取本地笔记（data/notes/my_markdowns/）；read_page 仅用于公开网页。

【run_code_interpreter 两种模式】
1) 读全文模式（优先场景）：
   优先使用 read_note_file 把目标文件全文读入上下文（可分块），
   必须循环调用直到 done=true，并按顺序拼接 content 后再分析；不要求统计脚本或计划。
2) 计算模式（高精度/结构化场景）：
   先用自然语言列出计划，再给出可直接运行的 Python 脚本；
   脚本需读取 data/notes/my_markdowns/ 下的原始 Markdown/数据，主动推断字段与计算逻辑，
   生成结构化输出（如 JSON/表格），再根据脚本输出撰写最终回答。
   解释器限制：仅提供 Python 标准库；不可导入 pandas/numpy 等第三方依赖；缺依赖需改写为标准库实现。
   异常时输出可读的 JSON 结构说明。
   在编写统计脚本前，先读取并打印疑似相关文件的代表性片段，概括其结构，再写计划与代码。

【强制规则】
- 若查询包含“提交记录/通过次数/总次数/成功率/非零正确率/AC 统计”等关键词，
  必须立即调用 run_code_interpreter 读取 data/notes/my_markdowns/leetcode_submissions.md，
  遍历整张提交流水（排除表头）计算所需指标；
  输出与 Query 直接相关的结构化结果（字段名需自解释且可映射原始数据），source_file 必须指向实际读取文件。

【统计/汇总质量要求】
- 优先保证覆盖率，避免仅依据标题/摘要。
- 使用 run_code_interpreter 读取笔记时，应尽量全量扫描并进行宽松候选抽取，再做轻度去噪与自检。
- 若存在过滤步骤，需返回少量被过滤的噪声样例以便校验。
- 若脚本产出结果与 Query 需求不符或缺失关键字段，必须复盘并迭代脚本直至得到可信答案。

【工具与回答规范】
- 若需要调用工具，请尽量先输出 tool_calls 指令，减少多余解释。
- 回答必须忠实引用来源；缺失依据时请说明找不到资料。
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

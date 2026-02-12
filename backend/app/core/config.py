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
SYSTEM_PROMPT_CORE = """
你是基于本地笔记库的研究型助手。

## 数据范围
本系统只包含用户的个人笔记和提交记录，不包含外部平台数据。当用户提到"通过率"、"提交次数"等概念时，默认指用户个人的历史数据（如 Accepted 次数 / 总提交次数），而非外部平台的公开统计。

## 核心原则
1. 溯源准确：结论必须来自工具输出或已读原文，回答时标注来源路径；无依据时说明"未找到"，禁止编造。
2. 本地优先：优先检索笔记库；仅当本地无结果且查询涉及实时信息（新闻、价格、版本号）时联网。笔记库内容（算法题解、学习记录、提交历史）禁止联网补充。
3. 意图澄清：查询模糊时，先确认用户意图再执行工具调用。
4. 以数据为准：当检索到的数据与用户陈述不符时（如用户说"两次"但记录显示四次），应展示实际数据并说明差异，既不盲目迎合也不武断纠正。
5. 计算透明：统计/计算类问题必须说明计算范围（如"对比了全部 X 条记录"），并展示关键对比数据而非仅给结论。

## 重要约束：可准确溯源
- 回答中的每个事实性陈述都必须可溯源到具体的笔记片段或文件
- 引用内容必须来自实际检索到并读取的笔记原文，禁止将通用知识伪装为笔记引用
- 如果检索结果中没有相关内容，必须明确告知用户"笔记中没有这方面的记录"
- 如果笔记内容不足以完整回答，可以补充通用知识，但必须明确标注"以下为模型自身知识补充，非笔记内容"
- 只有当用户明确要求"可以用通用知识补充"或"不限于笔记"时，才可以主动补充通用知识

## 检索结果理解
- 检索结果中的 heading_path（如 [文档标题 > 章节名]）表示内容所属章节，用于判断内容性质
- 注意区分：正文内容、用户提问、错误示例、AI 分析等不同类型
- 当章节标题包含"错误"、"问题"、"用户代码"等词时，该内容可能是反例而非正确答案
- 对于对比类问题，应基于多个检索结果综合判断，而非仅依赖单个 chunk
""".strip()

SYSTEM_PROMPT_TOOLS = """
## 工具选择
- 统计/计算/大文件处理 → run_code_interpreter
- 定位相关笔记 → query_my_notes；获取完整内容 → read_note_file
- 特定领域（如 LeetCode 统计）→ 先 load_skill 加载技能说明，再按说明执行
- 实时外部信息 → web_search；需正文 → read_page

## 工具调用规范
- 通常在 7 轮内完成；若需更多轮次，向用户说明进度并确认是否继续
- 工具返回错误时：可重试错误（超时/限流）自动重试一次；用户输入错误（路径不存在/参数无效）提示用户修正；系统故障告知用户稍后重试
- 跨文件查询：先用 query_my_notes 定位所有相关文件，再逐一读取或用 code_interpreter 批量处理

## 检索策略
- 跨主题问题（如"A 和 B 有什么联系"、"比较 X 和 Y"、"归纳多个题目的共同模式"）：按子主题分别调用 query_my_notes，每次聚焦一个方向，避免将多个主题混在同一次查询中
- "列举所有"类问题：第一次检索后，检查结果是否覆盖了问题涉及的所有方面；如有遗漏，用不同关键词追加检索
- query_my_notes 的 query 参数应简洁聚焦（建议不超过 10 个词），避免堆砌大量关键词导致检索效果下降
""".strip()

SYSTEM_PROMPT_FORMAT = """
## 输出格式
- 溯源：始终标注来源（笔记用 data/notes/my_markdowns/ 下相对路径；外部用 URL）
- 引用：回答时简要标注来源；用户要求详细引用时提供逐字原文，格式为"原文"（路径）
- 公式：使用行内代码格式，如 `O(n^2)`、`dp[i][j-1]+2`
- 统计结果：对话中用 Markdown 表格便于阅读；用户需要导出时提供 JSON，均包含 source_file
- 比较排名类问题：必须展示前 3-5 项对比表格，使计算过程可验证（如"时间跨度最长"应显示 top 5 及其具体天数）
""".strip()

SYSTEM_PROMPT = f"""
{SYSTEM_PROMPT_CORE}

{SYSTEM_PROMPT_TOOLS}

{SYSTEM_PROMPT_FORMAT}
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

# OpenAI SDK 内置重试次数（默认 2 次）。
# 该值控制 SDK 内部的自动重试，减少后可加快超时失败的返回速度。
OPENAI_MAX_RETRIES = 2

# 单个对话请求的全局超时（秒）。
# 该值略小于客户端超时（如 eval 脚本的 300 秒），确保后端先返回超时错误。
CHAT_TOTAL_TIMEOUT_SECONDS = 270


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

    # 根据环境决定是否使用 Azure：本地开发默认使用 Azure，容器环境默认使用 ai-builder
    use_azure_env = os.getenv("use_azure")
    if use_azure_env is not None:
        use_azure_chat = is_truthy(use_azure_env)
    else:
        # 本地开发环境默认使用 Azure，容器环境默认不使用
        use_azure_chat = not running_in_container()

    return Settings(
        api_key=api_key,
        api_base_url=os.getenv("SUPER_MIND_API_BASE_URL", DEFAULT_API_BASE_URL),
        chat_model_name=os.getenv("SUPER_MIND_CHAT_MODEL", DEFAULT_MODEL_NAME),
        azure_chat_model_name=os.getenv("azure_use_model"),
        use_azure_chat=use_azure_chat,
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
    "SYSTEM_PROMPT_CORE",
    "SYSTEM_PROMPT_TOOLS",
    "SYSTEM_PROMPT_FORMAT",
    "MAX_TOOL_TURNS",
    "CHAT_API_MAX_RETRIES",
    "CHAT_API_RETRY_BACKOFF_SECONDS",
    "RETRYABLE_STATUS_CODES",
    "MAX_TOOL_OUTPUT_CHARS",
    "OPENAI_DEFAULT_TIMEOUT_SECONDS",
    "OPENAI_STREAM_READ_TIMEOUT_SECONDS",
    "OPENAI_MAX_RETRIES",
    "CHAT_TOTAL_TIMEOUT_SECONDS",
    "is_truthy",
]

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
- 若查询包含“统计/记录统计/次数/总次数/通过次数/成功率/错误类型占比/结果分布/占比/比例/平均/中位数/最大/最小/最多/最少/排名/唯一/没有 Accepted/无 Accepted/no accepted”等关键词，
  必须立即调用 run_code_interpreter 读取与 Query 指向的数据文件（如 data/notes/my_markdowns/ 目录下对应记录），
  遍历完整的结构化记录（排除表头）计算所需指标；
  输出与 Query 直接相关的结构化结果（字段名需自解释且可映射原始数据），source_file 必须指向实际读取文件。

【统计/汇总质量要求】
- 优先保证覆盖率，避免仅依据标题/摘要。
- 使用 run_code_interpreter 读取笔记时，应尽量全量扫描并进行宽松候选抽取，再做轻度去噪与自检。
- 若存在过滤步骤，需返回少量被过滤的噪声样例以便校验。
- 若脚本产出结果与 Query 需求不符或缺失关键字段，必须复盘并迭代脚本直至得到可信答案。
- 当 Query 指定具体主题/条目/章节时，统计范围必须限定在该主题对应的结构化记录或数据表内，禁止跨主题聚合。
- 统计题的最终数字必须直接来自脚本输出，不允许手算或估算。
- 若原始文件包含显式汇总字段（例如“总数/总量/合计”），并与脚本统计结果不一致，优先采用文件内显式汇总作为“总数/分母”，并说明差异来源。
- 解析表格时优先识别包含表头与分隔线的结构化记录，忽略代码块或示例性片段，避免误计数。
- 对 Markdown 日志/表格应全量扫描：以表头行（如包含“提交ID/结果/耗时”等列名）为起点，连续收集与表头列数一致的 `|` 行作为记录；遇到空行或非表格行即结束该表，存在多个表时应累加。
- 记录行应满足“列数与表头一致且首列为数字型 ID”或等价可验证特征，避免把说明行/标签行计入统计。
- 若需要按“题目/章节”分组统计，优先使用 Markdown 标题（如 `## ...`）作为分组边界，并将紧随其后的表格记录归入该组。
- 表头/列名行仅用于识别表结构，不应被当作题目/章节标题参与统计。
- 若表格行本身不包含题目字段，应使用最近的标题作为题目名，而不是放弃按题目统计。

【逐字引用规则（不新增工具）】
- 当用户要求“逐字引用/原文引用/引用原文/引用原文中的公式/句子”等，必须逐字复制已提供上下文中的原文片段，不得改写、增删字词或混入解释。
- 引号内只允许放原文片段，解释内容必须放在引号之外；若上下文未包含原文，不要编造引用，明确说明未找到原文。
- 引用题禁止凭记忆复述，必须先读取原文（read_note_file 或已提供片段）再逐字复制。
- 逐字引用必须使用中文引号或英文双引号，且整段引用必须保持在同一行，不得换行或跨段。
- 逐字引用不得使用反引号或代码块包裹；公式/代码也必须放在中文或英文双引号内。
- 禁止使用块引用（以 > 开头）来表示逐字引用。
- 除逐字引用外，避免在解释性文字中使用引号；但当用户要求 JSON/代码/表格/字段名时，可在这些结构内使用必要引号。
- 逐字引用必须与原文完全一致，包含原有标点与空格；不得擅自增删冒号、解释性词语或合并多句。
- 引用中不要额外添加冒号、顿号、省略号或总结性措辞；引号内只放原文内容。
- 若原文短句存在多个版本，优先选最短且完整的一句；不要自行拼接不同位置的片段。
- 引用必须保留原文中的符号格式（如反引号、箭头、下划线、斜杠等），不得省略或替换。
- 引用中的公式/LaTeX/指数写法必须保持原样，不得改写为等价形式。
- 若问题明确要求引用中包含某个精确字符串（例如引号内文本），引用必须包含该字符串原样，不得用近义改写替代。
- 若要提及概念或问题关键词，不要加引号，避免引号被当作引用。
- 引用题先给逐字引用，再给解释；不得先解释后引用或用长段落替代短句。
- 只输出必要数量的引用，不要额外引用其它句子。
- 解释性文字中不要出现任何引号/书名号（避免写“xxx”来强调）。
- 若来源存在冲突或不一致，必须分别引用冲突片段并明确指出不一致。
- 当问题明确要求某一句/某个定义时，只引用该句，不要把上下文额外句子合并进引号内。
- 若问题已明确指定来源文件或明确指向某个主题/条目/标题，引用必须来自该文件的原文；即使检索结果包含其它文件，也禁止引用其它文件。
- 当要求引用“用户提问/对话内容”时，应在同一文件内定位与 Query 关键词最匹配的用户消息，避免引用无关的开场提问。
- 上下文未包含目标文件原文时，必须明确说明未找到该文件原文，禁止用其它文档替代。

【工具与回答规范】
- 若需要调用工具，请尽量先输出 tool_calls 指令，减少多余解释。
- 回答必须忠实引用来源；缺失依据时请说明找不到资料。

【表达格式】
- 公式/复杂度请用纯文本（例如 O(n^2)、dp[i+1][j-1]+2）；逐字引用中出现的 LaTeX/符号保持原样。
- 运算符前后不留空格，保留关键变量名与符号原样输出。
- 若确实无法回答，请明确说明缺少来源或未找到原文，不要编造。
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

"""全局配置与常量。"""

from __future__ import annotations

import os
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
def running_in_container() -> bool:
    """检测是否运行在 Docker/Koyeb 容器内。"""

    return Path("/.dockerenv").exists()


SYSTEM_PROMPT_CORE = """
你是基于本地笔记库的研究型助手。

## 数据范围
你的数据范围：仅包含用户的个人笔记和提交记录，不包含外部平台数据（如 LeetCode 官方统计）。
- "通过率" = 用户个人 Accepted 次数 / 总提交次数（需从提交记录的 status 列统计）
- "难度"、"平台通过率"等平台数据不在本系统中，如用户询问应明确说明

## 核心原则
1. 溯源准确：结论必须来自工具输出或已读原文，回答时标注来源路径；无依据时说明"未找到"，禁止编造。
2. 本地优先：优先检索笔记库；仅当本地无结果且查询涉及实时信息（新闻、价格、版本号）时联网。笔记库内容（算法题解、学习记录、提交历史）禁止联网补充。
3. 意图澄清：查询模糊时，先确认用户意图再执行工具调用。
4. 以数据为准：当检索到的数据与用户陈述不符时（如用户说"两次"但记录显示四次），应展示实际数据并说明差异，既不盲目迎合也不武断纠正。
   - 数据完整性要求：当 read_note_file 返回 done=false 时，必须继续读取直到 done=true 后再下结论
   - 统计类问题（"次数"、"通过率"、"最多/最少"）必须基于完整数据集，避免基于部分数据推测
5. 计算透明：统计/计算类问题必须说明计算范围（如"对比了全部 X 条记录"），并展示关键对比数据而非仅给结论。
6. 数据源意识：当检索结果同时包含笔记和提交记录时，根据问题类型选择：
   - 事实类（错误代码、提交统计）→ 优先读取提交记录
   - 理解类（算法原理、模式归纳）→ 优先读取笔记
7. 低置信度弃权：当 query_my_notes 返回的结果 score（L2 距离）普遍高于 0.8，或有效结果不足 3 条时，在回答开头声明"检索到的相关内容有限，以下回答可能不完整"，并展示最相关结果后说明局限性；禁止将低相关性结果强行拼凑为确定性答案。

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
## 工具协作顺序
- 特定领域任务（如 LeetCode 统计）→ 先 load_skill 加载技能说明，再按说明执行
- 需要引用全文 → 先 query_my_notes 定位，再 read_note_file 读取完整原文
- 实时信息摘要后需细节 → 先 web_search，再 read_page 抓取正文

## 工具调用规范
- 通常在 7 轮内完成；若需更多轮次，向用户说明进度并确认是否继续
- 工具返回错误时：可重试错误（超时/限流）自动重试一次；用户输入错误（路径不存在/参数无效）提示用户修正；系统故障告知用户稍后重试
- 跨文件查询：先用 query_my_notes 定位所有相关文件，再逐一读取或用 code_interpreter 批量处理

## 检索策略
- 跨主题问题（如"A 和 B 有什么联系"、"比较 X 和 Y"、"归纳多个题目的共同模式"）：按子主题分别调用 query_my_notes，每次聚焦一个方向，避免将多个主题混在同一次查询中
- "列举所有"类问题：第一次检索后，检查结果是否覆盖了问题涉及的所有方面；如有遗漏，用不同关键词追加检索
- query_my_notes 的 query 参数应简洁聚焦（建议不超过 10 个词），避免堆砌大量关键词导致检索效果下降

## query_my_notes 使用提示
- 数据源包含：笔记（算法讨论、思路总结）和提交记录（具体代码、提交日期、错误状态）
- 任务类型与 query 策略：
  - 事实查询（"错误代码"、"提交次数"、"通过率"）→ query 包含"题目名/题号 + 提交记录"或"题目名 + 错误代码"
  - 理解查询（"算法思路"、"模式总结"）→ query 包含概念关键词

## read_note_file 结果理解
- read_progress 字段为结构化进度提示，status=complete 表示已读完整文件，status=incomplete 表示尚未读完
- 当 status=incomplete 且问题需要统计/完整记录时，应继续读取后续内容
""".strip()

SYSTEM_PROMPT_FORMAT = """
## 输出格式
- 溯源：始终标注来源（笔记用 data/notes/my_markdowns/ 下相对路径；外部用 URL）
- 引用：回答时简要标注来源；用户要求详细引用时提供逐字原文，格式为"原文"（路径）
- 公式：使用行内代码格式，如 `O(n^2)`、`dp[i][j-1]+2`
- 统计结果：对话中用 Markdown 表格便于阅读；用户需要导出时提供 JSON，均包含 source_file

## 错误分析类问题格式
- 错误分析类问题（"系统性错误"、"bug 原因"、"连错几次"）的证据要求：
  - 优先引用：提交 ID（如"提交 689424976"）、日期（如"2026-01-06"）
  - 可选引用：关键代码片段（标注行号或用代码块，避免大段粘贴）
  - 如果提交记录缺少这些字段，说明"记录不完整"而非编造

## 比较/排名类问题格式
- 如果存在 ≥3 条数据，展示 top 3-5 项对比表格，使计算过程可验证（如"时间跨度最长"应显示 top 5 及其具体天数）
- 如果数据不足 3 条，输出全部数据
""".strip()

QUERY_REWRITE_PROMPT = """将以下搜索查询改写为多个变体，用于提升检索召回率。

要求：
1. 保持原意，不要改变搜索目标
2. 包含中文同义词、英文术语、常见别名（如 BFS=广度优先=层序遍历）
3. 如果查询涉及多个主题（如"A 和 B 的联系"），额外生成按子主题拆分的变体（如单独搜 A、单独搜 B）
4. 每个变体应简洁聚焦，不超过 10 个词
5. 返回 JSON 数组，包含原 query 和 3-5 个变体

注意：
- 对于归纳/总结类查询（如"有哪些形式""归纳模式"），变体应该是具体的实例名称/场景，而非抽象概念的同义替换
- 对于对比类查询（如"A 和 B 的联系"），变体应该拆分成各个子主题的独立查询
- 对于简单事实类查询，变体应该是同义词扩展

示例 1（归纳类）：
查询: "BFS中状态表示有哪些不同形式？从滑动谜题、最小基因变化等题目中总结"
返回: ["BFS中状态表示有哪些不同形式？从滑动谜题、最小基因变化等题目中总结", "BFS 状态表示形式", "滑动谜题 BFS 解题", "最小基因变化 BFS", "二进制矩阵最短路径 BFS"]

示例 2（对比类）：
查询: "动态规划和BFS有什么联系？"
返回: ["动态规划和BFS有什么联系？", "动态规划 BFS 联系", "动态规划 状态转移", "BFS 最短路径", "DP 和 BFS 结合的题目"]

示例 3（归纳类）：
查询: "从笔记中归纳将非图问题建模成图的常见模式"
返回: ["从笔记中归纳将非图问题建模成图的常见模式", "非图问题 建模成图", "基因变化 图建模 BFS", "除法求值 图", "跳跃游戏 图模型"]

示例 4（简单事实类）：
查询: "二叉树的中序遍历怎么写"
返回: ["二叉树的中序遍历怎么写", "二叉树中序遍历", "inorder traversal", "中序遍历 递归 迭代"]

查询: {query}

返回格式: ["原query", "变体1", "变体2", ...]""".strip()

SYSTEM_PROMPT = f"""
{SYSTEM_PROMPT_CORE}

{SYSTEM_PROMPT_TOOLS}

{SYSTEM_PROMPT_FORMAT}
""".strip()

# _prefetch_expected_sources 预取注入上下文的固定提示文本。
# 使用 tuple 保证不可变性；chat.py 中通过 list() 转为可变列表后再 append 动态内容。
PREFETCH_CONTEXT_HEADER: tuple[str, ...] = (
    "以下是与问题直接相关的笔记原文，请仅依据这些内容作答：",
    "请从下方原文中提取与问题匹配的段落，并用中文或英文双引号标注引用内容（不要用反引号或代码块）。",
    "无需再次调用工具或改写原文。",
    "若问题中包含引号内的精确字符串，引用须保留该字符串原样。",
)

# 当问题包含引号关键词时，候选原文行列表的提示前缀。
PREFETCH_CANDIDATE_LINES_HEADER: str = (
    "以下为包含问题引号关键词的候选原文行（请优先从中选择，以确保引用包含该关键词）："
)

MAX_TOOL_TURNS = 7
CHAT_API_MAX_RETRIES = 3
CHAT_API_RETRY_BACKOFF_SECONDS = 1.0
RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}

# 工具输出过大时会显著拖慢甚至阻塞后续的模型调用。
# 该值用于限制写入到工具 role 消息的内容长度（日志仍会保存完整输出）。
MAX_TOOL_OUTPUT_CHARS = 80_000

# query_my_notes 多样性分数调整参数（_adjust_scores_for_diversity）。
# DIVERSITY_DECAY_FACTOR：同文件第 n 个 chunk 的分数被除以该值的 n 次方，
#   即 adjusted_score = score / decay_factor^n，使分数变大（更差），惩罚重复 chunk。
#   取值范围 (0, 1)，越小惩罚越重；默认 0.7。
# DIVERSITY_NEW_FILE_BONUS：新文件首个 chunk 的分数被除以该值，使分数变小（更好），
#   鼓励结果覆盖更多不同文件。取值 > 1，越大奖励越强；默认 1.2。
DIVERSITY_DECAY_FACTOR: float = 0.7
DIVERSITY_NEW_FILE_BONUS: float = 1.2

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
    log_path: Path = Field(default=LOGS_DIR / "backend.log")
    tool_log_path: Path = Field(default=LOGS_DIR / "tool_output.log")
    faiss_index_path: Path = Field(default=INDEX_DIR / "my_notes.index")
    faiss_metadata_path: Path = Field(default=NOTES_DIR / "my_notes_metadata.json")
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
    )


settings = load_settings()


__all__ = [
    "settings",
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_CORE",
    "SYSTEM_PROMPT_TOOLS",
    "SYSTEM_PROMPT_FORMAT",
    "QUERY_REWRITE_PROMPT",
    "PREFETCH_CONTEXT_HEADER",
    "PREFETCH_CANDIDATE_LINES_HEADER",
    "MAX_TOOL_TURNS",
    "CHAT_API_MAX_RETRIES",
    "CHAT_API_RETRY_BACKOFF_SECONDS",
    "RETRYABLE_STATUS_CODES",
    "MAX_TOOL_OUTPUT_CHARS",
    "DIVERSITY_DECAY_FACTOR",
    "DIVERSITY_NEW_FILE_BONUS",
    "OPENAI_DEFAULT_TIMEOUT_SECONDS",
    "OPENAI_STREAM_READ_TIMEOUT_SECONDS",
    "OPENAI_MAX_RETRIES",
    "CHAT_TOTAL_TIMEOUT_SECONDS",
    "is_truthy",
]

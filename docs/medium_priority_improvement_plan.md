# 中优先级优化实施方案

本文档为 Second Brain RAG 笔记问答系统的 4 项中优先级优化提供详细实施方案，包含问题描述、具体改动（before/after 代码对比）、改动文件与行号，以及验证方法。

---

## 汇总表格

| 事项 | 改动文件 | 改动类型 | 预期效果 |
|------|----------|----------|----------|
| 1. 系统提示词工具路由去冗余 | `backend/app/core/config.py` | 删减提示词内容 | 减少约 4 行 token 浪费，工具路由责任归一 |
| 2. 增加低置信度弃权机制 | `backend/app/core/config.py` | 新增提示词规则 | 避免低质量检索结果被强行综合，提升答案诚实度 |
| 3. 多样性分数参数提取为命名常量 | `backend/app/core/config.py`、`backend/app/services/tools.py` | 提取魔法数为常量并更新调用处 | 可配置、可测试、可 eval 覆盖 |
| 4. 预取注入上下文提取为常量 | `backend/app/core/config.py`、`backend/app/services/chat.py` | 提取固定字符串为常量 | 便于版本管理和测试覆盖，职责分离 |

---

## 事项 1：系统提示词工具路由去冗余

### 问题描述

`SYSTEM_PROMPT_TOOLS` 中的 `## 工具选择` 节（`config.py` 行 91-97）与各工具 schema 的 `description` 字段存在内容重叠：

- 系统提示词说"统计/计算/大文件处理 → run_code_interpreter"，`MCP_CODE_INTERPRETER_SCHEMA` description 已描述适用场景
- 系统提示词说"定位相关笔记 → query_my_notes；获取完整内容 → read_note_file"，两个工具 schema description 已各自覆盖
- 系统提示词说"实时外部信息 → web_search；需正文 → read_page"，同样在工具 schema 中已有

**设计原则：** 工具 schema 的 `description` 承担"什么情况下调用我"的职责；`SYSTEM_PROMPT_TOOLS` 的工具路由节只保留**跨工具的协作顺序**。

### 改动文件与行号

文件：`backend/app/core/config.py`，行 91-97

**Before:**
```python
SYSTEM_PROMPT_TOOLS = """
## 工具选择
- 统计/计算/大文件处理 → run_code_interpreter
- 定位相关笔记 → query_my_notes；获取完整内容 → read_note_file
- 特定领域（如 LeetCode 统计）→ 先 load_skill 加载技能说明，再按说明执行
- 实时外部信息 → web_search；需正文 → read_page

## 工具调用规范
```

**After:**
```python
SYSTEM_PROMPT_TOOLS = """
## 工具协作顺序
- 特定领域任务（如 LeetCode 统计）→ 先 load_skill 加载技能说明，再按说明执行
- 需要引用全文 → 先 query_my_notes 定位，再 read_note_file 读取完整原文
- 实时信息摘要后需细节 → 先 web_search，再 read_page 抓取正文

## 工具调用规范
```

删除了 `run_code_interpreter` 独立路由（已在 schema description 中）；保留并重写了跨工具协作顺序逻辑（`load_skill` 先后顺序、`query` 后 `read`、`web_search` 后 `read_page`）。

### 验证方法

1. 阅读改动后 `SYSTEM_PROMPT_TOOLS`，确认不再有单个工具独立路由规则
2. 确认每个工具 schema description 已涵盖各自适用场景
3. 运行评估脚本（Q01 笔记定位类、Q14 统计类、Q15 技能类）验证 pass rate 无下降
4. 对比改动前后 `SYSTEM_PROMPT_TOOLS` 字符数，预期减少约 80-100 字符（约 30-40 token）

---

## 事项 2：在 SYSTEM_PROMPT_CORE 中增加低置信度弃权机制

### 问题描述

当前"核心原则"第 1 条处理"未找到"场景，但缺少当检索结果**存在但相关性低**时的明确退出路径。FAISS L2 距离 > 0.8 表示相关性低，此时应主动声明不确定性而非强行综合。

边界说明：
- 第 1 条（已有）：`results: []`，完全未找到 → 告知"未找到"
- 第 7 条（新增）：找到了结果但 score 普遍 > 0.8（相关性低）→ 声明"内容有限，可能不完整"

### 改动文件与行号

文件：`backend/app/core/config.py`，在行 75-76（`6. 数据源意识` 节结束后）插入，位于 `## 重要约束：可准确溯源` 之前

**Before:**
```python
6. 数据源意识：当检索结果同时包含笔记和提交记录时，根据问题类型选择：
   - 事实类（错误代码、提交统计）→ 优先读取提交记录
   - 理解类（算法原理、模式归纳）→ 优先读取笔记

## 重要约束：可准确溯源
```

**After:**
```python
6. 数据源意识：当检索结果同时包含笔记和提交记录时，根据问题类型选择：
   - 事实类（错误代码、提交统计）→ 优先读取提交记录
   - 理解类（算法原理、模式归纳）→ 优先读取笔记
7. 低置信度弃权：当 query_my_notes 返回的结果 score（L2 距离）普遍高于 0.8，或有效结果不足 3 条时，在回答开头声明"检索到的相关内容有限，以下回答可能不完整"，并展示最相关结果后说明局限性；禁止将低相关性结果强行拼凑为确定性答案。

## 重要约束：可准确溯源
```

### 验证方法

1. 构造与笔记库不相关的问题，观察高 L2 距离场景下模型是否主动声明"检索到的相关内容有限"
2. 运行评估集确认正常问答（相关性高）的 pass rate 无下降
3. 阅读改动后 `SYSTEM_PROMPT_CORE`，确认第 7 条与第 1 条语义互补

---

## 事项 3：将多样性分数调整参数提取为命名常量

### 问题描述

`tools.py` 行 666-671 的 `_adjust_scores_for_diversity` 函数中 `decay_factor=0.7` 和 `new_file_bonus=1.2` 是硬编码魔法数，无文档说明来源，无法通过配置修改，也未纳入 eval 覆盖。

实现逻辑：
- `decay_factor=0.7`：同文件第 n 个 chunk → `adjusted_score = score / 0.7^n`，使分数变大（更差），惩罚重复
- `new_file_bonus=1.2`：新文件首个 chunk → `adjusted_score = score / 1.2`，使分数变小（更好），鼓励覆盖更多文件

### 改动文件与行号

- 新增常量：`backend/app/core/config.py`，行 159（`MAX_TOOL_OUTPUT_CHARS` 定义之后）
- 更新 import：`backend/app/services/tools.py`，行 24-31
- 更新函数签名：`backend/app/services/tools.py`，行 666-671

**改动 1：config.py 新增常量（行 159 之后插入）**

Before:
```python
MAX_TOOL_OUTPUT_CHARS = 80_000

# OpenAI / Azure OpenAI 请求超时（秒）。
```

After:
```python
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
```

同时在 `__all__` 末尾追加 `"DIVERSITY_DECAY_FACTOR"` 和 `"DIVERSITY_NEW_FILE_BONUS"`。

**改动 2：tools.py 更新 import（行 24-31）**

Before:
```python
from ..core.config import (
    CHAT_API_MAX_RETRIES,
    CHAT_API_RETRY_BACKOFF_SECONDS,
    RETRYABLE_STATUS_CODES,
    is_truthy,
    running_in_container,
    settings,
)
```

After:
```python
from ..core.config import (
    CHAT_API_MAX_RETRIES,
    CHAT_API_RETRY_BACKOFF_SECONDS,
    DIVERSITY_DECAY_FACTOR,
    DIVERSITY_NEW_FILE_BONUS,
    RETRYABLE_STATUS_CODES,
    is_truthy,
    running_in_container,
    settings,
)
```

**改动 3：tools.py 更新函数签名（行 666-671）**

Before:
```python
def _adjust_scores_for_diversity(
    results: List[tuple[float, int]],
    metadata: List[dict[str, Any]],
    decay_factor: float = 0.7,
    new_file_bonus: float = 1.2,
) -> List[tuple[float, int]]:
```

After:
```python
def _adjust_scores_for_diversity(
    results: List[tuple[float, int]],
    metadata: List[dict[str, Any]],
    decay_factor: float = DIVERSITY_DECAY_FACTOR,
    new_file_bonus: float = DIVERSITY_NEW_FILE_BONUS,
) -> List[tuple[float, int]]:
```

调用处行 751（`_adjust_scores_for_diversity(all_results, metadata)`）无需改动。

### 验证方法

1. `./.venv/bin/python -m pytest -q` 确认无回归
2. Python REPL 验证：`from backend.app.core.config import DIVERSITY_DECAY_FACTOR, DIVERSITY_NEW_FILE_BONUS; assert DIVERSITY_DECAY_FACTOR == 0.7; assert DIVERSITY_NEW_FILE_BONUS == 1.2`
3. `inspect.signature(_adjust_scores_for_diversity)` 确认默认值引用常量名而非字面量

---

## 事项 4：`_prefetch_expected_sources` 注入上下文提取为独立常量

### 问题描述

`chat.py` 中 `_prefetch_expected_sources` 函数（行 259-285）的 `parts` 列表包含硬编码固定提示文本，散落在函数体内，不便于版本管理和测试覆盖。

当前固定文本（命令式语言，高优先级方案已改为协作式，本事项在此基础上进一步常量化）：
- 行 259-264：`parts` 列表的 4 条初始化字符串
- 行 280：候选原文行的提示前缀

### 改动文件与行号

- 新增常量：`backend/app/core/config.py`，`SYSTEM_PROMPT` 定义之后、`MAX_TOOL_TURNS` 之前（约行 151）
- 更新 import：`backend/app/services/chat.py`，行 18-28
- 更新函数体：`backend/app/services/chat.py`，行 259-264 和行 280

**改动 1：config.py 新增常量（行 151 前插入）**

Before:
```python
""".strip()

MAX_TOOL_TURNS = 7
```

After:
```python
""".strip()

# _prefetch_expected_sources 预取注入上下文的固定提示文本。
# 使用 tuple 保证不可变性；chat.py 中通过 list() 转为可变列表后再 append 动态内容。
PREFETCH_CONTEXT_HEADER: tuple[str, ...] = (
    "以下是与问题直接相关的笔记原文，供你准确引用：",
    "请从下方原文中提取与问题匹配的段落，用引号标注引用内容，无需再次调用工具。",
    "引用时请使用中文或英文双引号（不要用反引号或代码块包裹）。",
    "若问题中包含引号内的精确字符串，引用须保留该字符串原样。",
)

# 当问题包含引号关键词时，候选原文行列表的提示前缀。
PREFETCH_CANDIDATE_LINES_HEADER: str = (
    "以下为包含问题引号关键词的候选原文行（请优先从中选择，以确保引用包含该关键词）："
)

MAX_TOOL_TURNS = 7
```

同时在 `__all__` 末尾追加 `"PREFETCH_CONTEXT_HEADER"` 和 `"PREFETCH_CANDIDATE_LINES_HEADER"`。

**改动 2：chat.py 更新 import（行 18-28）**

Before:
```python
from ..core.config import (
    CHAT_API_MAX_RETRIES,
    CHAT_API_RETRY_BACKOFF_SECONDS,
    CHAT_TOTAL_TIMEOUT_SECONDS,
    MAX_TOOL_TURNS,
    MAX_TOOL_OUTPUT_CHARS,
    OPENAI_DEFAULT_TIMEOUT_SECONDS,
    OPENAI_STREAM_READ_TIMEOUT_SECONDS,
    SYSTEM_PROMPT,
    settings,
)
```

After:
```python
from ..core.config import (
    CHAT_API_MAX_RETRIES,
    CHAT_API_RETRY_BACKOFF_SECONDS,
    CHAT_TOTAL_TIMEOUT_SECONDS,
    MAX_TOOL_TURNS,
    MAX_TOOL_OUTPUT_CHARS,
    OPENAI_DEFAULT_TIMEOUT_SECONDS,
    OPENAI_STREAM_READ_TIMEOUT_SECONDS,
    PREFETCH_CANDIDATE_LINES_HEADER,
    PREFETCH_CONTEXT_HEADER,
    SYSTEM_PROMPT,
    settings,
)
```

**改动 3：chat.py 更新函数体（2 处改动）**

Before（行 259-264 和行 280）:
```python
    parts = [
        "以下是评估模式强制注入的笔记内容，回答必须仅依据这些内容：",
        "该任务为逐字引用，请直接从下方原文复制并用引号包裹，避免再调用工具或改写。",
        "逐字引用必须使用中文或英文双引号，不要用反引号或代码块包裹。",
        "若问题中出现引号内的精确字符串，引用必须包含该字符串原样，且不得用近义改写替代。",
    ]
    ...
        parts.append("以下为包含问题引号关键词的候选原文行（必须从中选择，确保包含该关键词）：")
```

After:
```python
    parts: List[str] = list(PREFETCH_CONTEXT_HEADER)
    ...
        parts.append(PREFETCH_CANDIDATE_LINES_HEADER)
```

动态部分（`for line in candidate_lines[:12]`、`for source, content in snippets`）保持不变。

### 验证方法

1. `./.venv/bin/python -m pytest -q` 确认无回归
2. Python REPL 验证：`from backend.app.core.config import PREFETCH_CONTEXT_HEADER, PREFETCH_CANDIDATE_LINES_HEADER; assert len(PREFETCH_CONTEXT_HEADER) == 4; assert "候选原文行" in PREFETCH_CANDIDATE_LINES_HEADER`
3. 构造包含引号关键词的 query 调用 `_prefetch_expected_sources`，对比改动前后返回字符串完全一致
4. 检查改动后函数体，确认不再有硬编码固定提示字符串

---

## 实施顺序建议

| 顺序 | 事项 | 理由 |
|------|------|------|
| 1 | 事项 3（多样性参数常量化） | 最小改动，无行为变更，风险最低 |
| 2 | 事项 4（预取上下文常量化） | 纯重构，无行为变更，涉及两个文件 |
| 3 | 事项 1（提示词去冗余） | 有 token 节省效果，需运行评估集验证无回归 |
| 4 | 事项 2（低置信度弃权机制） | 有行为变更，需在低相关性场景专项测试，建议最后实施 |

---

生成日期：2026-02-24

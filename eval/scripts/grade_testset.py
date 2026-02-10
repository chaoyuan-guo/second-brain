#!/usr/bin/env python3
"""Grader for evaluation testsets with partial scoring support.

部分得分机制：
- retrieval_score: 检索得分 (0~1)
- actual_retrieval_score: 实际检索得分 (0~1)
- content_score: 内容得分 (0~1)
- citation_score: 引用得分 (0~1)
- total_score: 加权总分 (0~1)

Usage:
  python eval/scripts/grade_testset.py --answers path/to/answers.json
  python eval/scripts/grade_testset.py --testset eval/testsets/testset.json --answers answers.json --output report.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from eval.scripts.eval_config import (
    get_config,
    get_pass_threshold,
    get_scoring_weights,
    load_eval_config,
)

DEFAULT_API_BASE_URL = "https://space.ai-builders.com/backend/v1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

# 笔记目录路径
NOTES_DIR = Path(__file__).resolve().parents[2] / "data" / "notes" / "my_markdowns"

# 预计算统计数据路径
PRECOMPUTED_STATS_PATH = Path(__file__).resolve().parents[1] / "config" / "precomputed_stats.json"

# 源文档内容缓存
_source_content_cache: Dict[str, str] = {}

# 全局 embedding 缓存
_embedding_cache: Dict[str, List[float]] = {}

# query rewrite 缓存（避免重复 LLM 调用）
_rewrite_cache: Dict[str, List[str]] = {}

# 预计算统计数据缓存
_precomputed_stats: Optional[Dict[str, Any]] = None


def load_precomputed_stats() -> Dict[str, Any]:
    """加载预计算的统计数据。"""
    global _precomputed_stats
    if _precomputed_stats is not None:
        return _precomputed_stats

    if PRECOMPUTED_STATS_PATH.exists():
        try:
            _precomputed_stats = json.loads(PRECOMPUTED_STATS_PATH.read_text(encoding="utf-8"))
        except Exception:
            _precomputed_stats = {}
    else:
        _precomputed_stats = {}
    return _precomputed_stats


def resolve_dynamic_value(value: Any) -> Any:
    """解析动态值引用（如 $stats.xxx）。

    支持格式：
    - $stats.leetcode_submissions.submission_results.Accepted
    - $stats.leetcode_submissions.pass_rate

    Args:
        value: 值，可能是动态引用字符串

    Returns:
        解析后的实际值，如果无法解析则返回原值
    """
    if not isinstance(value, str) or not value.startswith("$stats."):
        return value

    # 解析 $stats.xxx.yyy.zzz 格式
    path = value[7:]  # 去掉 "$stats."
    parts = path.split(".")

    stats = load_precomputed_stats()
    current = stats

    try:
        for part in parts:
            if isinstance(current, dict):
                current = current[part]
            else:
                return value  # 无法继续解析，返回原值
        return current
    except (KeyError, TypeError):
        return value  # 解析失败，返回原值


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_unknown(answer: str) -> bool:
    """检测答案是否表示"不知道"。

    改进：排除"虽然没有...但是..."这类结构，避免误判。
    """
    config = get_config()
    unknown_patterns = config.unknown_detection.patterns
    exclusion_patterns = config.unknown_detection.exclusion_patterns

    # 先检查排除模式
    for pat in exclusion_patterns:
        if re.search(pat, answer):
            return False
    # 再检查 unknown 模式
    for pat in unknown_patterns:
        if re.search(pat, answer):
            return True
    return False


def is_unknown_with_confidence(answer: str) -> tuple:
    """检测答案是否表示"不知道"，并返回置信度。

    Returns:
        (is_unknown: bool, confidence: float, matched_pattern: str)
    """
    config = get_config()
    unknown_patterns = config.unknown_detection.patterns
    exclusion_patterns = config.unknown_detection.exclusion_patterns

    # 先检查排除模式
    for pat in exclusion_patterns:
        if re.search(pat, answer):
            return False, 0.0, ""

    # 检查 unknown 模式
    for pat in unknown_patterns:
        match = re.search(pat, answer)
        if match:
            # 如果匹配在答案开头附近（前100字符），置信度更高
            confidence = 0.9 if match.start() < 100 else 0.7
            return True, confidence, pat
    return False, 0.0, ""


def strip_markdown(text: str) -> str:
    """剥离 Markdown 格式标记。"""
    # 移除行内代码反引号
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # 移除加粗/斜体标记
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    return text


def normalize_math_expr(text: str) -> str:
    """归一化数学表达式中的空格。"""
    # 移除运算符周围空格: "2^(h+1) - 1" -> "2^(h+1)-1"
    text = re.sub(r'\s*([+\-*/^=])\s*', r'\1', text)
    return text


def normalize_text(text: str) -> str:
    """增强的文本归一化。"""
    text = strip_markdown(text)
    text = normalize_math_expr(text)
    return re.sub(r"\s+", " ", text).strip().lower()


def contains_normalized(
    haystack: str,
    needle: str,
    synonyms: Optional[List[str]] = None
) -> Tuple[bool, str]:
    """检查文本是否包含目标或其同义词。

    Args:
        haystack: 待搜索的文本
        needle: 目标文本
        synonyms: 题目级同义词列表（可选）

    Returns:
        (is_match: bool, matched_term: str)
    """
    if not haystack or not needle:
        return False, ""

    normalized_haystack = normalize_text(haystack)

    # 1. 精确匹配
    if normalize_text(needle) in normalized_haystack:
        return True, needle

    # 2. 正则匹配（如果 needle 包含正则元字符）
    if re.search(r'[\\^$.*+?{}[\]|()]', needle):
        try:
            if re.search(needle, haystack, re.IGNORECASE):
                return True, f"regex:{needle}"
        except re.error:
            pass

    # 3. 题目级同义词匹配
    if synonyms:
        for syn in synonyms:
            if normalize_text(syn) in normalized_haystack:
                return True, syn

    # 4. 全局同义词匹配
    config = get_config()
    if hasattr(config, 'synonyms') and needle in config.synonyms.global_synonyms:
        for global_syn in config.synonyms.global_synonyms[needle]:
            if normalize_text(global_syn) in normalized_haystack:
                return True, global_syn

    # 5. 语义类别匹配
    if hasattr(config, 'semantic_categories') and config.semantic_categories.categories:
        for category, patterns in config.semantic_categories.categories.items():
            if needle in patterns:
                for pattern in patterns:
                    if normalize_text(pattern) in normalized_haystack:
                        return True, f"semantic:{category}:{pattern}"

    return False, ""


def extract_quotes(answer: str) -> List[str]:
    """从答案中提取引用内容。

    支持以下格式：
    1. 中英文引号：""、''、「」、『』、""、''
    2. Markdown 引用块：> 开头的行
    3. 代码块中的引用（```...```）
    """
    quotes: List[str] = []

    # 1. 各种引号格式
    quote_patterns = [
        r"\u201c([^\u201d]{4,})\u201d",   # 中文双引号 ""
        r"\"([^\"]{4,})\"",                # 英文双引号 ""
        r"\u2018([^\u2019]{4,})\u2019",   # 中文单引号 ''
        r"'([^']{4,})'",                   # 英文单引号 ''
        r"\u300c([^\u300d]{4,})\u300d",   # 日式引号 「」
        r"\u300e([^\u300f]{4,})\u300f",   # 日式双引号 『』
        r"\u3010([^\u3011]{4,})\u3011",   # 方括号 【】
    ]
    for pat in quote_patterns:
        for match in re.findall(pat, answer):
            cleaned = match.strip()
            if cleaned:
                quotes.append(cleaned)

    # 2. Markdown 引用块（> 开头的行）
    blockquote_pattern = r"^>\s*(.+)$"
    for match in re.findall(blockquote_pattern, answer, re.MULTILINE):
        cleaned = match.strip()
        if len(cleaned) >= 4:
            quotes.append(cleaned)

    # 3. 代码块内容（作为引用的一种形式）
    code_block_pattern = r"```[\w]*\n([\s\S]*?)```"
    for match in re.findall(code_block_pattern, answer):
        cleaned = match.strip()
        if len(cleaned) >= 10:  # 代码块至少10字符才算有效引用
            quotes.append(cleaned)

    return quotes


def load_source_content(source_path: str) -> str:
    """加载源文档内容，使用缓存避免重复读取。"""
    normalized = normalize_source_path(source_path)
    if normalized in _source_content_cache:
        return _source_content_cache[normalized]

    # 尝试从笔记目录加载
    for file in NOTES_DIR.glob("*.md"):
        if normalize_source_path(file.name) == normalized:
            try:
                content = file.read_text(encoding="utf-8")
                _source_content_cache[normalized] = content
                return content
            except Exception:
                pass

    _source_content_cache[normalized] = ""
    return ""


def get_quote_context(quote: str, source_content: str, window_size: int = 100) -> Optional[Dict]:
    """获取引用在源文档中的上下文。

    Args:
        quote: 引用内容
        source_content: 源文档内容
        window_size: 上下文窗口大小（字符数）

    Returns:
        {
            "before": str,      # 引用前的文本
            "after": str,       # 引用后的文本
            "position": int,    # 引用在文档中的位置
            "heading": str      # 所属标题（如果有）
        }
        如果找不到引用则返回 None
    """
    normalized_quote = normalize_text(quote)
    normalized_source = normalize_text(source_content)

    # 查找引用位置
    pos = normalized_source.find(normalized_quote)
    if pos == -1:
        return None

    # 提取上下文
    before_start = max(0, pos - window_size)
    after_end = min(len(normalized_source), pos + len(normalized_quote) + window_size)

    before = normalized_source[before_start:pos]
    after = normalized_source[pos + len(normalized_quote):after_end]

    # 尝试提取所属标题（在原始文档中查找）
    heading = ""
    lines_before_pos = source_content[:pos].split('\n')
    for line in reversed(lines_before_pos):
        if line.strip().startswith('#'):
            heading = line.strip()
            break

    return {
        "before": before.strip(),
        "after": after.strip(),
        "position": pos,
        "heading": heading
    }


def quote_matches_source(quote: str, source_path: str) -> Tuple[bool, Optional[Dict]]:
    """检查引用内容是否在源文档中存在。

    使用模糊匹配，允许空白字符差异。
    增强版：返回更多匹配信息，包括上下文。

    Returns:
        (是否匹配, 上下文信息字典)
        上下文信息包含: before, after, position, heading
    """
    source_content = load_source_content(source_path)
    if not source_content:
        return False, None

    # 规范化后比较
    normalized_quote = normalize_text(quote)
    normalized_source = normalize_text(source_content)

    # 直接包含检查
    if normalized_quote in normalized_source:
        context = get_quote_context(quote, source_content)
        return True, context

    # 对于较短的引用，尝试更宽松的匹配
    quote_len = len(normalized_quote)
    if quote_len < 50:
        # 分词后检查关键词
        quote_words = set(normalized_quote.split())
        if len(quote_words) >= 3:
            # 动态阈值：越短的引用要求越高
            if quote_len < 20:
                threshold = 0.95
            elif quote_len < 50:
                threshold = 0.85
            else:
                threshold = 0.8

            # 至少达到阈值的词在源文档中
            matches = sum(1 for w in quote_words if w in normalized_source)
            if matches / len(quote_words) >= threshold:
                # 关键词匹配成功，但没有精确位置信息
                return True, None

    return False, None


def evaluate_quote_relevance(quote: str, question_query: str) -> float:
    """评估引用与问题的相关性。

    Args:
        quote: 引用内容
        question_query: 问题文本

    Returns:
        0.0-1.0 的相关性分数
    """
    if not question_query.strip():
        return 1.0  # 如果没有问题文本，默认相关

    # 规范化
    normalized_quote = normalize_text(quote)
    normalized_query = normalize_text(question_query)

    # 对于中文文本，使用字符级别的 n-gram 匹配（bigram）
    def extract_ngrams(text: str, n: int = 2) -> set:
        """提取 n-gram。"""
        if len(text) < n:
            return {text}
        return {text[i:i+n] for i in range(len(text) - n + 1)}

    # 提取关键词（英文按空格分词，中文使用 bigram）
    stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can'}

    # 先尝试空格分词（适用于英文或分词后的中文）
    quote_tokens = normalized_quote.split()
    query_tokens = normalized_query.split()

    # 如果分词结果太少（说明是中文未分词），使用 n-gram
    if len(query_tokens) <= 2:
        quote_ngrams = extract_ngrams(normalized_quote, n=2)
        query_ngrams = extract_ngrams(normalized_query, n=2)

        # 去除停用词
        quote_ngrams = {ng for ng in quote_ngrams if not any(sw in ng for sw in stop_words)}
        query_ngrams = {ng for ng in query_ngrams if not any(sw in ng for sw in stop_words)}

        if not query_ngrams:
            return 1.0

        # 计算交集
        common_ngrams = quote_ngrams & query_ngrams
        relevance = len(common_ngrams) / len(query_ngrams)

        # 也考虑引用中包含的 n-gram 比例（给予较小权重）
        if len(quote_ngrams) > 0:
            quote_coverage = len(common_ngrams) / len(quote_ngrams)
            relevance = 0.7 * relevance + 0.3 * quote_coverage
    else:
        # 英文模式：按词匹配
        quote_words = set(w for w in quote_tokens if w not in stop_words and len(w) > 1)
        query_words = set(w for w in query_tokens if w not in stop_words and len(w) > 1)

        if not query_words:
            return 1.0

        # 计算交集
        common_words = quote_words & query_words
        relevance = len(common_words) / len(query_words)

        # 也考虑引用中包含的关键词比例（给予较小权重）
        if len(quote_words) > 0:
            quote_coverage = len(common_words) / len(quote_words)
            relevance = 0.7 * relevance + 0.3 * quote_coverage

    return min(1.0, relevance)


def extract_source_claims(answer: str) -> List[Dict[str, str]]:
    """从答案中提取来源声明。

    提取模式：
    - "在 xxx.md 中提到/记录/有"
    - "来源: xxx.md"
    - "根据 xxx.md"
    - "xxx.md 中记载"

    Returns:
        List[{"source": str, "context": str}]
    """
    patterns = [
        r'在\s*[`「\[]?([^`」\]\s]+\.md)[`」\]]?\s*中[有提到记录]',
        r'来源[：:]\s*[`「\[]?([^`」\]\s]+\.md)',
        r'根据\s*[`「\[]?([^`」\]\s]+\.md)',
        r'[`「\[]?([^`」\]\s]+\.md)[`」\]]?\s*中[记载显示]',
    ]
    claims = []
    for pattern in patterns:
        for match in re.finditer(pattern, answer):
            claims.append({
                'source': match.group(1),
                'context': answer[max(0, match.start()-50):match.end()+100]
            })
    return claims


def verify_negative_answer(
    question: Dict[str, Any],
    answer: str,
    tool_events: Optional[List[Dict[str, Any]]] = None
) -> Optional[Dict[str, Any]]:
    """验证 Negative 题目的答案，检测虚假来源声明。

    对于 allow_unknown=True 的题目，检查：
    1. 答案是否声称了某个来源
    2. 如果声称了来源，验证该来源是否真的包含相关信息

    Args:
        question: 题目配置
        answer: 答案文本
        tool_events: 工具调用事件（可选）

    Returns:
        如果检测到虚假来源声明，返回评估结果；否则返回 None 使用标准评估
    """
    if not question.get("allow_unknown"):
        return None

    claims = extract_source_claims(answer)
    if not claims:
        return None  # 没有来源声明，使用标准评估

    # 验证每个来源声明
    for claim in claims:
        source = claim['source']
        source_path = NOTES_DIR / source

        # 检查声称的来源是否存在
        if not source_path.exists():
            # 来源文件不存在，但可能是路径问题，尝试模糊匹配
            found = False
            for file in NOTES_DIR.glob("*.md"):
                if normalize_source_path(file.name) == normalize_source_path(source):
                    found = True
                    source_path = file
                    break

            if not found:
                return {
                    "score": 0.0,
                    "behavior": "false_citation",
                    "matched_must_have": [],
                    "matched_should_have": [],
                    "matched_evidence": [],
                    "is_unknown_answer": False,
                    "details": f"声称的来源不存在: {source}"
                }

    return None  # 来源存在，使用标准评估


def normalize_source_path(value: str) -> str:
    return Path(value).name.lower()


def load_retrieval_assets() -> Tuple[Any, List[Dict[str, Any]]]:
    try:
        import faiss  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("faiss is required for recall@k") from exc

    root = Path(__file__).resolve().parents[2]
    index_path = root / "data" / "indexes" / "my_notes.index"
    metadata_path = root / "data" / "notes" / "my_notes_metadata.json"
    if not index_path.exists() or not metadata_path.exists():
        raise RuntimeError("Index or metadata file missing for recall@k")
    index = faiss.read_index(str(index_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return index, metadata


def _get_openai_client():
    """获取 OpenAI 客户端（单例模式）。"""
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass

    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("openai package is required for recall@k") from exc

    api_key = os.getenv("SUPER_MIND_API_KEY") or os.getenv("AI_BUILDER_TOKEN")
    if not api_key:
        raise RuntimeError("Missing API key for embeddings")
    base_url = os.getenv("SUPER_MIND_API_BASE_URL", DEFAULT_API_BASE_URL)
    return OpenAI(api_key=api_key, base_url=base_url)


# 复用 tools.py 中的 QUERY_REWRITE_PROMPT
QUERY_REWRITE_PROMPT = """将以下搜索查询改写为多个变体，用于提升检索召回率。

要求：
1. 保持原意，不要改变搜索目标
2. 包含中文同义词、英文术语、常见别名（如 BFS=广度优先=层序遍历）
3. 如果查询涉及多个主题（如"A 和 B 的联系"），额外生成按子主题拆分的变体（如单独搜 A、单独搜 B）
4. 每个变体应简洁聚焦，不超过 10 个词
5. 返回 JSON 数组，包含原 query 和 3-5 个变体

查询: {query}

返回格式: ["原query", "变体1", "变体2", ...]"""

DEFAULT_CHAT_MODEL = "gpt-5"


def rewrite_query(query: str) -> List[str]:
    """用 LLM 生成 query 变体列表，与 tools.py 中逻辑一致。"""
    if query in _rewrite_cache:
        return _rewrite_cache[query]
    try:
        client = _get_openai_client()
        model = os.getenv("SUPER_MIND_CHAT_MODEL", DEFAULT_CHAT_MODEL)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": QUERY_REWRITE_PROMPT.format(query=query)}],
            temperature=0.3,
            max_completion_tokens=200,
        )
        content = (response.choices[0].message.content or "").strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        variants = json.loads(content)
        if not isinstance(variants, list):
            _rewrite_cache[query] = [query]
            return [query]
        if query not in variants:
            variants.insert(0, query)
        result = variants[:5]
        _rewrite_cache[query] = result
        return result
    except Exception:
        _rewrite_cache[query] = [query]
        return [query]


def embed_query(query: str) -> List[float]:
    """获取单个文本的 embedding。"""
    client = _get_openai_client()
    response = client.embeddings.create(model=DEFAULT_EMBEDDING_MODEL, input=query)
    if not response.data:
        raise RuntimeError("Empty embedding response")
    return response.data[0].embedding


def embed_batch(texts: List[str], batch_size: int = 100) -> Dict[str, List[float]]:
    """批量获取多个文本的 embedding。

    Args:
        texts: 文本列表
        batch_size: 每批次最大文本数（OpenAI 限制）

    Returns:
        {text: embedding} 字典
    """
    if not texts:
        return {}

    client = _get_openai_client()
    result: Dict[str, List[float]] = {}

    # 去重并过滤已缓存的
    unique_texts = [t for t in set(texts) if t and t not in _embedding_cache]

    if not unique_texts:
        return {t: _embedding_cache[t] for t in texts if t in _embedding_cache}

    # 分批处理
    for i in range(0, len(unique_texts), batch_size):
        batch = unique_texts[i:i + batch_size]
        try:
            response = client.embeddings.create(model=DEFAULT_EMBEDDING_MODEL, input=batch)
            for j, data in enumerate(response.data):
                text = batch[j]
                embedding = data.embedding
                _embedding_cache[text] = embedding
                result[text] = embedding
        except Exception as exc:
            # 批量失败时回退到单个调用
            print(f"⚠ 批量 embedding 失败，回退到串行: {exc}", file=sys.stderr)
            for text in batch:
                try:
                    if text not in _embedding_cache:
                        _embedding_cache[text] = embed_query(text)
                    result[text] = _embedding_cache[text]
                except Exception:
                    pass

    # 补充已缓存的结果
    for t in texts:
        if t in _embedding_cache and t not in result:
            result[t] = _embedding_cache[t]

    return result


def get_cached_embedding(text: str) -> List[float]:
    """获取文本的 embedding，使用缓存避免重复调用 API。"""
    if text not in _embedding_cache:
        _embedding_cache[text] = embed_query(text)
    return _embedding_cache[text]


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """计算两段文本的语义相似度（余弦相似度）。"""
    import numpy as np  # type: ignore

    try:
        emb1 = np.array(get_cached_embedding(text1), dtype="float32")
        emb2 = np.array(get_cached_embedding(text2), dtype="float32")
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
    except Exception:
        return 0.0


def semantic_match(
    answer: str,
    target: str,
    threshold: Optional[float] = None
) -> Tuple[bool, float, str]:
    """检查答案是否语义匹配目标文本。

    Args:
        answer: 答案文本
        target: 目标文本
        threshold: 相似度阈值，默认从配置获取

    Returns:
        (is_match: bool, similarity: float, match_type: str)
        match_type: "exact" | "semantic" | "none"
    """
    if threshold is None:
        threshold = get_config().thresholds.semantic_similarity_threshold

    # 先精确匹配
    if contains_normalized(answer, target):
        return True, 1.0, "exact"

    # 分句找最相似的
    sentences = re.split(r'[。.!?！？\n]', answer)
    max_sim = 0.0
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 5:
            continue
        sim = compute_semantic_similarity(sent, target)
        max_sim = max(max_sim, sim)

    if max_sim >= threshold:
        return True, max_sim, "semantic"
    return False, max_sim, "none"


def precompute_embeddings(questions: List[Dict[str, Any]]) -> None:
    """预计算所有评估文本的 embedding 并缓存（批量调用）。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    texts = set()

    # 并发预热 rewrite_query 缓存
    queries_to_rewrite = []
    for q in questions:
        query = q.get("query", "")
        if query and len(query) >= 5 and query not in _rewrite_cache:
            queries_to_rewrite.append(query)

    if queries_to_rewrite:
        print(f"  并发生成 {len(queries_to_rewrite)} 个 query 变体...", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(rewrite_query, q): q for q in queries_to_rewrite}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    pass

    for q in questions:
        # 收集 query（用于 recall@k 计算）
        query = q.get("query", "")
        if query and len(query) >= 5:
            texts.add(query)
            # 预计算变体 embedding（用于多路召回 recall@k）
            for variant in rewrite_query(query):
                if variant and len(variant) >= 3:
                    texts.add(variant)

        # 收集 content_rules 中的文本（用于语义匹配）
        rules = q.get("content_rules", {})
        for item in rules.get("must_have", []) + rules.get("evidence", []):
            text = item.get("text") if isinstance(item, dict) else str(item)
            if text and len(text) >= 5:
                texts.add(text)

    # 过滤已缓存的
    texts_to_embed = [t for t in texts if t not in _embedding_cache]

    if not texts_to_embed:
        return

    # 批量调用 API
    print(f"  批量计算 {len(texts_to_embed)} 个 embedding...", file=sys.stderr)
    embed_batch(texts_to_embed)


def compute_recall_at_k(
    questions: List[Dict[str, Any]],
    k_values: List[int],
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    import numpy as np  # type: ignore

    index, metadata = load_retrieval_assets()
    k_max = max(k_values) if k_values else 0
    results: List[Dict[str, Any]] = []
    agg: Dict[str, Dict[str, float]] = {}

    for question in questions:
        expected_sources = [normalize_source_path(s) for s in (question.get("expected_sources") or question.get("sources") or [])]
        case_type = str(question.get("case_type") or "").strip().lower()
        q_type = str(question.get("type") or "").strip().lower()
        allow_any_source = case_type == "multi_source" or q_type == "multi_doc"
        if not expected_sources or k_max == 0:
            results.append({"id": question.get("id"), "recall": {}})
            continue
        query = str(question.get("query") or "")
        if not query:
            results.append({"id": question.get("id"), "recall": {}})
            continue
        # 使用全局缓存（已在 precompute_embeddings 中批量计算）
        query_embedding = get_cached_embedding(query)
        embedding = np.array(query_embedding, dtype="float32").reshape(1, -1)
        distances, indices = index.search(embedding, min(k_max, index.ntotal))
        retrieved_sources: List[str] = []
        for idx in indices[0]:
            if idx == -1 or idx >= len(metadata):
                continue
            record = metadata[idx]
            source_path = record.get("source_path")
            if source_path:
                retrieved_sources.append(normalize_source_path(str(source_path)))

        recall_map: Dict[str, float] = {}
        for k in k_values:
            top_k = retrieved_sources[:k]
            if allow_any_source:
                recall_value = 1.0 if any(s in top_k for s in expected_sources) else 0.0
            else:
                hits = sum(1 for s in expected_sources if s in top_k)
                recall_value = hits / max(len(expected_sources), 1)
            recall_map[str(k)] = recall_value
            agg.setdefault(str(k), {"sum": 0.0, "count": 0.0})
            agg[str(k)]["sum"] += recall_value
            agg[str(k)]["count"] += 1.0
        results.append({"id": question.get("id"), "recall": recall_map})

    summary: Dict[str, Dict[str, float]] = {}
    for k, values in agg.items():
        count = values["count"]
        summary[k] = {
            "mean_recall": (values["sum"] / count) if count else 0.0,
            "count": count,
        }
    return summary, results


def compute_multipath_recall_at_k(
    questions: List[Dict[str, Any]],
    k_values: List[int],
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    """多路召回版 recall@k，模拟后端实际检索行为。

    与单路召回的区别：
    1. 为每个 query 生成 3-5 个变体（使用 rewrite_query）
    2. 每个变体独立检索 top-k*2 条结果
    3. 合并所有结果，去重（按 source_path + chunk_index）
    4. 按距离排序，取最终的 top-k

    这与后端 tools.py:query_my_notes 的实现保持一致。
    """
    import numpy as np  # type: ignore

    index, metadata = load_retrieval_assets()
    k_max = max(k_values) if k_values else 0
    results: List[Dict[str, Any]] = []
    agg: Dict[str, Dict[str, float]] = {}

    for question in questions:
        expected_sources = [normalize_source_path(s) for s in (question.get("expected_sources") or question.get("sources") or [])]
        case_type = str(question.get("case_type") or "").strip().lower()
        q_type = str(question.get("type") or "").strip().lower()
        allow_any_source = case_type == "multi_source" or q_type == "multi_doc"

        if not expected_sources or k_max == 0:
            results.append({"id": question.get("id"), "recall": {}})
            continue

        query = str(question.get("query") or "")
        if not query:
            results.append({"id": question.get("id"), "recall": {}})
            continue

        # 多路召回：生成变体 → 各自检索 → 合并去重
        variants = rewrite_query(query)
        all_results: List[Tuple[float, int]] = []
        seen_chunks: set = set()

        for variant in variants:
            emb = np.array(get_cached_embedding(variant), dtype="float32").reshape(1, -1)
            distances, indices = index.search(emb, min(k_max * 2, index.ntotal))
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1 or idx >= len(metadata):
                    continue
                chunk_key = (metadata[idx].get("source_path"), metadata[idx].get("chunk_index"))
                if chunk_key in seen_chunks:
                    continue
                seen_chunks.add(chunk_key)
                all_results.append((float(dist), int(idx)))

        all_results.sort(key=lambda x: x[0])

        # 从合并结果中提取 source_path
        retrieved_sources: List[str] = []
        for _, idx in all_results:
            sp = metadata[idx].get("source_path")
            if sp:
                retrieved_sources.append(normalize_source_path(str(sp)))

        # 计算各 k 值的 recall
        recall_map: Dict[str, float] = {}
        for k in k_values:
            top_k = retrieved_sources[:k]
            if allow_any_source:
                recall_value = 1.0 if any(s in top_k for s in expected_sources) else 0.0
            else:
                hits = sum(1 for s in expected_sources if s in top_k)
                recall_value = hits / max(len(expected_sources), 1)
            recall_map[str(k)] = recall_value
            agg.setdefault(str(k), {"sum": 0.0, "count": 0.0})
            agg[str(k)]["sum"] += recall_value
            agg[str(k)]["count"] += 1.0
        results.append({"id": question.get("id"), "recall": recall_map})

    summary: Dict[str, Dict[str, float]] = {}
    for k, values in agg.items():
        count = values["count"]
        summary[k] = {
            "mean_recall": (values["sum"] / count) if count else 0.0,
            "count": count,
        }
    return summary, results


def compute_actual_retrieval_score(
    question: Dict[str, Any],
    tool_events: Optional[List[Dict[str, Any]]],
) -> Optional[float]:
    """基于模型实际读取的文件计算召回率。"""

    expected_sources = [
        normalize_source_path(s)
        for s in (question.get("expected_sources") or question.get("sources") or [])
    ]
    if not expected_sources:
        return 1.0
    if tool_events is None:
        return None

    case_type = str(question.get("case_type") or "").strip().lower()
    q_type = str(question.get("type") or "").strip().lower()
    allow_any_source = case_type == "multi_source" or q_type == "multi_doc"

    actual_sources = set()
    for event in tool_events:
        if event.get("tool_name") != "read_note_file":
            continue
        if event.get("stage") != "end":
            continue
        args = event.get("arguments") or {}
        if not isinstance(args, dict):
            continue
        file_path = args.get("file_path") or args.get("path") or ""
        if file_path:
            actual_sources.add(normalize_source_path(str(file_path)))

    if not actual_sources:
        return 0.0

    if allow_any_source:
        return 1.0 if any(s in actual_sources for s in expected_sources) else 0.0

    hits = sum(1 for s in expected_sources if s in actual_sources)
    return hits / len(expected_sources)


def compute_retrieval_metrics(
    question: Dict[str, Any],
    tool_events: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """计算检索的 precision 和 recall。

    Args:
        question: 题目配置
        tool_events: 工具调用事件列表

    Returns:
        {
            "precision": float (0~1),
            "recall": float (0~1),
            "f1": float (0~1),
            "retrieved_sources": List[str],
            "expected_sources": List[str],
            "relevant_retrieved": List[str],
            "missed_sources": List[str],
            "irrelevant_sources": List[str]
        }
    """
    expected_sources = set(
        normalize_source_path(s)
        for s in (question.get("expected_sources") or question.get("sources") or [])
    )

    if not expected_sources or tool_events is None:
        return {
            "precision": 1.0 if not expected_sources else 0.0,
            "recall": 1.0 if not expected_sources else 0.0,
            "f1": 1.0 if not expected_sources else 0.0,
            "retrieved_sources": [],
            "expected_sources": list(expected_sources),
            "relevant_retrieved": [],
            "missed_sources": list(expected_sources),
            "irrelevant_sources": []
        }

    # 从工具事件中收集检索到的源文档
    retrieved_sources = set()
    for event in tool_events:
        # 从 query_my_notes 工具的结果收集
        if event.get("tool_name") == "query_my_notes" and event.get("stage") == "end":
            sources = event.get("retrieved_sources", [])
            for src in sources:
                if isinstance(src, str):
                    retrieved_sources.add(normalize_source_path(src))
                elif isinstance(src, dict) and "source_path" in src:
                    retrieved_sources.add(normalize_source_path(src["source_path"]))

        # 也从 read_note_file 收集实际读取的文件
        if event.get("tool_name") == "read_note_file" and event.get("stage") == "end":
            args = event.get("arguments") or {}
            if isinstance(args, dict):
                file_path = args.get("file_path") or args.get("path") or ""
                if file_path:
                    retrieved_sources.add(normalize_source_path(str(file_path)))

    # 计算指标
    relevant_retrieved = retrieved_sources & expected_sources
    missed_sources = expected_sources - retrieved_sources
    irrelevant_sources = retrieved_sources - expected_sources

    precision = len(relevant_retrieved) / len(retrieved_sources) if retrieved_sources else 0.0
    recall = len(relevant_retrieved) / len(expected_sources) if expected_sources else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "retrieved_sources": list(retrieved_sources),
        "expected_sources": list(expected_sources),
        "relevant_retrieved": list(relevant_retrieved),
        "missed_sources": list(missed_sources),
        "irrelevant_sources": list(irrelevant_sources)
    }


def match_chunk(
    chunk_spec: Dict[str, Any],
    metadata_record: Dict[str, Any]
) -> Tuple[bool, float]:
    """检查单个 chunk 是否匹配规格。

    Args:
        chunk_spec: chunk 规格，包含:
            - source: 源文件名
            - chunk_index: chunk 索引（可选）
            - heading_path: 标题路径（可选，精确匹配）
            - heading_path_pattern: 标题路径正则（可选）
            - chunk_type: chunk 类型（可选）
            - weight: 权重（默认 1.0）
        metadata_record: 元数据记录

    Returns:
        (is_match: bool, weight: float)
    """
    weight = chunk_spec.get("weight", 1.0)

    # 源文件匹配
    expected_source = chunk_spec.get("source", "")
    if expected_source:
        record_source = normalize_source_path(str(metadata_record.get("source_path", "")))
        if normalize_source_path(expected_source) != record_source:
            return False, 0.0

    # chunk_index 精确匹配
    if "chunk_index" in chunk_spec:
        expected_idx = chunk_spec["chunk_index"]
        record_idx = metadata_record.get("chunk_index")
        if record_idx != expected_idx:
            return False, 0.0

    # heading_path 精确匹配
    if "heading_path" in chunk_spec:
        expected_path = chunk_spec["heading_path"]
        record_path = metadata_record.get("heading_path", "")
        if expected_path != record_path:
            return False, 0.0

    # heading_path_pattern 正则匹配
    if "heading_path_pattern" in chunk_spec:
        pattern = chunk_spec["heading_path_pattern"]
        record_path = metadata_record.get("heading_path", "")
        if not re.search(pattern, record_path, re.IGNORECASE):
            return False, 0.0

    # chunk_type 匹配
    if "chunk_type" in chunk_spec:
        expected_type = chunk_spec["chunk_type"]
        record_type = metadata_record.get("chunk_type", "")
        if expected_type != record_type:
            return False, 0.0

    return True, weight


def compute_chunk_recall_at_k(
    questions: List[Dict[str, Any]],
    k_values: List[int],
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    """计算 chunk 级别的 recall@k。

    支持 expected_chunks 字段进行 chunk 级别匹配，向后兼容文档级匹配。

    Args:
        questions: 题目列表
        k_values: k 值列表

    Returns:
        (summary, results) 元组
    """
    import numpy as np  # type: ignore

    index, metadata = load_retrieval_assets()
    k_max = max(k_values) if k_values else 0
    results: List[Dict[str, Any]] = []
    agg: Dict[str, Dict[str, float]] = {}

    for question in questions:
        expected_chunks = question.get("expected_chunks", [])
        expected_sources = [
            normalize_source_path(s)
            for s in (question.get("expected_sources") or question.get("sources") or [])
        ]
        chunk_match_mode = question.get("chunk_match_mode", "any")  # "any" 或 "all"

        # 如果没有 expected_chunks，回退到文档级匹配
        use_chunk_level = bool(expected_chunks)

        case_type = str(question.get("case_type") or "").strip().lower()
        q_type = str(question.get("type") or "").strip().lower()
        allow_any_source = case_type == "multi_source" or q_type == "multi_doc"

        if (not expected_sources and not expected_chunks) or k_max == 0:
            results.append({"id": question.get("id"), "recall": {}, "chunk_level": use_chunk_level})
            continue

        query = str(question.get("query") or "")
        if not query:
            results.append({"id": question.get("id"), "recall": {}, "chunk_level": use_chunk_level})
            continue

        # 使用全局缓存（已在 precompute_embeddings 中批量计算）
        query_embedding = get_cached_embedding(query)
        embedding = np.array(query_embedding, dtype="float32").reshape(1, -1)
        distances, indices = index.search(embedding, min(k_max, index.ntotal))

        # 收集检索到的 chunks 信息
        retrieved_records: List[Dict[str, Any]] = []
        for idx in indices[0]:
            if idx == -1 or idx >= len(metadata):
                continue
            retrieved_records.append(metadata[idx])

        recall_map: Dict[str, float] = {}
        for k in k_values:
            top_k_records = retrieved_records[:k]

            if use_chunk_level:
                # Chunk 级别匹配
                total_weight = sum(c.get("weight", 1.0) for c in expected_chunks)
                matched_weight = 0.0
                matched_chunks = []

                for chunk_spec in expected_chunks:
                    for record in top_k_records:
                        is_match, weight = match_chunk(chunk_spec, record)
                        if is_match:
                            matched_weight += weight
                            matched_chunks.append(chunk_spec)
                            break  # 每个 chunk_spec 只匹配一次

                if chunk_match_mode == "any":
                    # 只要匹配到任意一个就算成功
                    recall_value = 1.0 if matched_chunks else 0.0
                else:
                    # 按权重计算召回率
                    recall_value = matched_weight / total_weight if total_weight > 0 else 0.0
            else:
                # 文档级匹配（向后兼容）
                top_k_sources = [
                    normalize_source_path(str(r.get("source_path", "")))
                    for r in top_k_records
                ]
                if allow_any_source:
                    recall_value = 1.0 if any(s in top_k_sources for s in expected_sources) else 0.0
                else:
                    hits = sum(1 for s in expected_sources if s in top_k_sources)
                    recall_value = hits / max(len(expected_sources), 1)

            recall_map[str(k)] = recall_value
            agg.setdefault(str(k), {"sum": 0.0, "count": 0.0})
            agg[str(k)]["sum"] += recall_value
            agg[str(k)]["count"] += 1.0

        results.append({
            "id": question.get("id"),
            "recall": recall_map,
            "chunk_level": use_chunk_level
        })

    summary: Dict[str, Dict[str, float]] = {}
    for k, values in agg.items():
        count = values["count"]
        summary[k] = {
            "mean_recall": (values["sum"] / count) if count else 0.0,
            "count": count,
        }
    return summary, results


# ========== 部分得分评估函数 ==========


def evaluate_numeric_validations(
    answer: str,
    validations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """评估数值校验规则。

    Args:
        answer: 答案文本
        validations: 数值校验规则列表，每项包含:
            - pattern: 正则表达式，支持多个捕获组（取第一个非空的组）
            - expected_value: 期望的数值
            - tolerance: 允许的误差范围（默认 0）
            - weight: 权重（默认 0.3）

    Returns:
        {
            "score": float (0~1),
            "matched_values": List[Dict],
            "details": str
        }
    """
    if not validations:
        return {"score": 1.0, "matched_values": [], "details": "无数值校验"}

    total_weight = 0.0
    earned_weight = 0.0
    matched_values = []

    for v in validations:
        pattern = v.get("pattern", "")
        expected = v.get("expected_value")
        tolerance = v.get("tolerance", 0)
        weight = v.get("weight", 0.3)

        total_weight += weight

        # 解析动态值引用
        expected = resolve_dynamic_value(expected)

        # 如果动态值解析失败（仍是字符串引用），跳过此校验
        if not pattern or expected is None:
            continue
        if isinstance(expected, str) and expected.startswith("$stats."):
            # 动态值未解析成功，跳过
            matched_values.append({
                "expected": expected,
                "actual": None,
                "passed": False,
                "pattern": pattern,
                "error": "动态值未解析（预计算数据不存在）"
            })
            continue

        # 确保 expected 是数字
        try:
            expected = float(expected)
        except (TypeError, ValueError):
            matched_values.append({
                "expected": expected,
                "actual": None,
                "passed": False,
                "pattern": pattern,
                "error": f"期望值不是数字: {expected}"
            })
            continue

        match = re.search(pattern, answer, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                # 支持多捕获组：取第一个非空的捕获组
                actual = None
                for group in match.groups():
                    if group is not None and group.strip():
                        actual = int(group.strip())
                        break

                if actual is None:
                    matched_values.append({
                        "expected": expected,
                        "actual": None,
                        "passed": False,
                        "pattern": pattern,
                        "error": "所有捕获组均为空"
                    })
                    continue

                if abs(actual - expected) <= tolerance:
                    earned_weight += weight
                    matched_values.append({
                        "expected": expected,
                        "actual": actual,
                        "passed": True,
                        "pattern": pattern
                    })
                else:
                    matched_values.append({
                        "expected": expected,
                        "actual": actual,
                        "passed": False,
                        "pattern": pattern
                    })
            except (ValueError, IndexError) as e:
                matched_values.append({
                    "expected": expected,
                    "actual": None,
                    "passed": False,
                    "pattern": pattern,
                    "error": f"无法解析数值: {e}"
                })
        else:
            matched_values.append({
                "expected": expected,
                "actual": None,
                "passed": False,
                "pattern": pattern,
                "error": "未匹配到模式"
            })

    passed_count = sum(1 for v in matched_values if v.get("passed"))
    return {
        "score": earned_weight / total_weight if total_weight > 0 else 1.0,
        "matched_values": matched_values,
        "details": f"数值校验: {passed_count}/{len(validations)}"
    }


def evaluate_content_score(
    question: Dict[str, Any],
    answer: str,
    tool_events: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """计算内容得分（部分得分机制）。

    Args:
        question: 题目配置
        answer: 答案文本
        tool_events: 工具调用事件列表（可选，用于 negative 题目评估）

    Returns:
        {
            "score": float (0~1),
            "matched_must_have": List[str],
            "matched_should_have": List[str],
            "matched_evidence": List[str],
            "details": str
        }
    """
    content_rules = question.get("content_rules", {})
    must_have = content_rules.get("must_have", [])
    should_have = content_rules.get("should_have", [])
    evidence = content_rules.get("evidence", [])
    unknown_indicators = content_rules.get("unknown_indicators", [])
    numeric_validations = content_rules.get("numeric_validations", [])

    # 是否启用语义匹配（默认启用）
    use_semantic_match = content_rules.get("use_semantic_match", True)

    # 处理 negative case
    if question.get("allow_unknown") and unknown_indicators:
        # 先检查是否有虚假来源声明
        false_citation_result = verify_negative_answer(question, answer, tool_events)
        if false_citation_result:
            return false_citation_result

        answer_text = answer or ""
        config = get_config()
        unknown_patterns = config.unknown_detection.patterns
        exclusion_patterns = config.unknown_detection.exclusion_patterns

        # hard_exclusion: "虽然...但是..."这类转折结构，表示实际上给出了答案
        hard_exclusion_patterns = exclusion_patterns

        # soft_exclusion: 直接检测回答中是否包含"补充选项"相关表述
        # 这类回答是合理的用户体验：先说笔记没有，再询问是否用通用知识补充
        soft_exclusion_keywords = [
            r"通用知识",
            r"不限于笔记",
            r"允许.*补充",
            r"用.*补充",
            r"可以.*补充",
        ]

        base_unknown = any(re.search(pat, answer_text) for pat in unknown_patterns)
        has_hard_exclusion = any(
            re.search(pat, answer_text) for pat in hard_exclusion_patterns
        )
        has_soft_exclusion = any(
            re.search(pat, answer_text) for pat in soft_exclusion_keywords
        )
        is_unk = base_unknown and not has_hard_exclusion
        confidence = 0.9 if (is_unk and has_soft_exclusion) else (1.0 if is_unk else 0.0)

        # 检查是否尝试了检索
        attempted_retrieval = False
        if tool_events:
            attempted_retrieval = any(
                e.get("tool_name") == "query_my_notes" and e.get("stage") == "start"
                for e in tool_events
            )

        if is_unk:
            if attempted_retrieval:
                # 正确行为：尝试检索后回答不知道
                details = "正确识别为未知问题（已尝试检索）"
                if has_soft_exclusion:
                    details += "，包含补充选项"
                return {
                    "score": confidence,
                    "matched_must_have": [],
                    "matched_should_have": [],
                    "matched_evidence": [],
                    "is_unknown_answer": True,
                    "behavior": "correct_rejection",
                    "attempted_retrieval": True,
                    "details": details
                }
            else:
                # 懒惰拒绝：未尝试检索就回答不知道，扣分
                lazy_penalty = get_config().unknown_detection.lazy_rejection_penalty
                details = "回答不知道但未尝试检索（扣分）"
                if has_soft_exclusion:
                    details += "，包含补充选项"
                return {
                    "score": confidence * lazy_penalty,
                    "matched_must_have": [],
                    "matched_should_have": [],
                    "matched_evidence": [],
                    "is_unknown_answer": True,
                    "behavior": "lazy_rejection",
                    "attempted_retrieval": False,
                    "details": details
                }
        else:
            # 没有匹配到 unknown pattern
            # 但如果包含"补充选项"相关表述，说明模型意图是说"不知道"，只是表述方式不同
            # 这种情况给予部分分数，而非直接判 hallucination
            if has_soft_exclusion:
                soft_score = config.unknown_detection.soft_unknown_score
                return {
                    "score": soft_score,
                    "matched_must_have": [],
                    "matched_should_have": [],
                    "matched_evidence": [],
                    "is_unknown_answer": True,
                    "behavior": "soft_unknown",
                    "attempted_retrieval": attempted_retrieval,
                    "details": "包含补充选项表述，视为合理的未知回答"
                }
            return {
                "score": 0.0,
                "matched_must_have": [],
                "matched_should_have": [],
                "matched_evidence": [],
                "is_unknown_answer": False,
                "behavior": "hallucination",
                "attempted_retrieval": attempted_retrieval,
                "details": "应回答不知道但给出了答案"
            }

    # 计算 must_have 得分
    total_weight = 0.0
    earned_weight = 0.0
    matched_must = []
    semantic_matches = []

    for item in must_have:
        if isinstance(item, dict):
            text = item.get("text", "")
            weight = item.get("weight", 0.2)
            item_synonyms = item.get("synonyms", [])
        else:
            text = str(item)
            weight = 1.0 / max(len(must_have), 1)
            item_synonyms = []

        total_weight += weight
        if not text:
            continue

        # 使用增强的匹配函数（支持同义词、正则）
        is_match, matched_term = contains_normalized(answer, text, item_synonyms)
        if is_match:
            earned_weight += weight
            matched_must.append({"text": text, "match_type": "exact", "matched_term": matched_term})
        elif use_semantic_match and len(text) >= 5:
            # 尝试语义匹配
            is_match, similarity, match_type = semantic_match(answer, text)
            if is_match and match_type == "semantic":
                semantic_match_weight = get_config().thresholds.semantic_match_weight
                earned_weight += weight * semantic_match_weight
                matched_must.append({"text": text, "match_type": "semantic", "similarity": similarity})
                semantic_matches.append(text)

    # 计算 should_have 加分
    matched_should = []
    bonus_weight = 0.0
    for item in should_have:
        if isinstance(item, dict):
            text = item.get("text", "")
            weight = item.get("weight", 0.1)
            item_synonyms = item.get("synonyms", [])
        else:
            text = str(item)
            weight = 0.1
            item_synonyms = []

        if text:
            is_match, matched_term = contains_normalized(answer, text, item_synonyms)
            if is_match:
                bonus_weight += weight
                matched_should.append(text)

    # 计算 evidence 得分
    matched_ev = []
    evidence_weight = 0.0
    evidence_total = 0.0
    for item in evidence:
        if isinstance(item, dict):
            text = item.get("text", "")
            weight = item.get("weight", 0.3)
            item_synonyms = item.get("synonyms", [])
        else:
            text = str(item)
            weight = 0.3
            item_synonyms = []

        evidence_total += weight
        if not text:
            continue

        # 使用增强的匹配函数
        is_match, matched_term = contains_normalized(answer, text, item_synonyms)
        if is_match:
            evidence_weight += weight
            matched_ev.append({"text": text, "match_type": "exact", "matched_term": matched_term})
        elif use_semantic_match and len(text) >= 5:
            # 尝试语义匹配
            is_match, similarity, match_type = semantic_match(answer, text)
            if is_match and match_type == "semantic":
                semantic_match_weight = get_config().thresholds.semantic_match_weight
                evidence_weight += weight * semantic_match_weight
                matched_ev.append({"text": text, "match_type": "semantic", "similarity": similarity})
                semantic_matches.append(text)

    # 计算数值校验得分
    numeric_result = evaluate_numeric_validations(answer, numeric_validations)
    numeric_score = numeric_result["score"]
    numeric_total_weight = sum(v.get("weight", 0.3) for v in numeric_validations)

    # 综合计算得分
    all_weights = total_weight + evidence_total + numeric_total_weight
    if all_weights > 0:
        base_score = (earned_weight + evidence_weight + numeric_score * numeric_total_weight) / all_weights
    else:
        base_score = 1.0 if not must_have and not evidence and not numeric_validations else 0.0

    # 加上 should_have 加分，但不超过 1.0
    final_score = min(1.0, base_score + bonus_weight)

    details_parts = [f"must_have: {len(matched_must)}/{len(must_have)}", f"evidence: {len(matched_ev)}/{len(evidence)}"]
    if numeric_validations:
        details_parts.append(numeric_result["details"])
    if semantic_matches:
        details_parts.append(f"semantic: {len(semantic_matches)}")

    return {
        "score": final_score,
        "matched_must_have": matched_must,
        "matched_should_have": matched_should,
        "matched_evidence": matched_ev,
        "semantic_matches": semantic_matches,
        "numeric_validations": numeric_result.get("matched_values", []),
        "is_unknown_answer": False,
        "details": ", ".join(details_parts)
    }


def _legacy_evaluate_citation_score(question: Dict[str, Any], answer: str) -> Dict[str, Any]:
    """计算引用得分（旧版本，保留作为降级备份）。

    增强版：
    1. 验证引用内容是否来自正确的源文档
    2. 检查引用上下文是否合理
    3. 评估引用与问题的相关性

    Returns:
        {
            "score": float (0~1),
            "has_quote": bool,
            "has_source": bool,
            "quote_validity_score": float (0~1),
            "context_match_score": float (0~1),
            "relevance_score": float (0~1),
            "valid_quotes": List[Dict],  # 包含上下文信息
            "invalid_quotes": List[str],
            "details": str
        }
    """
    citation_rules = question.get("citation_rules", {})
    require_quote = citation_rules.get("require_quote", False)
    require_source = citation_rules.get("require_source", False)

    # 如果不要求引用，直接满分
    if not require_quote and not require_source:
        return {
            "score": 1.0,
            "has_quote": False,
            "has_source": False,
            "quote_validity_score": 1.0,
            "context_match_score": 1.0,
            "relevance_score": 1.0,
            "valid_quotes": [],
            "invalid_quotes": [],
            "details": "不要求引用"
        }

    quotes = extract_quotes(answer)
    has_quote = len(quotes) > 0

    # 检查是否包含来源文件名
    sources = question.get("expected_sources", [])
    has_source = False
    for source in sources:
        basename = Path(source).stem.lower()
        if basename in answer.lower():
            has_source = True
            break

    # 增强版：验证引用内容、上下文和相关性
    valid_quotes = []
    invalid_quotes = []
    quote_validity_score = 1.0
    context_match_score = 1.0
    relevance_score = 1.0

    if quotes and sources:
        total_context_score = 0.0
        total_relevance_score = 0.0
        valid_quote_count = 0

        question_text = question.get("query", "")

        for quote in quotes:
            is_valid = False
            best_context = None

            # 检查引用是否在任何源文档中
            # 遍历所有来源，优先选择具有上下文的精确匹配结果
            for source in sources:
                matches, context = quote_matches_source(quote, source)
                if matches:
                    is_valid = True
                    if context:
                        # 找到精确匹配（有上下文），立即使用
                        best_context = context
                        break
                    # 关键词匹配，继续查找是否有更好的精确匹配

            if is_valid:
                # 计算引用与问题的相关性
                relevance = evaluate_quote_relevance(quote, question_text)
                total_relevance_score += relevance

                # 如果有上下文信息，说明是精确匹配
                if best_context:
                    total_context_score += 1.0
                    valid_quotes.append({
                        "quote": quote,
                        "context": best_context,
                        "relevance": relevance
                    })
                else:
                    # 关键词匹配，没有精确位置
                    total_context_score += 0.7  # 部分分
                    valid_quotes.append({
                        "quote": quote,
                        "context": None,
                        "relevance": relevance
                    })

                valid_quote_count += 1
            else:
                invalid_quotes.append(quote)

        # 计算各项得分
        total_quotes = len(quotes)
        quote_validity_score = valid_quote_count / total_quotes if total_quotes > 0 else 1.0

        if valid_quote_count > 0:
            context_match_score = total_context_score / valid_quote_count
            relevance_score = total_relevance_score / valid_quote_count
        else:
            context_match_score = 0.0
            relevance_score = 0.0

    # 计算综合得分
    score = 0.0
    if require_quote and require_source:
        base_score = 0.0
        if has_quote:
            base_score += 0.3  # 降低存在性权重
        if has_source:
            base_score += 0.2  # 降低来源标注权重
        # 引用有效性、上下文匹配、相关性各占一部分
        base_score += quote_validity_score * 0.25
        base_score += context_match_score * 0.15
        base_score += relevance_score * 0.10
        score = base_score
    elif require_quote:
        if has_quote:
            # 引用存在性 40%，有效性 30%，上下文 20%，相关性 10%
            score = 0.4 + quote_validity_score * 0.3 + context_match_score * 0.2 + relevance_score * 0.1
        else:
            score = 0.0
    elif require_source:
        score = 1.0 if has_source else 0.0

    return {
        "score": score,
        "has_quote": has_quote,
        "has_source": has_source,
        "quote_validity_score": quote_validity_score,
        "context_match_score": context_match_score,
        "relevance_score": relevance_score,
        "valid_quotes": valid_quotes,
        "invalid_quotes": invalid_quotes,
        "details": f"quote: {has_quote}, source: {has_source}, validity: {quote_validity_score:.2f}, context: {context_match_score:.2f}, relevance: {relevance_score:.2f}"
    }


def get_attribution_mode(question: Dict[str, Any]) -> str:
    """推断归因评估模式。

    Args:
        question: 题目配置

    Returns:
        "disabled" | "standard" | "strict"
    """
    citation_rules = question.get("citation_rules", {})

    # 优先使用新字段
    if "attribution_mode" in citation_rules:
        return citation_rules["attribution_mode"]

    # 向后兼容：从 require_quote/require_source 推断
    require_quote = citation_rules.get("require_quote", False)
    require_source = citation_rules.get("require_source", False)

    if not require_quote and not require_source:
        return "disabled"
    elif require_quote and require_source:
        return "strict"
    else:
        return "standard"


def extract_read_files_from_events(tool_events: Optional[List[Dict[str, Any]]]) -> List[str]:
    """从 tool_events 中提取已读取的文件路径。

    Args:
        tool_events: 工具调用事件列表

    Returns:
        已读取的文件路径列表（已归一化）
    """
    if not tool_events:
        return []

    read_files = []
    for event in tool_events:
        if event.get("tool_name") != "read_note_file":
            continue
        if event.get("stage") != "end":
            continue
        args = event.get("arguments") or {}
        if not isinstance(args, dict):
            continue
        file_path = args.get("file_path") or args.get("path") or ""
        if file_path:
            read_files.append(normalize_source_path(str(file_path)))

    return read_files


def compute_workflow_compliance(
    question: Dict[str, Any],
    tool_events: Optional[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """计算工作流合规得分。

    检查是否遵循 检索→阅读→回答 的标准流程。

    权重分配：
    - 是否调用 query_my_notes: 0.25
    - 是否调用 read_note_file: 0.50
    - 是否读取 expected_sources: 0.25

    Args:
        question: 题目配置
        tool_events: 工具调用事件列表

    Returns:
        {
            "score": float (0~1),
            "has_query": bool,
            "has_read": bool,
            "expected_sources_coverage": float (0~1),
            "details": str
        }
    """
    if not tool_events:
        return {
            "score": 0.0,
            "has_query": False,
            "has_read": False,
            "expected_sources_coverage": 0.0,
            "details": "无工具调用事件"
        }

    # 检查是否调用了 query_my_notes
    has_query = any(
        event.get("tool_name") == "query_my_notes" and event.get("stage") == "end"
        for event in tool_events
    )

    # 检查是否调用了 read_note_file
    read_files = extract_read_files_from_events(tool_events)
    has_read = len(read_files) > 0

    # 计算 expected_sources 覆盖率
    expected_sources = [
        normalize_source_path(s)
        for s in (question.get("expected_sources") or question.get("sources") or [])
    ]

    expected_sources_coverage = 0.0
    if expected_sources:
        read_files_set = set(read_files)
        hits = sum(1 for s in expected_sources if s in read_files_set)
        expected_sources_coverage = hits / len(expected_sources)
    else:
        # 没有 expected_sources，只要读了文件就给满分
        expected_sources_coverage = 1.0 if has_read else 0.0

    # 计算得分
    score = 0.0
    if has_query:
        score += 0.25
    if has_read:
        score += 0.50
    score += expected_sources_coverage * 0.25

    return {
        "score": score,
        "has_query": has_query,
        "has_read": has_read,
        "expected_sources_coverage": expected_sources_coverage,
        "details": f"query={has_query}, read={has_read}, coverage={expected_sources_coverage:.2f}"
    }


def extract_key_claims(answer: str, min_length: int = 10) -> List[str]:
    """从回答中提取关键陈述。

    过滤掉：
    - 元信息句（"笔记中提到..."）
    - 问句
    - 过短的句子
    - 代码块

    Args:
        answer: 回答文本
        min_length: 最小陈述长度

    Returns:
        关键陈述列表
    """
    config = get_config()
    meta_patterns = config.attribution_evaluation.meta_statement_patterns

    # 移除代码块
    answer_no_code = re.sub(r'```[\s\S]*?```', '', answer)
    answer_no_code = re.sub(r'`[^`]+`', '', answer_no_code)

    # 分句
    sentences = re.split(r'[。.!！\n]', answer_no_code)

    claims = []
    for sent in sentences:
        sent = sent.strip()

        # 过滤过短
        if len(sent) < min_length:
            continue

        # 过滤问句
        if sent.endswith('?') or sent.endswith('？'):
            continue

        # 过滤元信息句
        is_meta = False
        for pattern in meta_patterns:
            if re.match(pattern, sent):
                is_meta = True
                break
        if is_meta:
            continue

        claims.append(sent)

    return claims


def check_claim_attribution_with_llm(claims: List[str], document_content: str) -> List[bool]:
    """使用 LLM 判断陈述是否有文档支撑。

    批量处理多个陈述。

    Args:
        claims: 陈述列表
        document_content: 文档内容

    Returns:
        布尔列表，表示每个陈述是否有支撑
    """
    if not claims:
        return []

    try:
        client = _get_openai_client()
        model = os.getenv("SUPER_MIND_CHAT_MODEL", DEFAULT_CHAT_MODEL)

        # 构建 prompt
        claims_text = "\n".join(f"{i+1}. {claim}" for i, claim in enumerate(claims))
        prompt = f"""以下是一些陈述和一段文档内容。请判断每个陈述是否有文档内容支撑。

陈述列表：
{claims_text}

文档内容：
{document_content[:2000]}

请对每个陈述回答 yes 或 no，用逗号分隔。例如：yes,no,yes,no

只回答 yes/no 序列，不要其他内容。"""

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_completion_tokens=100,
        )

        content = (response.choices[0].message.content or "").strip().lower()

        # 解析回答
        results = []
        parts = content.split(',')
        for i, claim in enumerate(claims):
            if i < len(parts):
                answer = parts[i].strip()
                results.append(answer == "yes" or answer == "true")
            else:
                # 如果 LLM 没有返回足够的答案，默认为 False
                results.append(False)

        return results

    except Exception as e:
        # LLM 调用失败，默认全部为 False
        print(f"⚠ LLM 归因判断失败: {e}", file=sys.stderr)
        return [False] * len(claims)


def compute_content_attribution(
    answer: str,
    tool_events: Optional[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """计算内容归因得分。

    使用两阶段匹配（Embedding 粗筛 + LLM 精判）验证回答中的关键陈述
    是否有已读文档支撑。

    Args:
        answer: 回答文本
        tool_events: 工具调用事件列表

    Returns:
        {
            "score": float (0~1),
            "total_claims": int,
            "attributed_claims": int,
            "unattributed_claims": List[str],
            "details": str
        }
    """
    config = get_config()
    embedding_threshold = config.attribution_evaluation.embedding_threshold
    llm_uncertain_range = config.attribution_evaluation.llm_uncertain_range
    min_claim_length = config.attribution_evaluation.min_claim_length

    # 提取关键陈述
    claims = extract_key_claims(answer, min_claim_length)

    if not claims:
        # 没有提取到关键陈述，可能是回答太短或全是代码
        return {
            "score": 1.0,
            "total_claims": 0,
            "attributed_claims": 0,
            "unattributed_claims": [],
            "details": "无关键陈述"
        }

    # 加载已读文档内容
    read_files = extract_read_files_from_events(tool_events)
    if not read_files:
        # 没有读取任何文档，无法归因
        return {
            "score": 0.0,
            "total_claims": len(claims),
            "attributed_claims": 0,
            "unattributed_claims": claims,
            "details": "未读取任何文档"
        }

    # 合并所有已读文档内容
    all_docs_content = []
    for file_path in read_files:
        content = load_source_content(file_path)
        if content:
            all_docs_content.append(content)

    combined_docs = "\n\n".join(all_docs_content)

    if not combined_docs:
        return {
            "score": 0.0,
            "total_claims": len(claims),
            "attributed_claims": 0,
            "unattributed_claims": claims,
            "details": "文档内容为空"
        }

    # 将文档按标题/段落分块，避免整篇文档 embedding 被稀释
    doc_chunks = re.split(r'\n#{1,3} ', combined_docs)
    doc_chunks = [ch.strip() for ch in doc_chunks if len(ch.strip()) > 30]
    if not doc_chunks:
        doc_chunks = [combined_docs]

    # 批量预计算所有 claims 和 chunks 的 embedding，避免逐对 API 调用
    import numpy as np  # type: ignore
    all_texts = claims + doc_chunks
    embed_batch(all_texts)  # 一次性批量计算并缓存

    # 构建 embedding 矩阵，用矩阵运算计算相似度
    claim_embs = np.array([get_cached_embedding(c) for c in claims], dtype="float32")
    chunk_embs = np.array([get_cached_embedding(ch) for ch in doc_chunks], dtype="float32")

    # 归一化
    claim_norms = np.linalg.norm(claim_embs, axis=1, keepdims=True)
    chunk_norms = np.linalg.norm(chunk_embs, axis=1, keepdims=True)
    claim_norms = np.where(claim_norms == 0, 1, claim_norms)
    chunk_norms = np.where(chunk_norms == 0, 1, chunk_norms)
    claim_embs_normed = claim_embs / claim_norms
    chunk_embs_normed = chunk_embs / chunk_norms

    # 余弦相似度矩阵: [n_claims, n_chunks]
    sim_matrix = claim_embs_normed @ chunk_embs_normed.T
    # 每个 claim 取最高相似度
    max_similarities = sim_matrix.max(axis=1)

    # 两阶段匹配
    attributed_count = 0
    uncertain_claims = []
    unattributed_claims = []

    for i, claim in enumerate(claims):
        max_sim = float(max_similarities[i])

        if max_sim >= embedding_threshold:
            # 高相似度，直接判定为有支撑
            attributed_count += 1
        elif max_sim >= llm_uncertain_range[0] and max_sim <= llm_uncertain_range[1]:
            # 不确定区间，留给 LLM 精判
            uncertain_claims.append(claim)
        else:
            # 低相似度，判定为无支撑
            unattributed_claims.append(claim)

    # 阶段 2：LLM 精判不确定的陈述
    if uncertain_claims:
        llm_results = check_claim_attribution_with_llm(uncertain_claims, combined_docs)
        for claim, has_support in zip(uncertain_claims, llm_results):
            if has_support:
                attributed_count += 1
            else:
                unattributed_claims.append(claim)

    # 计算得分
    score = attributed_count / len(claims) if claims else 1.0

    return {
        "score": score,
        "total_claims": len(claims),
        "attributed_claims": attributed_count,
        "unattributed_claims": unattributed_claims,
        "details": f"attributed={attributed_count}/{len(claims)}"
    }


def compute_fidelity_score(
    answer: str,
    tool_events: Optional[List[Dict[str, Any]]],
    expected_sources: Optional[List[str]] = None
) -> Dict[str, Any]:
    """计算忠实度得分。

    检查三个层面：
    1. 伪造来源检测：声称引用了某文件，但未实际读取
    2. 知识边界标注：是否明确区分笔记内容和模型知识
    3. 未读文件直接回答：没有读文件但给出详细回答

    Args:
        answer: 回答文本
        tool_events: 工具调用事件列表
        expected_sources: 期望的源文档列表

    Returns:
        {
            "score": float (0~1),
            "has_fake_source": bool,
            "has_knowledge_disclaimer": bool,
            "answered_without_reading": bool,
            "details": str
        }
    """
    config = get_config()
    disclaimer_keywords = config.attribution_evaluation.knowledge_disclaimer_keywords

    # 检查是否读取了文件
    read_files = extract_read_files_from_events(tool_events)
    has_read = len(read_files) > 0

    # 1. 伪造来源检测
    has_fake_source = False
    if expected_sources:
        read_files_set = set(read_files)
        for source in expected_sources:
            normalized_source = normalize_source_path(source)
            basename = Path(source).stem.lower()

            # 如果答案中提到了这个文件名，但没有实际读取
            if basename in answer.lower() and normalized_source not in read_files_set:
                has_fake_source = True
                break

    if has_fake_source:
        # 伪造来源，直接 0 分
        return {
            "score": 0.0,
            "has_fake_source": True,
            "has_knowledge_disclaimer": False,
            "answered_without_reading": False,
            "details": "检测到伪造来源"
        }

    # 2. 知识边界标注
    has_knowledge_disclaimer = any(
        keyword in answer
        for keyword in disclaimer_keywords
    )

    # 3. 未读文件直接回答
    answered_without_reading = False
    if not has_read and len(answer.strip()) > 50:
        # 没有读文件但给出了详细回答（超过 50 字符）
        answered_without_reading = True

    # 计算得分
    if has_knowledge_disclaimer:
        score = 1.0
    elif answered_without_reading:
        score = 0.3
    else:
        score = 1.0

    return {
        "score": score,
        "has_fake_source": has_fake_source,
        "has_knowledge_disclaimer": has_knowledge_disclaimer,
        "answered_without_reading": answered_without_reading,
        "details": f"fake={has_fake_source}, disclaimer={has_knowledge_disclaimer}, no_read={answered_without_reading}"
    }


def evaluate_attribution_score(
    question: Dict[str, Any],
    answer: str,
    tool_events: Optional[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """评估归因可靠性得分（新版引用评估）。

    根据 attribution_mode 分发评估逻辑：
    - disabled: 不要求归因，直接满分
    - standard: 标准模式，评估三个维度
    - strict: 严格模式，更高要求

    三个维度：
    - workflow (0.35): 工作流合规性
    - attribution (0.50): 内容归因
    - fidelity (0.15): 忠实度

    Args:
        question: 题目配置
        answer: 回答文本
        tool_events: 工具调用事件列表

    Returns:
        {
            "score": float (0~1),
            "workflow": Dict,
            "attribution": Dict,
            "fidelity": Dict,
            "details": str
        }
    """
    attribution_mode = get_attribution_mode(question)

    # disabled 模式：直接满分
    if attribution_mode == "disabled":
        return {
            "score": 1.0,
            "workflow": {"score": 1.0},
            "attribution": {"score": 1.0},
            "fidelity": {"score": 1.0},
            "details": "归因评估已禁用"
        }

    # 计算三个维度
    workflow_result = compute_workflow_compliance(question, tool_events)
    attribution_result = compute_content_attribution(answer, tool_events)
    fidelity_result = compute_fidelity_score(
        answer,
        tool_events,
        question.get("expected_sources") or question.get("sources")
    )

    # 加权计算总分
    workflow_weight = 0.35
    attribution_weight = 0.50
    fidelity_weight = 0.15

    score = (
        workflow_result["score"] * workflow_weight +
        attribution_result["score"] * attribution_weight +
        fidelity_result["score"] * fidelity_weight
    )

    return {
        "score": score,
        "workflow": workflow_result,
        "attribution": attribution_result,
        "fidelity": fidelity_result,
        "details": f"workflow={workflow_result['score']:.2f}, attribution={attribution_result['score']:.2f}, fidelity={fidelity_result['score']:.2f}"
    }


def compute_arguments_hash(arguments: Any) -> str:
    """计算工具调用参数的哈希值，用于检测重复调用。"""
    import hashlib
    if arguments is None:
        return ""
    try:
        arg_str = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(arg_str.encode()).hexdigest()[:8]
    except (TypeError, ValueError):
        return str(arguments)[:50]


def check_call_order_violations(call_order: List[str]) -> float:
    """检查工具调用顺序的合理性。

    规则：
    - 检索工具（query_my_notes, read_note_file）应在代码执行（run_code_interpreter）之前
    - 返回违规惩罚分数 (0~1)，0 表示无违规
    """
    config = get_config()
    max_penalty = config.tool_evaluation.order_violation_max_penalty

    retrieval_tools = {"query_my_notes", "read_note_file"}
    execution_tools = {"run_code_interpreter"}

    penalty = 0.0
    last_execution_idx = -1

    for idx, tool in enumerate(call_order):
        if tool in execution_tools:
            last_execution_idx = idx
        elif tool in retrieval_tools and last_execution_idx >= 0:
            # 在代码执行之后又调用检索工具，可能是顺序不当
            penalty += 0.1

    return min(penalty, max_penalty)


def evaluate_tool_behavior(
    question: Dict[str, Any],
    tool_events: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """评估工具调用行为。

    Args:
        question: 题目配置
        tool_events: 工具调用事件列表

    Returns:
        {
            "score": float (0~1),
            "tools_used": List[str],
            "tool_count": int,
            "errors": List[str],
            "redundancy_ratio": float,
            "order_violations": float,
            "details": str
        }
    """
    tools_used: List[str] = []
    errors: List[str] = []
    tool_count = 0
    call_signatures: List[Tuple[str, str]] = []  # (tool_name, args_hash)
    call_order: List[str] = []

    for event in tool_events:
        stage = event.get("stage")
        tool_name = event.get("tool_name")

        if stage == "start":
            tool_count += 1
            if tool_name:
                call_order.append(tool_name)
                # 计算调用签名用于冗余检测
                args = event.get("arguments")
                args_hash = compute_arguments_hash(args)
                call_signatures.append((tool_name, args_hash))
        elif stage == "end":
            if tool_name and tool_name not in tools_used:
                tools_used.append(tool_name)
        elif stage == "error":
            error_msg = event.get("error")
            if error_msg:
                errors.append(error_msg)

    # 计算基础得分
    base_score = 1.0

    # 检查期望的工具是否被调用
    expected_tools = question.get("expected_tools", [])
    if expected_tools:
        matched = [t for t in expected_tools if t in tools_used]
        tool_match_score = len(matched) / len(expected_tools)
        base_score *= tool_match_score

    # 如果有错误，扣分
    if errors:
        base_score *= 0.8

    # 新增：冗余调用检测
    unique_signatures = set(call_signatures)
    redundancy_ratio = 0.0
    if call_signatures:
        redundancy_ratio = 1 - len(unique_signatures) / len(call_signatures)

    # 新增：调用顺序合理性检查
    order_violations = check_call_order_violations(call_order)

    # 应用惩罚
    config = get_config()
    redundancy_penalty_rate = config.tool_evaluation.redundancy_penalty_rate
    final_score = base_score * (1 - redundancy_ratio * redundancy_penalty_rate) * (1 - order_violations)

    return {
        "score": final_score,
        "tools_used": tools_used,
        "tool_count": tool_count,
        "errors": errors,
        "expected_tools": expected_tools,
        "redundancy_ratio": redundancy_ratio,
        "order_violations": order_violations,
        "details": f"tools: {len(tools_used)}, calls: {tool_count}, errors: {len(errors)}, redundancy: {redundancy_ratio:.2f}, order_penalty: {order_violations:.2f}"
    }


def evaluate_question(
    question: Dict[str, Any],
    answer: str,
    recall_score: float = 1.0,
    tool_events: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """完整评估，返回部分得分。

    Args:
        question: 题目配置
        answer: 答案文本
        recall_score: 检索召回得分 (0~1)，由 recall@k 计算得出
        tool_events: 工具调用事件列表（可选）

    Returns:
        {
            "retrieval_score": float,
            "actual_retrieval_score": float | None,
            "content_score": float,
            "citation_score": float,
            "tool_score": float,
            "total_score": float,
            "passed": bool,
            "details": {...}
        }
    """
    # 获取权重配置（优先级：题目配置 > 类别配置 > 默认配置）
    category = question.get("category", "default")
    weights = get_scoring_weights(question, category)
    retrieval_weight = weights.retrieval_weight
    content_weight = weights.content_weight
    citation_weight = weights.citation_weight
    tool_weight = weights.tool_weight

    # 计算各维度得分
    content_result = evaluate_content_score(question, answer, tool_events)
    citation_result = evaluate_attribution_score(question, answer, tool_events)
    actual_retrieval_score = compute_actual_retrieval_score(question, tool_events)

    # 计算检索指标（precision/recall/f1）
    retrieval_metrics = compute_retrieval_metrics(question, tool_events)

    # 使用传入的 recall_score 作为检索得分
    retrieval_score = recall_score

    # 计算工具行为得分（如果有工具事件）
    tool_result = None
    tool_score = 1.0
    if tool_events is not None and (tool_weight > 0 or question.get("expected_tools")):
        tool_result = evaluate_tool_behavior(question, tool_events)
        tool_score = tool_result["score"]

    # 如果有工具权重，重新分配其他权重
    if tool_weight > 0:
        # 按比例缩减其他权重
        scale = 1.0 - tool_weight
        retrieval_weight *= scale
        content_weight *= scale
        citation_weight *= scale

    # 计算加权总分
    total_score = (
        retrieval_score * retrieval_weight +
        content_result["score"] * content_weight +
        citation_result["score"] * citation_weight +
        tool_score * tool_weight
    )

    # 按题型获取通过阈值
    pass_threshold = get_pass_threshold(category)
    passed = total_score >= pass_threshold

    result = {
        "retrieval_score": retrieval_score,
        "actual_retrieval_score": actual_retrieval_score,
        "content_score": content_result["score"],
        "citation_score": citation_result["score"],
        "tool_score": tool_score,
        "total_score": total_score,
        "passed": passed,
        "pass_threshold": pass_threshold,
        "details": {
            "content": content_result,
            "citation": citation_result,
            "retrieval": {
                "recall_score": retrieval_score,
                "actual_retrieval_score": actual_retrieval_score,
                "metrics": retrieval_metrics,
            },
        }
    }

    if tool_result:
        result["details"]["tool"] = tool_result

    return result


def compute_retrieval_analysis(
    results: List[Dict[str, Any]],
    questions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """汇总检索指标分析。

    Args:
        results: 评估结果列表
        questions: 题目列表

    Returns:
        {
            "avg_precision": float,
            "avg_recall": float,
            "avg_f1": float,
            "questions_with_missed_sources": List[str],
            "common_missed_sources": Dict[str, int],
            "common_irrelevant_sources": Dict[str, int]
        }
    """
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    count = 0

    questions_with_missed = []
    missed_sources_counter: Dict[str, int] = {}
    irrelevant_sources_counter: Dict[str, int] = {}

    for result, question in zip(results, questions):
        metrics = result.get("details", {}).get("retrieval", {}).get("metrics", {})
        if not metrics:
            continue

        precision_sum += metrics.get("precision", 0)
        recall_sum += metrics.get("recall", 0)
        f1_sum += metrics.get("f1", 0)
        count += 1

        # 记录有遗漏源的问题
        missed = metrics.get("missed_sources", [])
        if missed:
            questions_with_missed.append(question.get("id", ""))
            for src in missed:
                missed_sources_counter[src] = missed_sources_counter.get(src, 0) + 1

        # 记录常见的不相关源
        irrelevant = metrics.get("irrelevant_sources", [])
        for src in irrelevant:
            irrelevant_sources_counter[src] = irrelevant_sources_counter.get(src, 0) + 1

    return {
        "avg_precision": precision_sum / count if count > 0 else 0.0,
        "avg_recall": recall_sum / count if count > 0 else 0.0,
        "avg_f1": f1_sum / count if count > 0 else 0.0,
        "questions_with_missed_sources": questions_with_missed,
        "common_missed_sources": dict(sorted(
            missed_sources_counter.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]),
        "common_irrelevant_sources": dict(sorted(
            irrelevant_sources_counter.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10])
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade answers against testsets")
    parser.add_argument("--testset", default="eval/testsets/testset.json", help="Path to testset JSON")
    parser.add_argument("--answers", required=True, help="Path to answers JSON")
    parser.add_argument("--output", help="Write report JSON to this path")
    parser.add_argument("--require-sources", action="store_true", help="Require answers to include source paths")
    parser.add_argument("--recall-k", default="1,3,5,10", help="Compute recall@k (comma-separated integers, default: 1,3,5,10)")
    parser.add_argument("--tool-traces", help="Path to tool traces JSON for tool behavior evaluation")
    parser.add_argument("--config", help="Path to evaluation config JSON (default: eval/config/eval_config.json)")

    args = parser.parse_args()

    # 加载评估配置
    if args.config:
        load_eval_config(Path(args.config))
    else:
        load_eval_config()

    config = get_config()

    testset = load_json(Path(args.testset))
    answers = load_json(Path(args.answers))

    # 加载工具追踪（可选）
    tool_traces: Dict[str, List[Dict[str, Any]]] = {}
    if args.tool_traces:
        try:
            tool_traces = load_json(Path(args.tool_traces))
            if not isinstance(tool_traces, dict):
                print(
                    f"⚠ 工具追踪格式异常: {args.tool_traces}",
                    file=sys.stderr,
                )
                tool_traces = {}
            else:
                print(
                    f"✓ 已加载工具追踪: {len(tool_traces)} 条",
                    file=sys.stderr,
                )
        except Exception as exc:
            print(
                f"⚠ 无法加载工具追踪 {args.tool_traces}: {exc}",
                file=sys.stderr,
            )
            tool_traces = {}

    meta = testset.get("meta", {})
    questions = testset.get("questions", [])
    if args.tool_traces:
        if not tool_traces:
            print("⚠ 工具追踪为空，actual_retrieval_score 将显示 N/A。", file=sys.stderr)
        elif len(tool_traces) < len(questions):
            print(
                "⚠ 工具追踪数量少于题目数，部分题目的 actual_retrieval_score 可能缺失。",
                file=sys.stderr,
            )

    # 只对有答案的题目预计算 embedding（避免对无答案题目做无用的 rewrite_query LLM 调用）
    answer_ids = set(answers.keys()) if isinstance(answers, dict) else set()
    questions_with_answers = [
        q for q in questions if q.get("id") in answer_ids
    ] if answer_ids else questions
    print(f"预计算 embedding 缓存（{len(questions_with_answers)}/{len(questions)} 题有答案）...")
    try:
        precompute_embeddings(questions_with_answers)
        print(f"✓ 已缓存 {len(_embedding_cache)} 个 embedding")
    except Exception as exc:
        print(f"⚠ embedding 预计算失败: {exc}，将在评分时按需计算")

    # 先计算 recall@k，构建 recall_map
    recall_summary = {}
    recall_results = []
    baseline_recall_summary = {}
    baseline_recall_results = []
    recall_map: Dict[str, float] = {}
    recall_k_for_scoring = config.recall.k_for_scoring  # 从配置获取用于评分的 k 值
    use_chunk_level = config.recall.use_chunk_level  # 是否使用 chunk 级别评估

    if args.recall_k:
        raw_values = [item.strip() for item in args.recall_k.split(",") if item.strip()]
        k_values = []
        for item in raw_values:
            try:
                value = int(item)
                if value > 0:
                    k_values.append(value)
            except ValueError:
                continue
        if k_values:
            try:
                # 只对有答案的题目计算 recall@k（避免无用的 rewrite_query LLM 调用）
                recall_questions = questions_with_answers if questions_with_answers else questions
                # 单路召回（baseline）
                if use_chunk_level:
                    baseline_recall_summary, baseline_recall_results = compute_chunk_recall_at_k(recall_questions, k_values)
                else:
                    baseline_recall_summary, baseline_recall_results = compute_recall_at_k(recall_questions, k_values)
                # 多路召回（主指标）
                recall_summary, recall_results = compute_multipath_recall_at_k(recall_questions, k_values)
                # 用多路召回的结果构建评分 recall_map
                for r in recall_results:
                    qid = r.get("id")
                    recall_value = r.get("recall", {}).get(str(recall_k_for_scoring), 1.0)
                    if qid:
                        recall_map[qid] = recall_value
            except Exception as exc:
                recall_summary = {"error": str(exc)}

    results = []
    passed = 0
    total_score_sum = 0.0

    # 只评估有答案的题目，跳过无答案题目避免无用计算
    questions_to_eval = [q for q in questions if q["id"] in answer_ids] if answer_ids else questions
    print(f"\n评估 {len(questions_to_eval)} 道题目（共 {len(questions)} 题）...")
    print("-" * 80)

    for i, q in enumerate(questions_to_eval, 1):
        qid = q["id"]
        category = q.get("category", "unknown")
        answer = answers.get(qid, "")

        # 获取该题目的 recall score，默认为 1.0
        recall_score = recall_map.get(qid, 1.0)

        # 获取该题目的工具事件（如果有）
        tool_events = tool_traces.get(qid)

        # 部分得分模式
        eval_result = evaluate_question(q, answer, recall_score=recall_score, tool_events=tool_events)
        total_score_sum += eval_result["total_score"]
        is_passed = eval_result["passed"]
        if is_passed:
            passed += 1

        # 打印每道题的评估结果
        status = "✓" if is_passed else "✗"
        score = eval_result["total_score"]
        content_score = eval_result["content_score"]
        retrieval_score = eval_result["retrieval_score"]
        actual_retrieval_score = eval_result.get("actual_retrieval_score")
        citation_score = eval_result["citation_score"]

        # 获取内容评估详情
        content_details = eval_result["details"].get("content", {})
        details_str = content_details.get("details", "")

        print(f"[{i:02d}/{len(questions_to_eval)}] {status} {qid}")
        actual_display = (
            f"{actual_retrieval_score:.2f}"
            if isinstance(actual_retrieval_score, (int, float))
            else "N/A"
        )
        print(
            "       分数: "
            f"{score:.2f} (内容:{content_score:.2f} 检索:{retrieval_score:.2f} "
            f"实际检索:{actual_display} 引用:{citation_score:.2f})"
        )
        print(f"       类别: {category} | {details_str}")

        results.append({
            "id": qid,
            "passed": is_passed,
            "total_score": eval_result["total_score"],
            "retrieval_score": eval_result["retrieval_score"],
            "actual_retrieval_score": eval_result.get("actual_retrieval_score"),
            "content_score": eval_result["content_score"],
            "citation_score": eval_result["citation_score"],
            "tool_score": eval_result["tool_score"],
            "details": eval_result["details"]
        })

    print("-" * 80)

    # 统计各类别
    category_stats = {}
    for q, result in zip(questions_to_eval, results):
        cat = q.get("category", "unknown")
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "passed": 0, "score_sum": 0.0}
        category_stats[cat]["total"] += 1
        if result["passed"]:
            category_stats[cat]["passed"] += 1
        category_stats[cat]["score_sum"] += result.get("total_score", 0.0)

    for cat, stats in category_stats.items():
        stats["pass_rate"] = stats["passed"] / stats["total"] if stats["total"] > 0 else 0.0
        stats["avg_score"] = stats["score_sum"] / stats["total"] if stats["total"] > 0 else 0.0

    # 计算检索指标汇总
    retrieval_analysis = compute_retrieval_analysis(results, questions_to_eval)

    # 构建报告
    avg_score = total_score_sum / len(results) if results else 0.0
    perfect_threshold = config.thresholds.perfect_score_threshold
    perfect_count = sum(1 for r in results if r.get("total_score", 0) >= perfect_threshold)

    report = {
        "meta": {
            "testset_name": meta.get("name", "unknown"),
            "scoring_mode": "partial",
            "config_version": config.version,
            "perfect_threshold": perfect_threshold,
            "use_chunk_level": use_chunk_level
        },
        "summary": {
            "total": len(results),
            "passed": passed,
            "pass_rate": (passed / len(results)) if results else 0.0,
            "avg_score": avg_score,
            "perfect_count": perfect_count,
            "perfect_rate": perfect_count / len(results) if results else 0.0
        },
        "category_stats": category_stats,
        "retrieval_analysis": retrieval_analysis,
        "results": results,
        "recall_summary": recall_summary,
        "recall_results": recall_results,
        "baseline_recall_summary": baseline_recall_summary,
        "baseline_recall_results": baseline_recall_results,
    }

    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✓ 评估完成: {passed}/{len(results)} 通过 ({report['summary']['pass_rate']:.1%})")
        print(f"  平均得分: {avg_score:.2f}")
        print(f"  满分率: {report['summary']['perfect_rate']:.1%} (阈值: {perfect_threshold})")
        print(f"  报告已保存至: {args.output}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

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


def normalize_text(text: str) -> str:
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


def quote_matches_source(quote: str, source_path: str) -> bool:
    """检查引用内容是否在源文档中存在。

    使用模糊匹配，允许空白字符差异。
    """
    source_content = load_source_content(source_path)
    if not source_content:
        return False

    # 规范化后比较
    normalized_quote = normalize_text(quote)
    normalized_source = normalize_text(source_content)

    # 直接包含检查
    if normalized_quote in normalized_source:
        return True

    # 对于较短的引用，尝试更宽松的匹配
    if len(normalized_quote) < 50:
        # 分词后检查关键词
        quote_words = set(normalized_quote.split())
        if len(quote_words) >= 3:
            # 至少 80% 的词在源文档中
            matches = sum(1 for w in quote_words if w in normalized_source)
            if matches / len(quote_words) >= 0.8:
                return True

    return False


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
    texts = set()

    for q in questions:
        # 收集 query（用于 recall@k 计算）
        query = q.get("query", "")
        if query and len(query) >= 5:
            texts.add(query)

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


def evaluate_citation_score(question: Dict[str, Any], answer: str) -> Dict[str, Any]:
    """计算引用得分。

    新增：验证引用内容是否来自正确的源文档。

    Returns:
        {
            "score": float (0~1),
            "has_quote": bool,
            "has_source": bool,
            "quote_validity_score": float (0~1),
            "valid_quotes": List[str],
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

    # 新增：验证引用内容是否来自源文档
    valid_quotes = []
    invalid_quotes = []
    quote_validity_score = 1.0

    if quotes and sources:
        for quote in quotes:
            is_valid = False
            for source in sources:
                if quote_matches_source(quote, source):
                    is_valid = True
                    break
            if is_valid:
                valid_quotes.append(quote)
            else:
                invalid_quotes.append(quote)

        # 有效引用率
        quote_validity_score = len(valid_quotes) / len(quotes) if quotes else 1.0

    # 计算得分
    score = 0.0
    if require_quote and require_source:
        base_score = 0.0
        if has_quote:
            base_score += 0.4
        if has_source:
            base_score += 0.3
        # 引用有效性占 0.3
        base_score += quote_validity_score * 0.3
        score = base_score
    elif require_quote:
        if has_quote:
            # 引用存在性 70%，有效性 30%
            score = 0.7 + quote_validity_score * 0.3
        else:
            score = 0.0
    elif require_source:
        score = 1.0 if has_source else 0.0

    return {
        "score": score,
        "has_quote": has_quote,
        "has_source": has_source,
        "quote_validity_score": quote_validity_score,
        "valid_quotes": valid_quotes,
        "invalid_quotes": invalid_quotes,
        "details": f"quote: {has_quote}, source: {has_source}, validity: {quote_validity_score:.2f}"
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
    citation_result = evaluate_citation_score(question, answer)
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

    # 预计算所有评估文本的 embedding（用于语义匹配）
    print("预计算 embedding 缓存...")
    try:
        precompute_embeddings(questions)
        print(f"✓ 已缓存 {len(_embedding_cache)} 个 embedding")
    except Exception as exc:
        print(f"⚠ embedding 预计算失败: {exc}，将在评分时按需计算")

    # 先计算 recall@k，构建 recall_map
    recall_summary = {}
    recall_results = []
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
                # 根据配置选择使用 chunk 级别还是文档级别评估
                if use_chunk_level:
                    recall_summary, recall_results = compute_chunk_recall_at_k(questions, k_values)
                else:
                    recall_summary, recall_results = compute_recall_at_k(questions, k_values)
                # 构建 {question_id: recall_score} 映射
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

    print(f"\n评估 {len(questions)} 道题目...")
    print("-" * 80)

    for i, q in enumerate(questions, 1):
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

        print(f"[{i:02d}/{len(questions)}] {status} {qid}")
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
    for q, result in zip(questions, results):
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
    retrieval_analysis = compute_retrieval_analysis(results, questions)

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

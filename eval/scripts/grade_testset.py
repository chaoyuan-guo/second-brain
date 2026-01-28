#!/usr/bin/env python3
"""Grader for evaluation testsets with partial scoring support.

部分得分机制：
- retrieval_score: 检索得分 (0~1)
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_API_BASE_URL = "https://space.ai-builders.com/backend/v1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

UNKNOWN_PATTERNS = [
    r"未知",
    r"不知道",
    r"无法确定",
    r"无相关信息",
    r"文档未覆盖",
    r"没有.*信息",
    r"未提及",
    r"没有.*相关",
    r"笔记.*没有",
    r"未找到",
    r"无法回答",
]

# 排除模式：包含这些模式时不应被判定为 unknown
UNKNOWN_EXCLUSION_PATTERNS = [
    r"虽然.*但",
    r"虽然.*不过",
    r"虽然没有.*可以",
    r"虽然未.*但",
]

# 语义匹配配置
SEMANTIC_SIMILARITY_THRESHOLD = 0.75  # 语义相似度阈值
SEMANTIC_MATCH_WEIGHT = 0.7  # 语义匹配给予 70% 权重（精确匹配为 100%）

# 全局 embedding 缓存
_embedding_cache: Dict[str, List[float]] = {}


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_unknown(answer: str) -> bool:
    """检测答案是否表示"不知道"。

    改进：排除"虽然没有...但是..."这类结构，避免误判。
    """
    # 先检查排除模式
    for pat in UNKNOWN_EXCLUSION_PATTERNS:
        if re.search(pat, answer):
            return False
    # 再检查 unknown 模式
    for pat in UNKNOWN_PATTERNS:
        if re.search(pat, answer):
            return True
    return False


def is_unknown_with_confidence(answer: str) -> tuple:
    """检测答案是否表示"不知道"，并返回置信度。

    Returns:
        (is_unknown: bool, confidence: float, matched_pattern: str)
    """
    # 先检查排除模式
    for pat in UNKNOWN_EXCLUSION_PATTERNS:
        if re.search(pat, answer):
            return False, 0.0, ""

    # 检查 unknown 模式
    for pat in UNKNOWN_PATTERNS:
        match = re.search(pat, answer)
        if match:
            # 如果匹配在答案开头附近（前100字符），置信度更高
            confidence = 0.9 if match.start() < 100 else 0.7
            return True, confidence, pat
    return False, 0.0, ""


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def contains_normalized(haystack: str, needle: str) -> bool:
    if not haystack or not needle:
        return False
    return normalize_text(needle) in normalize_text(haystack)


def extract_quotes(answer: str) -> List[str]:
    patterns = [
        r"“([^”]{4,})”",
        r"\"([^\"]{4,})\"",
        r"‘([^’]{4,})’",
        r"'([^']{4,})'",
        r"「([^」]{4,})」",
        r"『([^』]{4,})』",
    ]
    quotes: List[str] = []
    for pat in patterns:
        for match in re.findall(pat, answer):
            cleaned = match.strip()
            if cleaned:
                quotes.append(cleaned)
    return quotes


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


def embed_query(query: str) -> List[float]:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        load_dotenv = None

    if load_dotenv:
        load_dotenv()
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("openai package is required for recall@k") from exc

    api_key = os.getenv("SUPER_MIND_API_KEY") or os.getenv("AI_BUILDER_TOKEN")
    if not api_key:
        raise RuntimeError("Missing API key for embeddings")
    base_url = os.getenv("SUPER_MIND_API_BASE_URL", DEFAULT_API_BASE_URL)
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.embeddings.create(model=DEFAULT_EMBEDDING_MODEL, input=query)
    if not response.data:
        raise RuntimeError("Empty embedding response")
    return response.data[0].embedding


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
    threshold: float = SEMANTIC_SIMILARITY_THRESHOLD
) -> Tuple[bool, float, str]:
    """检查答案是否语义匹配目标文本。

    Args:
        answer: 答案文本
        target: 目标文本
        threshold: 相似度阈值

    Returns:
        (is_match: bool, similarity: float, match_type: str)
        match_type: "exact" | "semantic" | "none"
    """
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
    """预计算所有评估文本的 embedding 并缓存。"""
    texts = set()
    for q in questions:
        rules = q.get("content_rules", {})
        for item in rules.get("must_have", []) + rules.get("evidence", []):
            text = item.get("text") if isinstance(item, dict) else str(item)
            if text and len(text) >= 5:
                texts.add(text)

    # 批量预计算
    for text in texts:
        try:
            get_cached_embedding(text)
        except Exception:
            pass  # 忽略单个失败，继续处理其他


def compute_recall_at_k(
    questions: List[Dict[str, Any]],
    k_values: List[int],
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    import numpy as np  # type: ignore

    index, metadata = load_retrieval_assets()
    k_max = max(k_values) if k_values else 0
    results: List[Dict[str, Any]] = []
    agg: Dict[str, Dict[str, float]] = {}
    query_cache: Dict[str, List[float]] = {}

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
        if query not in query_cache:
            query_cache[query] = embed_query(query)
        embedding = np.array(query_cache[query], dtype="float32").reshape(1, -1)
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


# ========== 部分得分评估函数 ==========


def evaluate_numeric_validations(
    answer: str,
    validations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """评估数值校验规则。

    Args:
        answer: 答案文本
        validations: 数值校验规则列表，每项包含:
            - pattern: 正则表达式，第一个捕获组为数值
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

        if not pattern or expected is None:
            continue

        match = re.search(pattern, answer, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                actual = int(match.group(1))
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
            except (ValueError, IndexError):
                matched_values.append({
                    "expected": expected,
                    "actual": None,
                    "passed": False,
                    "pattern": pattern,
                    "error": "无法解析数值"
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


def evaluate_content_score(question: Dict[str, Any], answer: str) -> Dict[str, Any]:
    """计算内容得分（部分得分机制）。

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
        is_unk, confidence, _ = is_unknown_with_confidence(answer)
        if is_unk:
            return {
                "score": confidence,
                "matched_must_have": [],
                "matched_should_have": [],
                "matched_evidence": [],
                "is_unknown_answer": True,
                "details": "正确识别为未知问题"
            }
        else:
            return {
                "score": 0.0,
                "matched_must_have": [],
                "matched_should_have": [],
                "matched_evidence": [],
                "is_unknown_answer": False,
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
        else:
            text = str(item)
            weight = 1.0 / max(len(must_have), 1)

        total_weight += weight
        if not text:
            continue

        # 先尝试精确匹配
        if contains_normalized(answer, text):
            earned_weight += weight
            matched_must.append({"text": text, "match_type": "exact"})
        elif use_semantic_match and len(text) >= 5:
            # 尝试语义匹配
            is_match, similarity, match_type = semantic_match(answer, text)
            if is_match and match_type == "semantic":
                earned_weight += weight * SEMANTIC_MATCH_WEIGHT
                matched_must.append({"text": text, "match_type": "semantic", "similarity": similarity})
                semantic_matches.append(text)

    # 计算 should_have 加分
    matched_should = []
    bonus_weight = 0.0
    for item in should_have:
        if isinstance(item, dict):
            text = item.get("text", "")
            weight = item.get("weight", 0.1)
        else:
            text = str(item)
            weight = 0.1

        if text and contains_normalized(answer, text):
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
        else:
            text = str(item)
            weight = 0.3

        evidence_total += weight
        if not text:
            continue

        # 先尝试精确匹配
        if contains_normalized(answer, text):
            evidence_weight += weight
            matched_ev.append({"text": text, "match_type": "exact"})
        elif use_semantic_match and len(text) >= 5:
            # 尝试语义匹配
            is_match, similarity, match_type = semantic_match(answer, text)
            if is_match and match_type == "semantic":
                evidence_weight += weight * SEMANTIC_MATCH_WEIGHT
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

    Returns:
        {
            "score": float (0~1),
            "has_quote": bool,
            "has_source": bool,
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

    # 计算得分
    score = 0.0
    if require_quote and require_source:
        if has_quote:
            score += 0.5
        if has_source:
            score += 0.5
    elif require_quote:
        score = 1.0 if has_quote else 0.0
    elif require_source:
        score = 1.0 if has_source else 0.0

    return {
        "score": score,
        "has_quote": has_quote,
        "has_source": has_source,
        "details": f"quote: {has_quote}, source: {has_source}"
    }


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
            "details": str
        }
    """
    tools_used: List[str] = []
    errors: List[str] = []
    tool_count = 0

    for event in tool_events:
        stage = event.get("stage")
        if stage == "start":
            tool_count += 1
        elif stage == "end":
            tool_name = event.get("tool_name")
            if tool_name and tool_name not in tools_used:
                tools_used.append(tool_name)
        elif stage == "error":
            error_msg = event.get("error")
            if error_msg:
                errors.append(error_msg)

    # 计算得分
    score = 1.0

    # 检查期望的工具是否被调用
    expected_tools = question.get("expected_tools", [])
    if expected_tools:
        matched = [t for t in expected_tools if t in tools_used]
        tool_match_score = len(matched) / len(expected_tools)
        score *= tool_match_score

    # 如果有错误，扣分
    if errors:
        score *= 0.8

    return {
        "score": score,
        "tools_used": tools_used,
        "tool_count": tool_count,
        "errors": errors,
        "expected_tools": expected_tools,
        "details": f"tools: {len(tools_used)}, calls: {tool_count}, errors: {len(errors)}"
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
            "content_score": float,
            "citation_score": float,
            "tool_score": float,
            "total_score": float,
            "passed": bool,
            "details": {...}
        }
    """
    # 获取权重配置
    scoring = question.get("scoring", {})
    retrieval_weight = scoring.get("retrieval_weight", 0.3)
    content_weight = scoring.get("content_weight", 0.5)
    citation_weight = scoring.get("citation_weight", 0.2)
    tool_weight = scoring.get("tool_weight", 0.0)  # 默认不计入工具评分

    # 计算各维度得分
    content_result = evaluate_content_score(question, answer)
    citation_result = evaluate_citation_score(question, answer)

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

    # 通过阈值：总分 >= 0.6
    passed = total_score >= 0.6

    result = {
        "retrieval_score": retrieval_score,
        "content_score": content_result["score"],
        "citation_score": citation_result["score"],
        "tool_score": tool_score,
        "total_score": total_score,
        "passed": passed,
        "details": {
            "content": content_result,
            "citation": citation_result
        }
    }

    if tool_result:
        result["details"]["tool"] = tool_result

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade answers against testsets")
    parser.add_argument("--testset", default="eval/testsets/testset.json", help="Path to testset JSON")
    parser.add_argument("--answers", required=True, help="Path to answers JSON")
    parser.add_argument("--output", help="Write report JSON to this path")
    parser.add_argument("--require-sources", action="store_true", help="Require answers to include source paths")
    parser.add_argument("--recall-k", default="1,3,5,10", help="Compute recall@k (comma-separated integers, default: 1,3,5,10)")
    parser.add_argument("--tool-traces", help="Path to tool traces JSON for tool behavior evaluation")

    args = parser.parse_args()

    testset = load_json(Path(args.testset))
    answers = load_json(Path(args.answers))

    # 加载工具追踪（可选）
    tool_traces: Dict[str, List[Dict[str, Any]]] = {}
    if args.tool_traces:
        try:
            tool_traces = load_json(Path(args.tool_traces))
        except Exception:
            pass

    meta = testset.get("meta", {})
    questions = testset.get("questions", [])

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
    recall_k_for_scoring = 5  # 用于评分的 k 值

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

    for q in questions:
        qid = q["id"]
        answer = answers.get(qid, "")

        # 获取该题目的 recall score，默认为 1.0
        recall_score = recall_map.get(qid, 1.0)

        # 获取该题目的工具事件（如果有）
        tool_events = tool_traces.get(qid)

        # 部分得分模式
        eval_result = evaluate_question(q, answer, recall_score=recall_score, tool_events=tool_events)
        total_score_sum += eval_result["total_score"]
        if eval_result["passed"]:
            passed += 1
        results.append({
            "id": qid,
            "passed": eval_result["passed"],
            "total_score": eval_result["total_score"],
            "retrieval_score": eval_result["retrieval_score"],
            "content_score": eval_result["content_score"],
            "citation_score": eval_result["citation_score"],
            "tool_score": eval_result["tool_score"],
            "details": eval_result["details"]
        })

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

    # 构建报告
    avg_score = total_score_sum / len(results) if results else 0.0
    perfect_count = sum(1 for r in results if r.get("total_score", 0) >= 0.95)
    
    report = {
        "meta": {
            "testset_name": meta.get("name", "unknown"),
            "scoring_mode": "partial"
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
        "results": results,
        "recall_summary": recall_summary,
        "recall_results": recall_results,
    }

    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✓ 评估完成: {passed}/{len(results)} 通过 ({report['summary']['pass_rate']:.1%})")
        print(f"  平均得分: {avg_score:.2f}")
        print(f"  满分率: {report['summary']['perfect_rate']:.1%}")
        print(f"  报告已保存至: {args.output}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

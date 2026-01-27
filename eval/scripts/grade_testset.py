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
from typing import Any, Dict, List, Tuple

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
        expected_sources = [normalize_source_path(s) for s in (question.get("sources") or [])]
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

    for item in must_have:
        if isinstance(item, dict):
            text = item.get("text", "")
            weight = item.get("weight", 0.2)
        else:
            text = str(item)
            weight = 1.0 / max(len(must_have), 1)

        total_weight += weight
        if text and contains_normalized(answer, text):
            earned_weight += weight
            matched_must.append(text)

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
        if text and contains_normalized(answer, text):
            evidence_weight += weight
            matched_ev.append(text)

    # 综合计算得分
    if total_weight + evidence_total > 0:
        base_score = (earned_weight + evidence_weight) / (total_weight + evidence_total)
    else:
        base_score = 1.0 if not must_have and not evidence else 0.0

    # 加上 should_have 加分，但不超过 1.0
    final_score = min(1.0, base_score + bonus_weight)

    return {
        "score": final_score,
        "matched_must_have": matched_must,
        "matched_should_have": matched_should,
        "matched_evidence": matched_ev,
        "is_unknown_answer": False,
        "details": f"must_have: {len(matched_must)}/{len(must_have)}, evidence: {len(matched_ev)}/{len(evidence)}"
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


def evaluate_question(question: Dict[str, Any], answer: str) -> Dict[str, Any]:
    """完整评估，返回部分得分。

    Returns:
        {
            "retrieval_score": float,
            "content_score": float,
            "citation_score": float,
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

    # 计算各维度得分
    content_result = evaluate_content_score(question, answer)
    citation_result = evaluate_citation_score(question, answer)

    # retrieval_score 在这里设为 1.0，实际由 recall@k 单独计算
    retrieval_score = 1.0

    # 计算加权总分
    total_score = (
        retrieval_score * retrieval_weight +
        content_result["score"] * content_weight +
        citation_result["score"] * citation_weight
    )

    # 通过阈值：总分 >= 0.6
    passed = total_score >= 0.6

    return {
        "retrieval_score": retrieval_score,
        "content_score": content_result["score"],
        "citation_score": citation_result["score"],
        "total_score": total_score,
        "passed": passed,
        "details": {
            "content": content_result,
            "citation": citation_result
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade answers against testsets")
    parser.add_argument("--testset", default="eval/testsets/testset.json", help="Path to testset JSON")
    parser.add_argument("--answers", required=True, help="Path to answers JSON")
    parser.add_argument("--output", help="Write report JSON to this path")
    parser.add_argument("--require-sources", action="store_true", help="Require answers to include source paths")
    parser.add_argument("--recall-k", default="1,3,5,10", help="Compute recall@k (comma-separated integers, default: 1,3,5,10)")

    args = parser.parse_args()

    testset = load_json(Path(args.testset))
    answers = load_json(Path(args.answers))

    meta = testset.get("meta", {})

    results = []
    passed = 0
    total_score_sum = 0.0

    questions = testset.get("questions", [])
    for q in questions:
        qid = q["id"]
        answer = answers.get(qid, "")

        # 部分得分模式
        eval_result = evaluate_question(q, answer)
        total_score_sum += eval_result["total_score"]
        if eval_result["passed"]:
            passed += 1
        results.append({
            "id": qid,
            "passed": eval_result["passed"],
            "total_score": eval_result["total_score"],
            "content_score": eval_result["content_score"],
            "citation_score": eval_result["citation_score"],
            "details": eval_result["details"]
        })

    # 计算 recall@k
    recall_summary = {}
    recall_results = []
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
            except Exception as exc:
                recall_summary = {"error": str(exc)}

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
            "testset_version": version,
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

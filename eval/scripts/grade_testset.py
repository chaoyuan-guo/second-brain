#!/usr/bin/env python3
"""Baseline grader for evaluation testsets.

Usage:
  python eval/scripts/grade_testset.py --answers path/to/answers.json
  python eval/scripts/grade_testset.py --testset eval/testsets/testset_v4.json --answers answers.json --output report.json
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
]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_unknown(answer: str) -> bool:
    for pat in UNKNOWN_PATTERNS:
        if re.search(pat, answer):
            return True
    return False


def match_any_substring(patterns: List[str], text: str) -> bool:
    lowered = text.lower()
    for p in patterns:
        if p.lower() in lowered:
            return True
    return False


def match_any_regex(patterns: List[str], text: str) -> bool:
    for p in patterns:
        if re.search(p, text, flags=re.IGNORECASE):
            return True
    return False


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def contains_normalized(haystack: str, needle: str) -> bool:
    if not haystack or not needle:
        return False
    return normalize_text(needle) in normalize_text(haystack)


def load_sources(question: Dict[str, Any]) -> Dict[str, str]:
    sources = question.get("sources") or []
    loaded: Dict[str, str] = {}
    for source in sources:
        try:
            loaded[str(source)] = Path(source).read_text(encoding="utf-8")
        except FileNotFoundError:
            loaded[str(source)] = ""
    return loaded


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


def check_quote_consistency(
    question: Dict[str, Any],
    answer: str,
) -> List[int]:
    require_quotes = bool(question.get("quote_required"))
    if not require_quotes:
        return []
    quotes = extract_quotes(answer)
    if not quotes:
        return [0]

    sources_map = load_sources(question)
    missing: List[int] = []
    for idx, quote in enumerate(quotes, start=1):
        found = False
        for source_text in sources_map.values():
            if contains_normalized(source_text, quote):
                found = True
                break
        if not found:
            missing.append(idx)
    return missing


def check_evidence(
    question: Dict[str, Any],
    answer: str,
) -> List[int]:
    evidence = question.get("evidence") or []
    mode = str(question.get("evidence_mode") or "strict").strip().lower()
    if not evidence:
        return []
    if mode == "any":
        return []

    sources_map = load_sources(question)
    missing: List[int] = []
    if mode == "any_of":
        for item in evidence:
            if isinstance(item, str):
                text = item
                source = None
            else:
                text = str(item.get("text") or "")
                source = item.get("source")
            if not text or not contains_normalized(answer, text):
                continue
            if source:
                source_text = sources_map.get(str(source), "")
                if not contains_normalized(source_text, text):
                    continue
            return []
        return [0]

    for idx, item in enumerate(evidence, start=1):
        if isinstance(item, str):
            text = item
            source = None
        else:
            text = str(item.get("text") or "")
            source = item.get("source")
        if not text or not contains_normalized(answer, text):
            missing.append(idx)
            continue
        if source:
            source_text = sources_map.get(str(source), "")
            if not contains_normalized(source_text, text):
                missing.append(idx)
    return missing


def check_evidence_in_quotes(
    question: Dict[str, Any],
    answer: str,
) -> List[int]:
    if not question.get("evidence_quote_required"):
        return []
    evidence = question.get("evidence") or []
    if not evidence:
        return []
    mode = str(question.get("evidence_mode") or "strict").strip().lower()
    if mode == "any":
        return []

    quotes = extract_quotes(answer)
    if not quotes:
        return [0]

    def quote_contains(text: str) -> bool:
        return any(contains_normalized(quote, text) for quote in quotes)

    if mode == "any_of":
        for item in evidence:
            if isinstance(item, str):
                text = item
            else:
                text = str(item.get("text") or "")
            if text and quote_contains(text):
                return []
        return [0]

    missing: List[int] = []
    for idx, item in enumerate(evidence, start=1):
        if isinstance(item, str):
            text = item
        else:
            text = str(item.get("text") or "")
        if not text or not quote_contains(text):
            missing.append(idx)
    return missing


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


def check_requirement(req: Dict[str, Any], answer: str) -> bool:
    if "any_of" in req and req["any_of"]:
        if match_any_substring(req["any_of"], answer):
            return True
    if "any_of_regex" in req and req["any_of_regex"]:
        if match_any_regex(req["any_of_regex"], answer):
            return True
    return False


def evaluate_question(
    question: Dict[str, Any],
    answer: str,
) -> Tuple[bool, List[int], List[int], List[int]]:
    missing: List[int] = []
    allow_unknown = bool(question.get("allow_unknown"))
    must_have = question.get("must_have", [])
    missing_evidence: List[int] = []
    missing_quotes: List[int] = []

    if allow_unknown and is_unknown(answer):
        return True, missing, missing_evidence, missing_quotes
    if allow_unknown and not must_have:
        return False, [0], [0], [0]
    if (not allow_unknown) and is_unknown(answer):
        return False, [0], [0], [0]

    for idx, req in enumerate(must_have, start=1):
        if not check_requirement(req, answer):
            missing.append(idx)

    if not missing:
        missing_evidence = check_evidence(question, answer)
    if not missing and not missing_evidence:
        missing_evidence = missing_evidence + check_evidence_in_quotes(question, answer)
    if not missing and not missing_evidence:
        missing_quotes = check_quote_consistency(question, answer)
    return (
        len(missing) == 0
        and len(missing_evidence) == 0
        and len(missing_quotes) == 0,
        missing,
        missing_evidence,
        missing_quotes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade answers against testsets")
    parser.add_argument("--testset", default="eval/testsets/testset_v4.json", help="Path to testset JSON")
    parser.add_argument("--answers", required=True, help="Path to answers JSON")
    parser.add_argument("--output", help="Write report JSON to this path")
    parser.add_argument("--require-sources", action="store_true", help="Require answers to include source paths")
    parser.add_argument("--recall-k", default="", help="Compute recall@k (comma-separated integers)")
    args = parser.parse_args()

    testset = load_json(Path(args.testset))
    answers = load_json(Path(args.answers))

    results = []
    passed = 0

    questions = testset.get("questions", [])
    for q in questions:
        qid = q["id"]
        answer = answers.get(qid, "")
        ok, missing, missing_evidence, missing_quotes = evaluate_question(q, answer)
        if ok and args.require_sources:
            sources = q.get("sources") or []
            if sources:
                answer_lower = answer.lower()
                matched = False
                for source in sources:
                    normalized = str(source)
                    basename = Path(normalized).name.lower()
                    if normalized.lower() in answer_lower or basename in answer_lower:
                        matched = True
                        break
                if not matched:
                    ok = False
                    missing = missing + [0]
        if ok:
            passed += 1
        results.append(
            {
                "id": qid,
                "passed": ok,
                "missing_requirements": missing,
                "missing_evidence": missing_evidence,
                "missing_quotes": missing_quotes,
            }
        )

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

    report = {
        "total": len(results),
        "passed": passed,
        "pass_rate": (passed / len(results)) if results else 0.0,
        "results": results,
        "recall_summary": recall_summary,
        "recall_results": recall_results,
    }

    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

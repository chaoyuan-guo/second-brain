#!/usr/bin/env python3
"""Baseline grader for testset_v2.json.

Usage:
  python eval/scripts/grade_testset_v2.py --answers path/to/answers.json
  python eval/scripts/grade_testset_v2.py --testset eval/testsets/testset_v2.json --answers answers.json --output report.json

answers.json format:
{
  "S01_BFS_max_level_sum": "your answer...",
  "S02_rotting_oranges_multi_source": "..."
}
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def check_requirement(req: Dict[str, Any], answer: str) -> bool:
    if "any_of" in req and req["any_of"]:
        if match_any_substring(req["any_of"], answer):
            return True
    if "any_of_regex" in req and req["any_of_regex"]:
        if match_any_regex(req["any_of_regex"], answer):
            return True
    return False


def evaluate_question(question: Dict[str, Any], answer: str) -> Tuple[bool, List[int]]:
    missing: List[int] = []
    allow_unknown = bool(question.get("allow_unknown"))
    must_have = question.get("must_have", [])

    if allow_unknown and is_unknown(answer):
        return True, missing
    if allow_unknown and not must_have:
        return False, [0]
    if (not allow_unknown) and is_unknown(answer):
        return False, [0]

    for idx, req in enumerate(must_have, start=1):
        if not check_requirement(req, answer):
            missing.append(idx)

    return len(missing) == 0, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade answers against testset_v2.json")
    parser.add_argument("--testset", default="eval/testsets/testset_v2.json", help="Path to testset JSON")
    parser.add_argument("--answers", required=True, help="Path to answers JSON")
    parser.add_argument("--output", help="Write report JSON to this path")
    parser.add_argument("--require-sources", action="store_true", help="Require answers to include source paths")
    args = parser.parse_args()

    testset = load_json(Path(args.testset))
    answers = load_json(Path(args.answers))

    results = []
    passed = 0

    for q in testset.get("questions", []):
        qid = q["id"]
        answer = answers.get(qid, "")
        ok, missing = evaluate_question(q, answer)
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
            }
        )

    report = {
        "total": len(results),
        "passed": passed,
        "pass_rate": (passed / len(results)) if results else 0.0,
        "results": results,
    }

    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""验证 testset 配置与实际数据的一致性。

Usage:
  python eval/scripts/validate_testset.py --testset eval/testsets/testset.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# 相对路径导入
from eval.scripts.grade_testset import resolve_dynamic_value, normalize_source_path

# 笔记目录路径
NOTES_DIR = Path(__file__).resolve().parents[2] / "data" / "notes" / "my_markdowns"


def validate_testset(testset_path: Path, notes_dir: Path) -> List[Dict[str, Any]]:
    """验证 testset 配置与实际数据的一致性。

    Args:
        testset_path: testset JSON 文件路径
        notes_dir: 笔记目录路径

    Returns:
        问题列表，每项包含:
        - id: 题目 ID
        - issue: 问题类型
        - details: 问题详情
    """
    issues: List[Dict[str, Any]] = []

    if not testset_path.exists():
        issues.append({
            "id": "",
            "issue": "testset_not_found",
            "details": f"Testset 文件不存在: {testset_path}"
        })
        return issues

    try:
        testset = json.loads(testset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        issues.append({
            "id": "",
            "issue": "invalid_json",
            "details": f"Testset JSON 格式错误: {e}"
        })
        return issues

    questions = testset.get("questions", [])
    if not questions:
        issues.append({
            "id": "",
            "issue": "empty_testset",
            "details": "Testset 中没有题目"
        })
        return issues

    for q in questions:
        qid = q.get("id", "unknown")

        # 1. 检查期望来源是否存在
        expected_sources = q.get("expected_sources", [])
        for source in expected_sources:
            source_path = notes_dir / source
            if not source_path.exists():
                # 尝试模糊匹配
                found = False
                for file in notes_dir.glob("*.md"):
                    if normalize_source_path(file.name) == normalize_source_path(source):
                        found = True
                        break
                if not found:
                    issues.append({
                        "id": qid,
                        "issue": "missing_source",
                        "details": f"期望来源不存在: {source}"
                    })

        # 2. 检查动态值引用是否可解析
        content_rules = q.get("content_rules", {})
        for nv in content_rules.get("numeric_validations", []):
            expected = nv.get("expected_value")
            if isinstance(expected, str) and expected.startswith("$stats."):
                resolved = resolve_dynamic_value(expected)
                if resolved == expected:
                    # 动态值未解析成功
                    issues.append({
                        "id": qid,
                        "issue": "unresolved_dynamic_value",
                        "details": f"动态值未解析: {expected}"
                    })

        # 3. 检查 must_have 配置是否合理
        must_have = content_rules.get("must_have", [])
        for item in must_have:
            if isinstance(item, dict):
                text = item.get("text", "")
                if not text:
                    issues.append({
                        "id": qid,
                        "issue": "empty_must_have",
                        "details": "must_have 条目缺少 text 字段"
                    })

        # 4. 检查 negative 题目配置
        if q.get("allow_unknown"):
            unknown_indicators = content_rules.get("unknown_indicators", [])
            if not unknown_indicators:
                issues.append({
                    "id": qid,
                    "issue": "missing_unknown_indicators",
                    "details": "allow_unknown=true 但缺少 unknown_indicators"
                })

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="验证 testset 配置")
    parser.add_argument(
        "--testset",
        type=Path,
        default=Path("eval/testsets/testset.json"),
        help="Testset JSON 文件路径"
    )
    parser.add_argument(
        "--notes-dir",
        type=Path,
        default=NOTES_DIR,
        help="笔记目录路径"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="输出问题列表到 JSON 文件"
    )
    args = parser.parse_args()

    print(f"验证 testset: {args.testset}")
    issues = validate_testset(args.testset, args.notes_dir)

    if issues:
        print(f"\n发现 {len(issues)} 个问题:")
        for issue in issues:
            print(f"  [{issue['id']}] {issue['issue']}: {issue['details']}")

        if args.output:
            args.output.write_text(
                json.dumps(issues, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            print(f"\n问题列表已保存至: {args.output}")

        sys.exit(1)
    else:
        print("✓ 验证通过，未发现问题")
        sys.exit(0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""预计算统计数据，用于动态验证 statistics 类型题目。

Usage:
    python eval/scripts/precompute_stats.py --output eval/config/precomputed_stats.json
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def compute_leetcode_stats(file_path: Path) -> Dict[str, Any]:
    """计算 leetcode_submissions.md 的统计数据。"""
    if not file_path.exists():
        return {}

    content = file_path.read_text(encoding="utf-8")

    # 统计提交结果
    submission_results = {
        "Accepted": len(re.findall(r'\| Accepted \|', content)),
        "Wrong Answer": len(re.findall(r'\| Wrong Answer \|', content)),
        "Runtime Error": len(re.findall(r'\| Runtime Error \|', content)),
        "Time Limit Exceeded": len(re.findall(r'\| Time Limit Exceeded \|', content)),
    }

    # 统计唯一题目数（匹配 "## 题目名 (`题号`)" 格式）
    unique_problems = len(re.findall(r'^## (.+?) \(`(\d+)`\)', content, re.MULTILINE))

    # 统计总提交数
    total_submissions = sum(submission_results.values())

    # 计算通过率
    pass_rate = (submission_results["Accepted"] / total_submissions * 100) if total_submissions > 0 else 0.0

    # 提取日期范围（匹配 "| 2024-xx-xx |" 格式）
    dates = re.findall(r'\| (\d{4}-\d{2}-\d{2}) \|', content)
    date_range = {}
    if dates:
        parsed_dates = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
        date_range = {
            "start": min(parsed_dates).strftime("%Y-%m-%d"),
            "end": max(parsed_dates).strftime("%Y-%m-%d"),
            "total_days": (max(parsed_dates) - min(parsed_dates)).days + 1
        }

    # 统计标签（假设有标签行，格式：| 标签 |）
    # 这里简化处理，实际需要根据具体格式调整
    tag_counts = {}
    tag_pattern = r'标签[：:]\s*([^\n]+)'
    tag_matches = re.findall(tag_pattern, content)
    for tags_str in tag_matches:
        tags = [t.strip() for t in re.split(r'[,，、]', tags_str) if t.strip()]
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    # 找到最多的标签
    most_common_tag = max(tag_counts.items(), key=lambda x: x[1])[0] if tag_counts else None

    return {
        "submission_results": submission_results,
        "unique_problems": unique_problems,
        "total_submissions": total_submissions,
        "pass_rate": round(pass_rate, 2),
        "date_range": date_range,
        "tag_counts": tag_counts,
        "most_common_tag": most_common_tag
    }


def compute_stats(notes_dir: Path) -> Dict[str, Any]:
    """计算所有笔记的统计数据。"""
    stats = {}

    # 计算 LeetCode 提交记录统计
    leetcode_file = notes_dir / "leetcode_submissions.md"
    if leetcode_file.exists():
        stats["leetcode_submissions"] = compute_leetcode_stats(leetcode_file)

    # 可以在这里添加更多笔记文件的统计计算
    # 例如：
    # - BFS练习题的题目数量
    # - 动态规划笔记的章节数
    # - 等等

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="预计算笔记统计数据")
    parser.add_argument(
        "--notes-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "notes" / "my_markdowns",
        help="笔记目录路径"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="输出 JSON 文件路径"
    )

    args = parser.parse_args()

    if not args.notes_dir.exists():
        print(f"错误: 笔记目录不存在: {args.notes_dir}")
        return

    print(f"计算笔记统计数据: {args.notes_dir}")
    stats = compute_stats(args.notes_dir)

    # 保存到 JSON 文件
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"✓ 统计数据已保存至: {args.output}")

    # 打印摘要
    if "leetcode_submissions" in stats:
        lc_stats = stats["leetcode_submissions"]
        print("\nLeetCode 提交统计:")
        print(f"  总提交数: {lc_stats.get('total_submissions', 0)}")
        print(f"  Accepted: {lc_stats.get('submission_results', {}).get('Accepted', 0)}")
        print(f"  通过率: {lc_stats.get('pass_rate', 0):.2f}%")
        print(f"  唯一题目数: {lc_stats.get('unique_problems', 0)}")
        if lc_stats.get("date_range"):
            dr = lc_stats["date_range"]
            print(f"  日期范围: {dr['start']} 至 {dr['end']} ({dr['total_days']} 天)")
        if lc_stats.get("most_common_tag"):
            print(f"  最常见标签: {lc_stats['most_common_tag']}")


if __name__ == "__main__":
    main()

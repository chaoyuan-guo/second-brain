#!/usr/bin/env python3
"""演示引用评估增强功能的使用。"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from eval.scripts.grade_testset import (
    get_quote_context,
    quote_matches_source,
    evaluate_quote_relevance,
    _legacy_evaluate_citation_score as evaluate_citation_score,
)


def demo_get_quote_context():
    """演示上下文提取功能。"""
    print("=" * 60)
    print("演示 1: 上下文提取")
    print("=" * 60)

    source_content = """# 动态规划基础

动态规划是一种算法设计技术。它通过把原问题分解为相对简单的子问题的方式求解复杂问题。

## 核心思想

动态规划的核心思想是记忆化搜索，避免重复计算。

## 应用场景

常用于优化问题，如最短路径、背包问题等。"""

    quote = "动态规划是一种算法设计技术"
    context = get_quote_context(quote, source_content, window_size=50)

    if context:
        print(f"引用: {quote}")
        print(f"位置: {context['position']}")
        print(f"所属标题: {context['heading']}")
        print(f"前文: ...{context['before'][-30:]}")
        print(f"后文: {context['after'][:30]}...")
    else:
        print("未找到引用上下文")

    print()


def demo_quote_matches_source():
    """演示引用匹配功能。"""
    print("=" * 60)
    print("演示 2: 引用匹配（带上下文）")
    print("=" * 60)

    # 注意：这里需要实际的源文件才能正常运行
    # 这里只是演示 API 使用方式
    quote = "这是一个示例引用"
    source_path = "example_source.md"

    matches, context = quote_matches_source(quote, source_path)

    if matches:
        print(f"✓ 引用匹配成功")
        if context:
            print(f"  位置: {context['position']}")
            print(f"  标题: {context['heading']}")
        else:
            print(f"  匹配类型: 关键词匹配（无精确位置）")
    else:
        print(f"✗ 引用未找到")

    print()


def demo_evaluate_quote_relevance():
    """演示相关性评估功能。"""
    print("=" * 60)
    print("演示 3: 引用相关性评估")
    print("=" * 60)

    test_cases = [
        {
            "quote": "动态规划是一种算法设计技术，通过分解子问题来求解",
            "question": "什么是动态规划",
            "expected": "高相关性"
        },
        {
            "quote": "今天天气很好，适合出门散步",
            "question": "什么是动态规划",
            "expected": "低相关性"
        },
        {
            "quote": "递归是一种函数调用自身的技术",
            "question": "什么是动态规划",
            "expected": "中等相关性（相关主题）"
        }
    ]

    for i, case in enumerate(test_cases, 1):
        relevance = evaluate_quote_relevance(case["quote"], case["question"])
        print(f"测试 {i}: {case['expected']}")
        print(f"  问题: {case['question']}")
        print(f"  引用: {case['quote']}")
        print(f"  相关性: {relevance:.2f}")
        print()


def demo_evaluate_citation_score():
    """演示完整的引用评分功能。"""
    print("=" * 60)
    print("演示 4: 完整引用评分")
    print("=" * 60)

    question = {
        "query": "什么是动态规划",
        "citation_rules": {
            "require_quote": True,
            "require_source": True
        },
        "expected_sources": ["dynamic_programming.md", "algorithms.md"]
    }

    answer = """动态规划是一种算法设计技术。"动态规划通过分解子问题来求解复杂问题"。

参考来源: dynamic_programming.md"""

    result = evaluate_citation_score(question, answer)

    print(f"综合得分: {result['score']:.2f}")
    print(f"有引用: {result['has_quote']}")
    print(f"有来源: {result['has_source']}")
    print(f"引用有效性: {result['quote_validity_score']:.2f}")
    print(f"上下文匹配: {result['context_match_score']:.2f}")
    print(f"相关性: {result['relevance_score']:.2f}")
    print(f"详细信息: {result['details']}")
    print()

    if result['valid_quotes']:
        print("有效引用:")
        for i, vq in enumerate(result['valid_quotes'], 1):
            print(f"  {i}. {vq['quote'][:50]}...")
            print(f"     相关性: {vq['relevance']:.2f}")
            if vq['context']:
                print(f"     位置: {vq['context']['position']}")

    if result['invalid_quotes']:
        print("无效引用:")
        for i, iq in enumerate(result['invalid_quotes'], 1):
            print(f"  {i}. {iq[:50]}...")

    print()


def main():
    """运行所有演示。"""
    print()
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "引用评估增强功能演示" + " " * 21 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    try:
        demo_get_quote_context()
        demo_quote_matches_source()
        demo_evaluate_quote_relevance()
        demo_evaluate_citation_score()

        print("=" * 60)
        print("演示完成!")
        print("=" * 60)
        print()
        print("提示: 要在实际评估中使用这些功能，请运行:")
        print("  python -u eval/scripts/run_eval_stream.py \\")
        print("    --testset eval/testsets/testset.json \\")
        print("    --base-url http://127.0.0.1:9000 \\")
        print("    --strict-sources \\")
        print("    --report eval/reports/report.json")
        print()

    except Exception as e:
        print(f"演示过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

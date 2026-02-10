#!/usr/bin/env python3
"""测试引用评估增强功能。"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from eval.scripts.grade_testset import (
    get_quote_context,
    quote_matches_source,
    evaluate_quote_relevance,
    _legacy_evaluate_citation_score,
    evaluate_attribution_score,
    get_attribution_mode,
    compute_workflow_compliance,
    compute_fidelity_score,
    extract_key_claims,
)


def test_get_quote_context():
    """测试获取引用上下文。"""
    source_content = """# 标题1

这是一些前文内容。这里有一个重要的引用内容就在这里。后面还有一些文字。

## 标题2

更多内容。"""

    quote = "重要的引用内容"
    context = get_quote_context(quote, source_content, window_size=20)

    assert context is not None, "应该找到引用上下文"
    assert "前文" in context["before"] or "一些" in context["before"], "应该包含前文"
    assert "后面" in context["after"] or "一些" in context["after"], "应该包含后文"
    assert "标题1" in context["heading"] or context["heading"].startswith("#"), "应该找到标题"
    print("✓ test_get_quote_context 通过")


def test_quote_matches_source_with_context():
    """测试带上下文的引用匹配。"""
    # 注意：这个测试需要实际的源文档，这里只做基本验证
    matches, context = quote_matches_source("不存在的引用", "fake_source.md")
    assert not matches, "不存在的引用应该返回 False"
    assert context is None, "不存在的引用应该没有上下文"
    print("✓ test_quote_matches_source_with_context 通过")


def test_evaluate_quote_relevance():
    """测试引用相关性评估。"""
    # 测试高相关性
    quote = "动态规划是一种算法设计方法"
    question = "什么是动态规划"
    relevance = evaluate_quote_relevance(quote, question)
    assert relevance > 0.3, f"高相关性应该 > 0.3，实际: {relevance}"

    # 测试低相关性
    quote = "天气很好"
    question = "什么是动态规划"
    relevance = evaluate_quote_relevance(quote, question)
    assert relevance < 0.5, f"低相关性应该 < 0.5，实际: {relevance}"

    # 测试空问题
    relevance = evaluate_quote_relevance("任意引用", "")
    assert relevance == 1.0, "空问题应该返回 1.0"

    print("✓ test_evaluate_quote_relevance 通过")


def test_evaluate_citation_score_structure():
    """测试旧版引用评估返回结构。"""
    question = {
        "query": "测试问题",
        "citation_rules": {
            "require_quote": True,
            "require_source": True
        },
        "expected_sources": ["test_source.md"]
    }
    answer = '这是答案，"引用内容"，来源: test_source.md'

    result = _legacy_evaluate_citation_score(question, answer)

    # 检查返回结构
    assert "score" in result, "应该包含 score"
    assert "has_quote" in result, "应该包含 has_quote"
    assert "has_source" in result, "应该包含 has_source"
    assert "quote_validity_score" in result, "应该包含 quote_validity_score"
    assert "context_match_score" in result, "应该包含 context_match_score"
    assert "relevance_score" in result, "应该包含 relevance_score"
    assert "valid_quotes" in result, "应该包含 valid_quotes"
    assert "invalid_quotes" in result, "应该包含 invalid_quotes"
    assert "details" in result, "应该包含 details"

    # 检查 valid_quotes 结构（如果有）
    if result["valid_quotes"]:
        quote_item = result["valid_quotes"][0]
        assert isinstance(quote_item, dict), "valid_quotes 应该是字典列表"
        assert "quote" in quote_item, "应该包含 quote 字段"
        assert "context" in quote_item, "应该包含 context 字段"
        assert "relevance" in quote_item, "应该包含 relevance 字段"

    print("✓ test_evaluate_citation_score_structure 通过")


def test_no_citation_requirement():
    """测试不要求引用的情况。"""
    question = {
        "query": "测试问题",
        "citation_rules": {},
        "expected_sources": []
    }
    answer = "这是答案"

    result = _legacy_evaluate_citation_score(question, answer)

    assert result["score"] == 1.0, "不要求引用应该得满分"
    assert result["context_match_score"] == 1.0, "context_match_score 应该是 1.0"
    assert result["relevance_score"] == 1.0, "relevance_score 应该是 1.0"

    print("✓ test_no_citation_requirement 通过")


def test_multi_source_prefers_exact_match():
    """测试多来源场景下优先选择精确匹配。

    验证修复：当第一个来源只有关键词匹配，但后续来源有精确匹配时，
    应该选择精确匹配的结果（带上下文）。
    """
    from unittest.mock import patch, MagicMock

    # 模拟 quote_matches_source 函数的行为
    # source1: 关键词匹配（无上下文）
    # source2: 精确匹配（有上下文）
    call_count = [0]

    def mock_quote_matches_source(quote, source):
        call_count[0] += 1
        if "source1" in source:
            # 第一个来源：关键词匹配，无上下文
            return True, None
        elif "source2" in source:
            # 第二个来源：精确匹配，有上下文
            return True, {
                "before": "前文",
                "after": "后文",
                "position": 100,
                "heading": "# 标题"
            }
        return False, None

    question = {
        "query": "测试问题",
        "citation_rules": {
            "require_quote": True,
            "require_source": False
        },
        "expected_sources": ["source1.md", "source2.md"]
    }
    answer = '"这是一个测试引用"'

    with patch('eval.scripts.grade_testset.quote_matches_source', side_effect=mock_quote_matches_source):
        result = _legacy_evaluate_citation_score(question, answer)

    # 验证：应该遍历了两个来源（因为第一个没有上下文）
    assert call_count[0] == 2, f"应该检查两个来源，实际检查了 {call_count[0]} 个"

    # 验证：应该选择了有上下文的匹配结果
    assert result["context_match_score"] == 1.0, \
        f"应该选择精确匹配（context_match_score=1.0），实际: {result['context_match_score']}"

    # 验证：valid_quotes 应该包含上下文
    assert len(result["valid_quotes"]) == 1, "应该有一个有效引用"
    assert result["valid_quotes"][0]["context"] is not None, "有效引用应该包含上下文"
    assert result["valid_quotes"][0]["context"]["heading"] == "# 标题", "上下文应该包含正确的标题"

    print("✓ test_multi_source_prefers_exact_match 通过")


def test_get_attribution_mode():
    """测试归因模式推断。"""
    # disabled: 不要求引用
    q1 = {"citation_rules": {}}
    assert get_attribution_mode(q1) == "disabled"

    q2 = {"citation_rules": {"require_quote": False, "require_source": False}}
    assert get_attribution_mode(q2) == "disabled"

    # strict: 同时要求引用和来源
    q3 = {"citation_rules": {"require_quote": True, "require_source": True}}
    assert get_attribution_mode(q3) == "strict"

    # standard: 只要求部分
    q4 = {"citation_rules": {"require_quote": True, "require_source": False}}
    assert get_attribution_mode(q4) == "standard"

    # 新字段优先
    q5 = {"citation_rules": {"attribution_mode": "strict", "require_quote": False}}
    assert get_attribution_mode(q5) == "strict"

    print("✓ test_get_attribution_mode 通过")


def test_evaluate_attribution_disabled():
    """测试 disabled 模式下归因评估得满分。"""
    question = {
        "query": "测试问题",
        "citation_rules": {},
        "expected_sources": []
    }
    result = evaluate_attribution_score(question, "任何回答", None)

    assert result["score"] == 1.0, "disabled 模式应该得满分"
    assert "归因评估已禁用" in result["details"]

    print("✓ test_evaluate_attribution_disabled 通过")


def test_evaluate_attribution_structure():
    """测试归因评估返回结构。"""
    question = {
        "query": "测试问题",
        "citation_rules": {"require_quote": True, "require_source": True},
        "expected_sources": ["test_source.md"]
    }
    tool_events = [
        {
            "tool_name": "query_my_notes",
            "stage": "end",
            "arguments": {"query": "测试"}
        },
        {
            "tool_name": "read_note_file",
            "stage": "end",
            "arguments": {"file_path": "test_source.md"}
        }
    ]
    result = evaluate_attribution_score(question, "测试回答", tool_events)

    assert "score" in result, "应该包含 score"
    assert "workflow" in result, "应该包含 workflow"
    assert "attribution" in result, "应该包含 attribution"
    assert "fidelity" in result, "应该包含 fidelity"
    assert "details" in result, "应该包含 details"

    print("✓ test_evaluate_attribution_structure 通过")


def test_workflow_compliance_no_events():
    """测试无工具调用时的工作流合规性。"""
    question = {"expected_sources": ["test.md"]}
    result = compute_workflow_compliance(question, None)
    assert result["score"] == 0.0, "无工具调用应该得 0 分"

    print("✓ test_workflow_compliance_no_events 通过")


def test_workflow_compliance_full():
    """测试完整工作流的合规性。"""
    question = {"expected_sources": ["test.md"]}
    tool_events = [
        {"tool_name": "query_my_notes", "stage": "end", "arguments": {"query": "test"}},
        {"tool_name": "read_note_file", "stage": "end", "arguments": {"file_path": "test.md"}},
    ]
    result = compute_workflow_compliance(question, tool_events)
    assert result["score"] == 1.0, f"完整工作流应该得满分，实际: {result['score']}"
    assert result["has_query"] is True
    assert result["has_read"] is True
    assert result["expected_sources_coverage"] == 1.0

    print("✓ test_workflow_compliance_full 通过")


def test_fidelity_answered_without_reading():
    """测试未读文件直接回答的忠实度检测。"""
    result = compute_fidelity_score(
        "这是一个很长的详细回答，包含了很多内容，但没有读取任何文档。动态规划是一种通过把原问题分解为相对简单的子问题的方式求解复杂问题的方法。",
        [],  # 空工具事件
        ["test.md"]
    )
    assert result["answered_without_reading"] is True
    assert result["score"] == 0.3, f"未读文件直接回答应该得 0.3 分，实际: {result['score']}"

    print("✓ test_fidelity_answered_without_reading 通过")


def test_extract_key_claims():
    """测试关键陈述提取。"""
    answer = """根据文档记录，动态规划的时间复杂度为 O(n^2)。

空间复杂度是 O(n)。这道题使用了贪心算法来解决。

```python
def solve(n):
    return n * 2
```

短句。

笔记中提到了这个方法。

这是什么意思？"""

    claims = extract_key_claims(answer, min_length=10)

    # 应该过滤掉：代码块、过短句、元信息句、问句
    assert len(claims) > 0, "应该提取到关键陈述"
    assert all(len(c) >= 10 for c in claims), "所有陈述应该 >= 10 字符"
    assert all("```" not in c for c in claims), "不应该包含代码块"
    assert not any("什么意思" in c for c in claims), "不应该包含问句"

    print("✓ test_extract_key_claims 通过")


if __name__ == "__main__":
    print("运行引用评估增强功能测试...")
    print()

    try:
        test_get_quote_context()
        test_quote_matches_source_with_context()
        test_evaluate_quote_relevance()
        test_evaluate_citation_score_structure()
        test_no_citation_requirement()
        test_multi_source_prefers_exact_match()

        # 新增归因评估测试
        test_get_attribution_mode()
        test_evaluate_attribution_disabled()
        test_evaluate_attribution_structure()
        test_workflow_compliance_no_events()
        test_workflow_compliance_full()
        test_fidelity_answered_without_reading()
        test_extract_key_claims()

        print()
        print("=" * 50)
        print("所有测试通过! ✓")
        print("=" * 50)

    except AssertionError as e:
        print()
        print("=" * 50)
        print(f"测试失败: {e}")
        print("=" * 50)
        sys.exit(1)
    except Exception as e:
        print()
        print("=" * 50)
        print(f"测试出错: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 50)
        sys.exit(1)

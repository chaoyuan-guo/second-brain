#!/usr/bin/env python3
"""LLM-as-Judge grader for personalization evaluation testsets.

Uses an LLM to evaluate answers across four dimensions:
- personalization: 个性化命中
- precision: 精准简洁
- honesty: 诚实性
- traceability: 可追溯性

Usage:
  python eval/scripts/grade_by_llm.py --answers answers.json --output report.json
  python eval/scripts/grade_by_llm.py --testset eval/testsets/testset.json --answers answers.json --output report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_API_BASE_URL = "https://td06.openai.azure.com/openai/v1"
DEFAULT_CHAT_MODEL = "gpt-52"
PASS_THRESHOLD = 0.6
DIMENSIONS = ["personalization", "precision", "honesty", "traceability"]

JUDGE_SYSTEM_PROMPT = """\
你是一位严格的评估专家，负责评判一个基于个人笔记的 RAG 系统的回答质量。

你的评估必须基于以下四个维度，每个维度打 1-5 分：

**1. 个性化命中 (personalization)**
- 5分：完全基于用户个人数据（提交记录、笔记原文）回答，内容高度个性化
- 3分：部分引用了用户数据，但也混入了通用知识
- 1分：完全是通用回答，没有任何个性化内容

**2. 精准简洁 (precision)**
- 5分：直击问题核心，无冗余，每句话都有信息量
- 3分：回答了问题但有些冗余或不够聚焦
- 1分：答非所问或极度冗长

**3. 诚实性 (honesty)**
- 5分：所有陈述都有据可查，不确定时明确说明，不编造信息
- 3分：大部分准确但有少量无法验证的陈述
- 1分：编造了不存在的信息，或对不确定的内容表现得很确定

**4. 可追溯性 (traceability)**
- 5分：每个关键结论都标注了来源文件/提交ID，读者可以验证
- 3分：部分结论有来源标注
- 1分：没有任何来源标注

评分时请严格遵循以下原则：
- 参考答案要点仅供参考，不要求答案完全覆盖所有要点
- 重点关注本题的 dimension_focus 指定的维度
- 如果答案中出现了 anti_patterns 中描述的问题，相关维度应扣分
- 你的评分应该有区分度，不要给所有维度都打相同的分"""


def _build_judge_prompt(
    question: Dict[str, Any],
    answer: str,
) -> str:
    """Construct the user prompt for the LLM judge."""
    parts = [
        f"## 用户提问\n{question['query']}",
        f"\n## 系统回答\n{answer}",
        f"\n## 参考答案要点\n" + "\n".join(
            f"- {p}" for p in question.get("reference_key_points", [])
        ),
    ]

    criteria = question.get("evaluation_criteria", {})
    if criteria:
        parts.append("\n## 本题评分标准")
        for dim, desc in criteria.items():
            parts.append(f"- **{dim}**: {desc}")

    anti = question.get("anti_patterns", [])
    if anti:
        parts.append("\n## 反例（出现以下情况应扣分）")
        for a in anti:
            parts.append(f"- {a}")

    focus = question.get("dimension_focus", [])
    if focus:
        parts.append(f"\n## 本题重点维度\n{', '.join(focus)}")

    parts.append(
        "\n## 输出要求\n"
        "请严格输出以下 JSON 格式（不要添加任何其他内容）：\n"
        "```json\n"
        "{\n"
        '  "personalization": <1-5>,\n'
        '  "precision": <1-5>,\n'
        '  "honesty": <1-5>,\n'
        '  "traceability": <1-5>,\n'
        '  "reasoning": "<简要说明评分理由，100字以内>"\n'
        "}\n"
        "```"
    )
    return "\n".join(parts)


def _parse_judge_output(raw: str) -> Dict[str, Any]:
    """Parse the LLM judge output into scores and reasoning."""
    text = raw.strip()
    # Extract JSON from markdown code block if present
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            cleaned = part.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            if cleaned.startswith("{"):
                text = cleaned
                break

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end])
            except json.JSONDecodeError:
                return {
                    "personalization": 1,
                    "precision": 1,
                    "honesty": 1,
                    "traceability": 1,
                    "reasoning": f"Failed to parse judge output: {raw[:200]}",
                    "parse_error": True,
                }
        else:
            return {
                "personalization": 1,
                "precision": 1,
                "honesty": 1,
                "traceability": 1,
                "reasoning": f"Failed to parse judge output: {raw[:200]}",
                "parse_error": True,
            }

    scores: Dict[str, Any] = {}
    for dim in DIMENSIONS:
        val = data.get(dim, 1)
        scores[dim] = max(1, min(5, int(val)))
    scores["reasoning"] = data.get("reasoning", "")
    return scores


def compute_weighted_score(
    scores: Dict[str, int],
    weights: Dict[str, float],
) -> float:
    """Compute weighted total score normalized to 0-1."""
    total = 0.0
    for dim in DIMENSIONS:
        raw = scores.get(dim, 1)
        normalized = (raw - 1) / 4.0  # 1-5 -> 0-1
        total += normalized * weights.get(dim, 0.25)
    return round(total, 4)


def check_retrieval_correctness(
    expected_sources: List[str],
    tool_trace: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Rule-based check for retrieval correctness from tool traces."""
    result: Dict[str, Any] = {
        "expected": expected_sources,
        "retrieved": [],
        "recall": 0.0,
    }
    if not tool_trace or not expected_sources:
        return result

    # Extract retrieved sources from tool_traces
    retrieved: List[str] = []
    events = tool_trace.get("events", [])
    for event in events:
        if event.get("tool_name") == "query_my_notes":
            tool_output = event.get("output", "")
            # Parse sources from tool output
            if isinstance(tool_output, str):
                for src in expected_sources:
                    stem = Path(src).stem
                    if stem in tool_output or src in tool_output:
                        if src not in retrieved:
                            retrieved.append(src)
            elif isinstance(tool_output, dict):
                sources = tool_output.get("sources", [])
                for s in sources:
                    name = s if isinstance(s, str) else s.get("file", "")
                    if name not in retrieved:
                        retrieved.append(name)

    # Also check read_note_file calls
    for event in events:
        if event.get("tool_name") == "read_note_file":
            args = event.get("arguments", {})
            file_path = args.get("file_path", "") or args.get("path", "")
            if isinstance(file_path, str):
                for src in expected_sources:
                    if src in file_path or Path(src).stem in file_path:
                        if src not in retrieved:
                            retrieved.append(src)

    result["retrieved"] = retrieved
    if expected_sources:
        hits = sum(1 for s in expected_sources if s in retrieved)
        result["recall"] = round(hits / len(expected_sources), 2)
    return result


class LLMJudge:
    """LLM-based answer evaluator."""

    def __init__(
        self,
        model: Optional[str] = None,
        concurrency: int = 10,
    ):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required: pip install openai"
            ) from exc

        api_key = os.getenv("SUPER_MIND_API_KEY") or os.getenv("AI_BUILDER_TOKEN") or os.getenv("azure_api_key")
        if not api_key:
            raise RuntimeError(
                "Missing API key: set SUPER_MIND_API_KEY, AI_BUILDER_TOKEN, or azure_api_key"
            )
        base_url = os.getenv("SUPER_MIND_API_BASE_URL") or os.getenv("azure_base_url") or DEFAULT_API_BASE_URL

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model or os.getenv("azure_use_model") or os.getenv("SUPER_MIND_CHAT_MODEL", DEFAULT_CHAT_MODEL)
        self._concurrency = concurrency

    def _call_llm(self, system: str, user: str) -> str:
        """Make a single LLM call."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_completion_tokens=500,
        )
        return (response.choices[0].message.content or "").strip()

    def grade_answer(
        self,
        question: Dict[str, Any],
        answer: str,
        weights: Dict[str, float],
        tool_trace: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Grade a single answer."""
        qid = question.get("id", "unknown")
        print(f"  评分中: {qid} ...", flush=True)

        user_prompt = _build_judge_prompt(question, answer)
        raw_output = self._call_llm(JUDGE_SYSTEM_PROMPT, user_prompt)
        scores = _parse_judge_output(raw_output)

        total_score = compute_weighted_score(scores, weights)
        passed = total_score >= PASS_THRESHOLD

        retrieval_check = check_retrieval_correctness(
            question.get("expected_sources", []),
            tool_trace,
        )

        result = {
            "id": qid,
            "passed": passed,
            "total_score": total_score,
            "scores": {d: scores[d] for d in DIMENSIONS},
            "reasoning": scores.get("reasoning", ""),
            "retrieval_check": retrieval_check,
        }
        if scores.get("parse_error"):
            result["parse_error"] = True

        status = "✓" if passed else "✗"
        print(
            f"  {status} {qid}: {total_score:.2f} "
            f"(P={scores.get('personalization',0)} "
            f"R={scores.get('precision',0)} "
            f"H={scores.get('honesty',0)} "
            f"T={scores.get('traceability',0)})",
            flush=True,
        )
        return result

    def grade_all(
        self,
        testset: Dict[str, Any],
        answers: Dict[str, str],
        tool_traces: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Grade all answers and produce a report."""
        questions = testset.get("questions", [])
        dim_config = testset.get("evaluation_dimensions", {})
        weights = {k: v["weight"] for k, v in dim_config.items()}

        results: List[Dict[str, Any]] = []
        for q in questions:
            qid = q["id"]
            answer = answers.get(qid, "")
            if not answer:
                print(f"  ⚠ 跳过 {qid}: 无答案", flush=True)
                continue
            trace = (tool_traces or {}).get(qid)
            result = self.grade_answer(q, answer, weights, trace)
            results.append(result)

        return self._build_report(testset, results)

    def _build_report(
        self,
        testset: Dict[str, Any],
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build the final report from individual results."""
        meta = testset.get("meta", {})
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        avg_score = (
            sum(r["total_score"] for r in results) / total if total else 0.0
        )

        # Dimension averages
        dim_totals: Dict[str, List[int]] = {d: [] for d in DIMENSIONS}
        for r in results:
            for d in DIMENSIONS:
                dim_totals[d].append(r["scores"].get(d, 0))
        dim_averages = {
            d: round(sum(vals) / len(vals), 2) if vals else 0.0
            for d, vals in dim_totals.items()
        }

        # Category stats
        cat_results: Dict[str, List[Dict[str, Any]]] = {}
        for r in results:
            # Find the question to get its category
            cat = "unknown"
            for q in testset.get("questions", []):
                if q["id"] == r["id"]:
                    cat = q.get("category", "unknown")
                    break
            cat_results.setdefault(cat, []).append(r)

        category_stats = {}
        for cat, cat_res in cat_results.items():
            cat_total = len(cat_res)
            cat_passed = sum(1 for r in cat_res if r["passed"])
            cat_avg = (
                sum(r["total_score"] for r in cat_res) / cat_total
                if cat_total
                else 0.0
            )
            category_stats[cat] = {
                "total": cat_total,
                "passed": cat_passed,
                "pass_rate": round(cat_passed / cat_total, 3) if cat_total else 0.0,
                "avg_score": round(cat_avg, 3),
            }

        return {
            "meta": {
                "testset_name": meta.get("name", "unknown"),
                "scoring_mode": "llm_judge",
                "model": self._model,
                "version": meta.get("version", "unknown"),
            },
            "summary": {
                "total": total,
                "passed": passed,
                "pass_rate": round(passed / total, 3) if total else 0.0,
                "avg_score": round(avg_score, 3),
                "dimension_averages": dim_averages,
            },
            "category_stats": category_stats,
            "results": results,
        }


def load_answers(path: Path) -> Dict[str, str]:
    """Load answers file into {question_id: answer_text} mapping."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        # If it's already a dict, check if it has a specific structure
        if "answers" in data:
            data = data["answers"]
        elif all(isinstance(v, str) for v in data.values()):
            return data

    # Handle list format: [{id: ..., answer: ...}, ...]
    if isinstance(data, list):
        result = {}
        for item in data:
            qid = item.get("id", "")
            # Try different answer field names
            answer = (
                item.get("answer", "")
                or item.get("response", "")
                or item.get("text", "")
            )
            if qid and answer:
                result[qid] = answer
        return result

    return data if isinstance(data, dict) else {}


def load_tool_traces(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Load tool traces file if it exists."""
    if not path or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # Handle list format: convert to dict keyed by question id
        if isinstance(data, list):
            result = {}
            for item in data:
                qid = item.get("id", "")
                if qid:
                    result[qid] = item
            return result
        return data
    except Exception as e:
        print(f"⚠ 加载工具追踪失败: {e}", file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge grader for personalization evaluation"
    )
    parser.add_argument(
        "--testset",
        default="eval/testsets/testset.json",
        help="Path to testset JSON",
    )
    parser.add_argument(
        "--answers",
        required=True,
        help="Path to answers JSON",
    )
    parser.add_argument(
        "--tool-traces",
        help="Path to tool traces JSON (optional)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output report JSON",
    )
    parser.add_argument(
        "--model",
        help="Model for LLM judge (default: from env or gpt-5)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent grading calls",
    )
    args = parser.parse_args()

    # Load testset
    testset_path = Path(args.testset)
    if not testset_path.exists():
        print(f"✗ 测试集文件不存在: {testset_path}", file=sys.stderr)
        sys.exit(1)
    testset = json.loads(testset_path.read_text(encoding="utf-8"))
    print(f"✓ 加载测试集: {testset.get('meta', {}).get('name', 'unknown')} "
          f"({len(testset.get('questions', []))} 题)")

    # Load answers
    answers_path = Path(args.answers)
    if not answers_path.exists():
        print(f"✗ 答案文件不存在: {answers_path}", file=sys.stderr)
        sys.exit(1)
    answers = load_answers(answers_path)
    print(f"✓ 加载答案: {len(answers)} 条")

    # Load tool traces
    tool_traces = None
    if args.tool_traces:
        tool_traces = load_tool_traces(Path(args.tool_traces))
        if tool_traces:
            print(f"✓ 加载工具追踪: {len(tool_traces)} 条")

    # Initialize judge and grade
    judge = LLMJudge(model=args.model, concurrency=args.concurrency)
    print(f"\n开始 LLM 评分 (model={judge._model})...")
    print("=" * 60)

    report = judge.grade_all(testset, answers, tool_traces)

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Print summary
    summary = report["summary"]
    print("=" * 60)
    print(f"\n评分完成!")
    print(f"  总题数: {summary['total']}")
    print(f"  通过数: {summary['passed']}")
    print(f"  通过率: {summary['pass_rate']:.1%}")
    print(f"  平均分: {summary['avg_score']:.3f}")
    print(f"\n  维度平均分:")
    for dim, avg in summary["dimension_averages"].items():
        print(f"    {dim}: {avg:.2f}")

    print(f"\n  分类统计:")
    for cat, stats in report["category_stats"].items():
        print(
            f"    {cat}: {stats['passed']}/{stats['total']} "
            f"({stats['pass_rate']:.1%}, avg={stats['avg_score']:.3f})"
        )

    print(f"\n✓ 报告已保存至: {output_path}")


if __name__ == "__main__":
    main()

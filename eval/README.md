# 评估系统

## 概述

本评估系统基于 **LLM-as-Judge** 方法，评估 RAG 系统对用户个性化学习痕迹的理解和回答能力。

核心理念：大模型本身已内化算法知识，检索与否差别不大。真正有价值的是**用户专属的个性化数据**——错误模式、困惑点、具体错误代码、学习轨迹。评估应围绕这些维度展开。

## 评估维度

| 维度 | 权重 | 说明 |
|------|------|------|
| personalization（个性化命中） | 0.35 | 回答是否基于用户个人数据而非通用知识 |
| precision（精准简洁） | 0.25 | 是否直击问题核心，无冗余 |
| honesty（诚实性） | 0.25 | 是否有据可查，不确定时是否说明 |
| traceability（可追溯性） | 0.15 | 是否标注来源文件/提交ID |

## 题型分布（18 题）

| 类别 | 数量 | 说明 |
|------|------|------|
| error_pattern | 5 | 错误模式识别：分析用户提交记录中的系统性错误 |
| confusion_point | 5 | 困惑点定位：基于笔记中记录的困惑和讨论 |
| code_diagnosis | 4 | 具体错误代码诊断：针对特定提交的 bug 分析 |
| learning_trajectory | 4 | 学习轨迹分析：练习时间跨度、通过率、习惯 |

## 评分机制

- 每个维度 1-5 分，归一化到 0-1：`(score - 1) / 4`
- 加权总分：`personalization×0.35 + precision×0.25 + honesty×0.25 + traceability×0.15`
- 通过阈值：总分 ≥ 0.6
- 检索正确性作为补充信息记录，不影响通过判定

## 使用方法

### 运行完整评估（生成答案 + 评分）

```bash
# 启动后端服务
./start_services.sh start backend

# 运行评估
python -u eval/scripts/run_eval_stream.py \
  --base-url http://127.0.0.1:9000 \
  --concurrency 10 \
  --report eval/reports/report.json \
  2>&1 | tee eval/reports/eval.log
```

### 只运行指定题目

```bash
python -u eval/scripts/run_eval_stream.py \
  --question-ids Q01,Q14 \
  --base-url http://127.0.0.1:9000 \
  --report eval/reports/report.json
```

### 仅评分已有答案

```bash
python eval/scripts/grade_by_llm.py \
  --answers eval/reports/answers.json \
  --output eval/reports/report.json
```

### 指定评分模型

```bash
python eval/scripts/grade_by_llm.py \
  --answers eval/reports/answers.json \
  --tool-traces eval/reports/answers_tool_traces.json \
  --output eval/reports/report.json \
  --model gpt-5
```

## 报告格式

```json
{
  "meta": {
    "testset_name": "personalization_eval",
    "scoring_mode": "llm_judge",
    "model": "gpt-5",
    "version": "1.0"
  },
  "summary": {
    "total": 18,
    "passed": 15,
    "pass_rate": 0.833,
    "avg_score": 0.812,
    "dimension_averages": {
      "personalization": 4.1,
      "precision": 3.8,
      "honesty": 4.5,
      "traceability": 3.6
    }
  },
  "category_stats": {
    "error_pattern":       { "total": 5, "passed": 4, "pass_rate": 0.8, "avg_score": 0.78 },
    "confusion_point":     { "total": 5, "passed": 4, "pass_rate": 0.8, "avg_score": 0.80 },
    "code_diagnosis":      { "total": 4, "passed": 3, "pass_rate": 0.75, "avg_score": 0.76 },
    "learning_trajectory": { "total": 4, "passed": 4, "pass_rate": 1.0, "avg_score": 0.88 }
  },
  "results": [
    {
      "id": "Q01_coin_change_error_pattern",
      "passed": true,
      "total_score": 0.85,
      "scores": { "personalization": 5, "precision": 4, "honesty": 4, "traceability": 4 },
      "reasoning": "回答基于用户实际提交记录，引用了具体提交 ID...",
      "retrieval_check": { "expected": [...], "retrieved": [...], "recall": 1.0 }
    }
  ]
}
```

## 文件清单

| 文件 | 说明 |
|------|------|
| `eval/testsets/testset.json` | 评估集（18 道题，4 个维度） |
| `eval/testsets/testset.md` | 评估集可读文档 |
| `eval/scripts/grade_by_llm.py` | LLM-as-Judge 评分脚本 |
| `eval/scripts/run_eval_stream.py` | 评估运行脚本（生成答案 + 调用评分） |
| `eval/README.md` | 本文档 |

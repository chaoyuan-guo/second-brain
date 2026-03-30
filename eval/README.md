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
# 启动容器（评估需要直连 OpenCode，建议暴露 9090）
docker rm -f second_brain_opencode 2>/dev/null || true
docker build -t second_brain:opencode -f docker/Dockerfile.opencode .
docker run -d --name second_brain_opencode --restart unless-stopped \
  -p 9080:9080 -p 9090:9090 \
  --env-file .env.docker \
  -v "$PWD/data:/app/data" \
  second_brain:opencode

# 运行评估
./.venv/bin/python -u eval/scripts/run_eval_stream.py \
  --base-url http://127.0.0.1:9090 \
  --concurrency 10 \
  --report eval/reports/report.json \
  2>&1 | tee eval/reports/eval.log
```

补充说明：

- `9090` 是 OpenCode 直连端口，判断它是否活着时，不要用 `curl -f http://127.0.0.1:9090` 这种“只认 2xx”的探测。
- 根路径返回 `401/403/404/405` 也说明 HTTP 服务已经有响应，不等于 down。
- 更可靠的判断方式：
  - 看 `docker ps` 是否映射了 `9090->9090`
  - 直接运行 `run_eval_stream.py`，脚本会在启动前打印连通性探测结果

### 只运行指定题目

```bash
./.venv/bin/python -u eval/scripts/run_eval_stream.py \
  --question-ids Q01,Q14 \
  --base-url http://127.0.0.1:9090 \
  --report eval/reports/report.json
```

### 仅评分已有答案

```bash
./.venv/bin/python eval/scripts/grade_by_llm.py \
  --answers eval/reports/answers.json \
  --output eval/reports/report.json
```

### 指定评分模型

```bash
./.venv/bin/python eval/scripts/grade_by_llm.py \
  --answers eval/reports/answers.json \
  --tool-traces eval/reports/answers_tool_traces.json \
  --output eval/reports/report.json \
  --model gpt-5.4
```

## 报告格式

```json
{
  "meta": {
    "testset_name": "personalization_eval",
    "scoring_mode": "llm_judge",
    "model": "gpt-5.4",
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
    },
    "metrics": {
      "inline_citation_coverage": 0.78,
      "citation_accuracy": 0.92,
      "honesty_trigger_precision": 0.85
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

# 评估系统

## 概述

本评估系统用于测试 Agentic RAG 系统的能力，采用部分得分机制进行多维度评估。

## 题型分布

| 题型 | 数量 | 占比 | 说明 |
|------|------|------|------|
| understanding | 12 | 48% | 单文档知识理解 |
| reasoning | 6 | 24% | 跨文档推理 |
| negative | 5 | 20% | 诚实性测试（应回答不知道） |
| statistics | 2 | 8% | 统计分析（需代码执行） |
| **总计** | **25** | **100%** | |

## 评分机制

### 部分得分

每道题返回 0~1 的连续得分：
- `content_score`: 内容得分
- `citation_score`: 引用得分
- `total_score`: 加权总分
- `passed`: 是否通过（total_score >= 0.6）

### 内容得分计算

```
content_score = (matched_must_have_weight + matched_evidence_weight) / total_weight + bonus_from_should_have
```

- `must_have`: 必须包含的关键词，按权重计分
- `should_have`: 加分项，匹配则加分
- `evidence`: 原文证据，按权重计分

### Unknown 检测

- 排除 "虽然没有...但是..." 这类结构，避免误判
- 答案开头的 unknown 表达置信度更高

## 使用方法

### 运行完整评估

```bash
# 启动后端服务
./start_services.sh start backend

# 运行评估
python eval/scripts/run_eval_stream.py \
  --base-url http://127.0.0.1:9000 \
  --out eval/reports/answers.json \
  --report eval/reports/report.json \
  --concurrency 5
```

### 仅评分已有答案

```bash
python eval/scripts/grade_testset.py \
  --answers eval/reports/answers.json \
  --output eval/reports/report.json
```

## 报告格式

```json
{
  "meta": {
    "scoring_mode": "partial"
  },
  "summary": {
    "total": 25,
    "passed": 20,
    "pass_rate": 0.80,
    "avg_score": 0.82,
    "perfect_count": 15,
    "perfect_rate": 0.60
  },
  "category_stats": {
    "understanding": {"total": 12, "passed": 10, "avg_score": 0.85},
    "reasoning": {"total": 6, "passed": 4, "avg_score": 0.78},
    "negative": {"total": 5, "passed": 5, "avg_score": 0.90},
    "statistics": {"total": 2, "passed": 1, "avg_score": 0.65}
  },
  "results": [...],
  "recall_summary": {...}
}
```

## 文件清单

| 文件 | 说明 |
|------|------|
| `eval/testsets/testset.json` | 评估集 |
| `eval/scripts/gen_testset.py` | 评估集生成脚本 |
| `eval/scripts/grade_testset.py` | 评分脚本 |
| `eval/scripts/run_eval_stream.py` | 运行脚本 |
| `eval/README.md` | 本文档 |

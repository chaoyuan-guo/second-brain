# 评估系统

## 概述

本评估系统用于测试 Agentic RAG 系统的能力，采用部分得分机制进行多维度评估。

## 题型分布

| 题型 | 数量 | 说明 |
|------|------|------|
| understanding | 22 | 单文档知识理解 |
| reasoning | 11 | 跨文档推理、多步分析 |
| negative | 6 | 诚实性测试（应回答不知道） |
| statistics | 4 | 统计分析（需代码执行） |
| skill | 1 | 技能加载测试 |
| web_search | 2 | 联网搜索测试 |
| **总计** | **52** | |

## 评分机制

### 部分得分

每道题返回 0~1 的连续得分：
- `retrieval_score`: 检索召回得分 (recall@k)
- `content_score`: 内容得分
- `citation_score`: 引用得分（含引用正确性验证）
- `tool_score`: 工具调用行为得分
- `total_score`: 加权总分
- `passed`: 是否通过

### 通过阈值（按题型）

| 题型 | 阈值 | 说明 |
|------|------|------|
| understanding | 0.7 | 标准阈值 |
| reasoning | 0.65 | 允许部分推理错误 |
| negative | 0.8 | 对诚实性要求更高 |
| statistics | 0.7 | 标准阈值 |
| skill | 0.7 | 标准阈值 |
| web_search | 0.6 | 联网搜索结果不确定性较高 |
| default | 0.7 | 默认阈值 |

### 内容得分计算

```
content_score = (matched_must_have_weight + matched_evidence_weight + numeric_score) / total_weight + bonus_from_should_have
```

- `must_have`: 必须包含的关键词，按权重计分
- `should_have`: 加分项，匹配则加分
- `evidence`: 原文证据，按权重计分
- `numeric_validations`: 数值校验，支持多种格式匹配

### 引用得分计算

新增引用正确性验证：
- 检查引用内容是否真实存在于源文档中
- 有效引用率作为得分依据
- 支持模糊匹配，允许空白字符差异

### Negative 题目评估

增强的 Negative 题目评估逻辑：
- `correct_rejection`: 尝试检索后回答不知道，满分
- `lazy_rejection`: 未尝试检索就回答不知道，扣 50%
- `hallucination`: 应回答不知道但给出了答案，0 分

### 工具行为评估

- 期望工具调用检查
- **冗余调用检测**: 相同工具+参数的重复调用会扣分
- **调用顺序检查**: 检索应在代码执行之前，违规会扣分
- 错误处理：有错误时扣 20%

### Unknown 检测

- 排除 "虽然没有...但是..." 这类结构，避免误判
- 答案开头的 unknown 表达置信度更高

## 使用方法

### 运行完整评估

```bash
# 启动后端服务
./start_services.sh start backend

# 运行评估（自动生成答案并评分）
python eval/scripts/run_eval_stream.py \
  --base-url http://127.0.0.1:9000 \
  --out eval/reports/answers.json \
  --report eval/reports/report.json \
  --strict-sources \
  --recall-k 1,3,5,10 \
  --concurrency 5
```

### 仅评分已有答案

```bash
python eval/scripts/grade_testset.py \
  --testset eval/testsets/testset.json \
  --answers eval/reports/answers.json \
  --output eval/reports/report.json \
  --tool-traces eval/reports/answers_tool_traces.json \
  --recall-k 1,3,5,10
```

## 报告格式

```json
{
  "meta": {
    "testset_name": "agentic_rag_eval",
    "scoring_mode": "partial"
  },
  "summary": {
    "total": 52,
    "passed": 42,
    "pass_rate": 0.81,
    "avg_score": 0.78,
    "perfect_count": 30,
    "perfect_rate": 0.58
  },
  "category_stats": {
    "understanding": {"total": 22, "passed": 18, "avg_score": 0.82},
    "reasoning": {"total": 11, "passed": 8, "avg_score": 0.75},
    "negative": {"total": 6, "passed": 5, "avg_score": 0.85},
    "statistics": {"total": 4, "passed": 3, "avg_score": 0.72}
  },
  "results": [
    {
      "id": "Q01_xxx",
      "passed": true,
      "total_score": 0.85,
      "pass_threshold": 0.7,
      "retrieval_score": 1.0,
      "content_score": 0.8,
      "citation_score": 0.9,
      "tool_score": 1.0,
      "details": {
        "content": {
          "matched_must_have": [...],
          "semantic_matches": [...]
        },
        "citation": {
          "quote_validity_score": 0.9,
          "valid_quotes": [...],
          "invalid_quotes": [...]
        },
        "tool": {
          "redundancy_ratio": 0.0,
          "order_violations": 0.0
        }
      }
    }
  ],
  "recall_summary": {
    "5": {"mean_recall": 0.92, "count": 52}
  }
}
```

## 文件清单

| 文件 | 说明 |
|------|------|
| `eval/testsets/testset.json` | 评估集（52 道题） |
| `eval/scripts/grade_testset.py` | 评分脚本 |
| `eval/scripts/run_eval_stream.py` | 运行脚本 |
| `eval/README.md` | 本文档 |

## 更新日志

### v2 (2026-02)
- 新增引用正确性验证
- 增强 Negative 题目评估（检测 lazy_rejection）
- 新增工具调用质量评估（冗余检测、顺序检查）
- 按题型设置不同通过阈值
- 新增 5 道复杂推理题型

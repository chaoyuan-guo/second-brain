# Feature Index

`docs/features/` 下的 feature 目录直接带顺序编号，按完成顺序排序。

命名规则统一为 `NN_feature_name`，其中 `NN` 表示完成顺序。

## 已完成 Feature 顺序

| 编号 | 状态 | Feature | 目录 | 入口文档 | 说明 |
|---|---|---|---|---|---|
| F01 | 已完成 | `precise_traceability` | `docs/features/01_precise_traceability/` | `docs/features/01_precise_traceability/redesign.md` | 打通精准证据链路与原文定位，建立 native traceability 主路径 |
| F02 | 已完成 | `traceability_effect_optimization` | `docs/features/02_traceability_effect_optimization/` | `docs/features/02_traceability_effect_optimization/product_decision_memo.md` | 在 native 主路径稳定后，继续优化 citation 首屏证据确认体验 |
| F03 | 已完成 | `answer_engine_rearchitecture` | `docs/features/03_answer_engine_rearchitecture/` | `docs/features/03_answer_engine_rearchitecture/product_decision_memo.md` | 将默认结果页重构为 Answer First、Evidence Second、Process Last |

## 维护规则

1. 新 feature 完成并收口后，在本文件尾部追加下一个编号。
2. 目录名使用 `NN_feature_name`，例如 `04_new_feature`。
3. 如果 feature 之间存在明显承接关系，在该 feature 主文档抬头补一行 `上一阶段成果`。

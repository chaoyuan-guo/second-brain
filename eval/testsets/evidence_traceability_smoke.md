# 证据可追溯性 Smoke Testset

> 对应 JSON：`eval/testsets/evidence_traceability_smoke.json`

## 样本映射

- A01 -> 原评估集 `Q08_backtrack_path_pop`
- A02 -> 原评估集 `Q06_backtrack_param_design`
- A03 -> 原评估集 `Q07_bfs_step_counting`
- A04 -> 原评估集 `Q11_palindrome_subseq_submission_690994503`
- A05 -> 原评估集 `Q14_coin_change_two_errors_0106`
- A06 -> 新增 no-hit 样本：树状数组

## 使用目的

这组样本只做最小闭环验收：

1. 验证常见命中问题是否拿到 `precise/native`
2. 验证补偿定位和文件级来源是否被清晰降级
3. 验证 no-hit 场景是否诚实，不生成伪 citation

## 推荐执行

```bash
python eval/scripts/run_eval_stream.py \
  --testset eval/testsets/evidence_traceability_smoke.json \
  --base-url http://127.0.0.1:9090 \
  --concurrency 3 \
  --out eval/reports/evidence_traceability_smoke_answers.json \
  --report eval/reports/evidence_traceability_smoke_report.json
```

说明：

- `9090` 根路径的 `401/403/404/405` 都算“端口活着，HTTP 已响应”。
- 不要把“非 2xx”直接等同于“容器没启动”。

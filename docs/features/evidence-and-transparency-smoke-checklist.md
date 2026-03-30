# 证据可追溯性 Smoke 验收清单

> 用途：A01-A06 最小闭环验收
> 对应测试集：`eval/testsets/evidence_traceability_smoke.json`
> 关联契约：`docs/features/evidence-and-transparency-source-refs-contract.md`

## 1. 执行命令

```bash
python eval/scripts/run_eval_stream.py \
  --testset eval/testsets/evidence_traceability_smoke.json \
  --base-url http://127.0.0.1:9090 \
  --concurrency 3 \
  --out eval/reports/evidence_traceability_smoke_answers.json \
  --report eval/reports/evidence_traceability_smoke_report.json
```

补充：

- `9090` 根路径出现 `401/403/404/405` 仍然表示服务有响应。
- 不要用 `curl -f http://127.0.0.1:9090` 作为唯一判断条件。
- 优先看脚本自己的连通性探测输出。

## 2. 人工走查项

每题至少记录以下 5 项：

- 是否有 inline citation
- 是否有原生 `source_refs`
- `path / citation_id / snippet / char_offset` 是否齐全
- 点击后是否能直接看到支撑片段
- 是否出现伪 citation、错链或文件级来源冒充精准证据

## 3. 通过口径

- A01-A05：优先看 `precise/native`
- A06：必须先明确没有相关记录，且不生成伪 citation
- `precise/synthetic_read` 与 `file/*` 都可以保留，但只算降级路径

## 4. 记录模板

| ID | inline citation | native source_refs | 四字段齐全 | 点击直达片段 | 结论 |
|---|---|---|---|---|---|
| A01 |  |  |  |  |  |
| A02 |  |  |  |  |  |
| A03 |  |  |  |  |  |
| A04 |  |  |  |  |  |
| A05 |  |  |  |  |  |
| A06 |  |  |  |  |  |

# `source_refs` 最小字段契约

> 用途：为证据可追溯性主链路定义最低契约
> 关联：`docs/features/precise_traceability/rd_plan.md`

## 1. 目标

这份文档只回答一个问题：什么样的 `source_refs` 才能被视为精准片段证据。

## 2. 字段要求

### 必填字段

- `path`
  - 含义：来源文件路径
  - 约束：非空字符串
  - 作用：所有引用链路的最低要求

### 精准片段证据所需字段

只有同时具备以下 3 个字段，才算精准片段证据：

- `citation_id`
  - 含义：回答中的 `[cxx]` 与证据片段的稳定绑定 ID
- `snippet`
  - 含义：支撑当前回答的原文片段
- `char_offset`
  - 含义：该片段在原文中的字符偏移

## 3. 可选增强字段

- `heading`
  - 含义：命中片段所在章节
- `source_title`
  - 含义：更友好的来源标题
- `score`
  - 含义：检索距离或相关性分数

这些字段可以增强展示和排序，但不决定是否属于精准片段证据。

## 4. 分类规则

### 精准片段证据

满足以下条件：

- 有 `path`
- 有 `citation_id`
- 有 `snippet`
- 有 `char_offset`

前端语义：

- `kind = precise`
- 若来自 OpenCode 原生 `source_refs`，则 `provenance = native`

### 降级来源

只要缺少 `citation_id / snippet / char_offset` 中任意一个，就视为降级来源。

前端语义：

- `kind = file`
- `provenance` 按来源区分：
  - `native`：原生 `source_refs` 字段不完整
  - `synthetic_read`：读取调用补偿出的来源
  - `content_path`：从回答正文中提取的文件路径

## 5. 验收口径

以下表现才算精准回溯达标：

1. 回答中出现可点击 inline citation
2. 该 citation 绑定的是 `precise/native`
3. 点击后能基于 `char_offset + snippet` 直接落到支撑片段附近

以下表现都只能算降级路径，不能计入精准回溯达标：

- 只有文件路径，没有片段定位
- 只有 snippet，没有稳定 `citation_id`
- 只有读取补偿生成的定位

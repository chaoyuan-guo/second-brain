# Second Brain V2 研发设计方案

> 用途：指导 Answer Engine 结果页重构实施  
> 产品定义：`docs/features/answer_engine_rearchitecture/product_decision_memo.md`

---

## 1. 本期目标

本期只交付 4 个结果：

1. `回答` 回到主路径
2. `关键依据` 默认只展示 `2-4` 条
3. `过程层` 默认后置
4. `长任务` 不再锁死用户

本期不做：

- 不重写检索链路
- 不重写 SSE 主链路
- 不重做原文预览能力
- 不增强默认调试能力
- 不做 `继续追问`

---

## 2. 设计边界

当前实现是重要参考，不是设计边界。

允许重构局部前端结构，但不能做没这些能力：

- 回答展示
- 关键依据展示
- 原文定位预览
- 流式会话

本期优先改这些文件：

- `frontend/src/app/page.tsx`
- `frontend/src/app/components/chat/AnswerPanel.tsx`
- `frontend/src/app/components/chat/ReferencesPanel.tsx`
- `frontend/src/app/components/chat/ProcessPanel.tsx`
- `frontend/src/app/components/chat/ProcessOverviewBar.tsx`
- `frontend/src/app/hooks/useChatSessions.ts`

---

## 3. 关键设计决策

### 3.1 回答显示规则

- 生成中也允许显示回答
- 首选字段是 `message.content`
- `directAnswer`、`fullAnalysis`、`references` 仍视为终态增强字段
- 如果没有任何正文文本，不允许伪造阶段性回答

### 3.2 长任务状态机

前端单独维护展示态，不要求后端先补字段。

| 展示态 | 进入条件 | 页面必须出现 | 页面禁止出现 |
| --- | --- | --- | --- |
| `running` | 提交后 `< 5s` 且未结束 | 回答区或摘要条 | 默认展开过程流水 |
| `long_running` | `>= 5s` 且未结束 | 已确认回答或单条摘要条 | 锁死输入框 |
| `completed` | final event 且完成 | 回答、依据 | 系统内部术语 |
| `partial_completed` | final event 且部分完成 | 回答、依据或证据不足文案 | `步骤上限` 等内部词 |
| `failed` | final event 且失败 | 失败说明、重试入口 | 过程流水占主视觉 |

补充规则：

- `5s` 计时只和是否完成有关，不因流式正文持续到来而重置
- `long_running` 有正文时显示正文，没有正文时只显示单条摘要条
- 本期不支持同会话并发提问
- 本期必须支持继续编辑草稿、切换会话、离开当前页

### 3.3 默认证据选择规则

证据区默认只展示 `2-4` 条 citation item，计数单位不是 source group。

默认规则固定为：

1. 去重  
   去重键：`sourcePath + charOffsetStart + snippet`

2. 按可信优先级排序  
   - `precise + native + 有 charOffsetStart 或 snippet`
   - `precise + synthetic_read + 有 charOffsetStart 或 snippet`
   - `file + native/content_path`

3. 同优先级内再排序  
   - `retrievalScore` 有值的排前面
   - `retrievalScore` 越小越靠前
   - 分数相同时按 citation 顺序

4. 默认视图优先保证来源去重  
   - 先每个 `sourcePath` 取 1 条
   - 不足 `2` 条时再从剩余候选里补齐

5. 证据不足时允许少于 `2` 条  
   不允许为了凑数展示弱证据

默认态明确禁止：

- 排序按钮
- 质量统计
- 强标签：`精准片段`、`原生精准`、`高相关`

---

## 4. 模块改造

### 4.1 `frontend/src/app/hooks/useChatSessions.ts`

- 增加展示态：`running / long_running / completed / partial_completed / failed`
- 提交后启动 `5s` 长任务计时
- 生成中回答首选 `message.content`
- 把 `statusText` 改写成用户语言

### 4.2 `frontend/src/app/page.tsx`

- 去掉 assistant 消息对 `!message.isThinking` 的回答阻断
- 固定默认顺序为 `AnswerPanel -> ReferencesPanel -> ProcessPanel`
- composer 不再以“等待当前回复完成...”作为主状态
- pending 中允许继续编辑草稿和切换会话

### 4.3 `frontend/src/app/components/chat/AnswerPanel.tsx`

- 生成中只要已有正文就继续显示
- 不再内嵌 `ReferencesPanel`
- `完整分析` 保持折叠
- `partial_completed` 用产品化文案收口
- 不再把系统限制直接展示给用户

### 4.4 `frontend/src/app/components/chat/ReferencesPanel.tsx`

- 实现默认简化模式
- 按第 `3.3` 节规则筛选默认证据
- 默认只展示 `2-4` 条
- 保留原文定位

### 4.5 `frontend/src/app/components/chat/ProcessPanel.tsx`

- 去掉 `message.isThinking -> 自动展开`
- 默认永远折叠
- 默认态只保留摘要条与展开入口
- 调试详情不再属于默认路径

### 4.6 `frontend/src/app/components/chat/ProcessOverviewBar.tsx`

- 保留单条形态
- 改成人话摘要
- 去掉默认警告数、错误数等专家信息

---

## 5. 任务看板

按依赖拓扑拆成 4 批。每一批都必须完成后再进下一批。

当前进度：截至 `2026-04-01`，前三批已完成，第四批已完成主要验证与回归修复，但仍建议补一轮更严格的 `failed` 真链路手工验收。

### 第一批：状态与展示态

状态：`已完成`  
依赖：无

文件：

- `frontend/src/app/hooks/useChatSessions.ts`

任务：

- [x] 增加 `running / long_running / completed / partial_completed / failed`
- [x] 增加 `5s` 长任务切换
- [x] 把内部状态文案改成用户语言

完成信号：

- [x] 前端能区分 `running` 和 `long_running`
- [x] 默认主路径不再出现内部术语

### 第二批：结果页主结构

状态：`已完成`  
依赖：第一批完成

文件：

- `frontend/src/app/page.tsx`
- `frontend/src/app/components/chat/AnswerPanel.tsx`
- `frontend/src/app/components/chat/ProcessPanel.tsx`
- `frontend/src/app/components/chat/ProcessOverviewBar.tsx`

任务：

- [x] 去掉回答阻断
- [x] 固定默认顺序为 `AnswerPanel -> ReferencesPanel -> ProcessPanel`
- [x] 放开 pending 中的草稿编辑和会话切换
- [x] 保持生成中回答可见
- [x] 去掉 `AnswerPanel` 对 `ReferencesPanel` 的内嵌
- [x] 去掉 `ProcessPanel` 的思考态自动展开
- [x] 把 `ProcessOverviewBar` 改成单条用户摘要

完成信号：

- [x] 生成中有正文时回答仍可见
- [x] 过程层默认不展开
- [x] 输入框不再被等待态锁死

### 第三批：依据区收紧

状态：`已完成`  
依赖：第二批完成

文件：

- `frontend/src/app/components/chat/ReferencesPanel.tsx`

任务：

- [x] 实现默认简化模式
- [x] 按第 `3.3` 节规则筛选默认证据
- [x] 默认隐藏排序、统计、强标签

完成信号：

- [x] 默认依据只出 `2-4` 条
- [x] 默认无排序、统计、强标签

### 第四批：验证与收口

状态：`基本完成`  
依赖：第三批完成

文件：

- `frontend/src/__tests__/event-adapter.test.ts`
- `frontend/src/__tests__/ProcessPanel.test.tsx`
- `frontend/src/__tests__/AnswerPanel.test.tsx`
- 结果页编排相关测试
- 长任务态相关测试

任务：

- [x] 重写与旧默认展开过程相关的测试
- [x] 补充结果页主路径测试
- [x] 补充证据默认筛选测试
- [x] 补充 `5s` 长任务态相关规则测试
- [x] 跑完整手工 smoke
- [x] 修复“停止当前回答时误发草稿”的回归问题

完成信号：

- [x] 旧预期已同步替换
- [x] 新主路径有测试保护
- [x] 手工 smoke 通过
- [ ] `failed` 真链路的手工验收仍建议单独再跑一轮

---

## 6. 验收

### 6.1 产品验收

研发完成后，只看这 4 条：

1. 首屏先看到 `回答`
2. 默认只看到 `2-4` 条关键依据
3. 默认只看到一条过程摘要条
4. 超过 `5s` 不再锁死用户

当前结果：`4 / 4` 已达成。

### 6.2 手工验收 Case

至少跑下面 7 个 case：

1. `正常完成`  
   回答在首屏主路径，依据默认 `2-4` 条，过程层默认折叠。

2. `生成中已有正文`  
   final event 到来前如果已有正文，回答区持续可见。

3. `长任务无正文`  
   超过 `5s` 仍无正文时进入 `long_running`，只显示单条摘要条，不显示空壳回答卡，输入框仍可编辑。

4. `长任务有正文`  
   超过 `5s` 且已有正文时进入 `long_running`，已确认回答继续可见，用户仍可切换会话和离开当前页。

5. `部分完成`  
   展示已确认回答和证据不足文案，不出现 `步骤上限`、`工具失败` 等内部词。

6. `失败态`  
   展示失败说明和重试入口，过程流水不抢首屏，输入区不被锁死。

7. `依据与原文定位`  
   默认依据不超过 `4` 条；没有排序按钮、质量统计、强标签；每条依据都能一键打开原文定位。

当前结果：

- [x] `正常完成`
- [x] `生成中已有正文`
- [x] `长任务无正文`
- [x] `长任务有正文`
- [x] `部分完成`
- [ ] `失败态`
- [x] `依据与原文定位`

### 6.3 自动化测试覆盖对象

至少覆盖下面 6 类断言：

1. `useChatSessions.ts` 在 `5s` 后切换到 `long_running`
2. 生成中已有正文时回答区仍渲染
3. `ProcessPanel` 默认不因 `isThinking` 自动展开
4. `ReferencesPanel` 默认按第 `3.3` 节规则筛选并截断到 `2-4` 条
5. 默认主路径不渲染排序、统计、强标签
6. `partial_completed` / `failed` 不再展示内部术语

当前结果：

- [x] `ProcessPanel` 默认不自动展开
- [x] `ReferencesPanel` 默认筛选并截断到 `2-4` 条
- [x] 默认主路径不渲染排序、统计、强标签
- [x] `long_running` 用户文案规则有自动化保护
- [x] 生成中已有正文可见规则有自动化保护
- [x] `failed` 不直接展示内部术语有自动化保护

### 6.4 交付物

本期交付必须同时包含：

- 自动化测试结果
- 手工 smoke 记录
- `running / long_running / completed / partial_completed` 至少 4 个状态的截图或录屏
- 文案禁用词检查结果

当前结果：

- [x] 自动化测试结果
- [x] 手工 smoke 记录
- [x] `running / long_running / completed / partial_completed` 截图
- [x] 文案禁用词检查结果

---

## 7. 一句话收口

这份方案的目标不是“把现有页面再修一修”，而是把结果页改成一个能直接消费的答案页：先给回答，再给少量可信依据，过程层最后出现。

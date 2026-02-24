# 可溯源性改进实施计划

## Context

当前 Second Brain 系统已在后端收集了结构化的来源数据（`used_sources` 列表），并在每次对话结束时通过 NDJSON 流 emit `sources` 事件。但这些数据在到达用户之前就断掉了：前端的 `StreamEvent` 类型联合体没有 `sources` 变体，`ChatMessage` 接口没有 `sources` 字段，事件处理回调对 `sources` 事件零处理，用户只能靠 LLM 在文本中自行写出路径字符串来感知来源。

本计划目标：打通"来源已收集 → 前端结构化展示"这条链路，并补齐 `web_search`/`read_page` 工具的来源收集缺口。

---

## 阶段一：前端接收并展示来源（最高优先级）

**范围：纯前端改动，后端无需修改。**

后端已正确 emit `sources` 事件（`chat.py` 行 1422-1434），格式为：
```json
{"type": "sources", "sources": ["data/notes/my_markdowns/foo.md"], "ts": ...}
```

### 1.1 `chat-types.ts` — 添加类型定义

**文件：** `frontend/src/app/lib/chat-types.ts`

- 在 `StreamEvent` 联合类型末尾（行 34 后）追加 `sources` 变体：
  ```typescript
  | {
      type: 'sources';
      sources: string[];
      expected_sources?: string[];
      question_id?: string;
      ts?: number;
    }
  ```
- 在 `ChatMessage` 接口（行 3-12）末尾追加可选字段：
  ```typescript
  sources?: string[];
  ```
  > 注意：设为可选，确保 localStorage 中已存储的旧会话反序列化不报错（向后兼容）。

### 1.2 `useChatSessions.ts` — 处理 `sources` 事件

**文件：** `frontend/src/app/hooks/useChatSessions.ts`

在行 525 的 `if (event.type === 'done')` 块之前，插入 `sources` 事件处理：
```typescript
if (event.type === 'sources') {
  if (event.sources && event.sources.length > 0) {
    updateAssistantMessage(targetSessionId, assistantPlaceholder.id, (prev) => ({
      ...prev,
      sources: event.sources,
    }));
  }
  return;
}
```
> `updateAssistantMessage` 函数（行 237-256）已有正确的 immutable 更新模式，直接复用。
> `sources` 事件由后端在最终答案生成后、`done` 事件之前 emit，时序正确。

### 1.3 `page.tsx` — 渲染来源引用区块

**文件：** `frontend/src/app/page.tsx`

在行 461-466 的 `statusText` 渲染块之后，`message-meta` div 之前，插入来源区块：
```tsx
{message.role === 'assistant' && message.sources && message.sources.length > 0 && !message.isThinking && (
  <div className="sources-panel">
    <p className="sources-label">来源文件</p>
    <ul className="sources-list">
      {message.sources.map((sourcePath) => {
        const isUrl = sourcePath.startsWith('http://') || sourcePath.startsWith('https://');
        const fileName = isUrl ? sourcePath : (sourcePath.split('/').pop() ?? sourcePath);
        return (
          <li key={sourcePath} className="source-item">
            {isUrl ? (
              <a href={sourcePath} target="_blank" rel="noopener noreferrer" className="source-link">
                {sourcePath}
              </a>
            ) : (
              <>
                <span className="source-filename">{fileName}</span>
                <span className="source-path">{sourcePath}</span>
              </>
            )}
          </li>
        );
      })}
    </ul>
  </div>
)}
```
> 条件 `!message.isThinking`：避免流式生成过程中提前渲染。
> 同时处理本地路径和 URL 两种格式（为阶段二预留）。

### 1.4 `globals.css` — 添加样式

**文件：** `frontend/src/app/globals.css`

在 `.link-card` 相关样式（行 741-782）之后追加，参考现有 `link-card` 设计语言：
```css
.sources-panel {
  margin-top: 8px;
  border-top: 1px solid rgba(15, 23, 42, 0.08);
  padding-top: 8px;
}
.sources-label {
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--text-muted);
  margin: 0 0 6px 0;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.sources-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.source-item {
  display: flex;
  flex-direction: column;
  gap: 1px;
  padding: 4px 8px;
  border-radius: 6px;
  background: rgba(15, 23, 42, 0.03);
  border: 1px solid rgba(15, 23, 42, 0.06);
}
.source-filename {
  font-size: 0.82rem;
  font-weight: 500;
  color: rgba(15, 23, 42, 0.85);
}
.source-path {
  font-size: 0.72rem;
  color: var(--text-muted);
  word-break: break-all;
}
.source-link {
  font-size: 0.82rem;
  color: var(--accent);
  word-break: break-all;
  text-decoration: none;
}
.source-link:hover { text-decoration: underline; }
```

### 阶段一验证

1. `npm run build` — 确认 TypeScript 无类型错误
2. 启动前后端，发送涉及笔记检索的问题，确认回复底部出现来源文件区块
3. 检查 localStorage 中旧会话加载后不报错（`sources` 字段为 `undefined` 时区块不渲染）

---

## 阶段二：后端工具来源收集补全

**范围：仅后端改动，文件 `backend/app/services/chat.py`。**

### 2.1 `web_search` 来源收集

**文件：** `backend/app/services/chat.py`，行 1618-1635

在 `messages.append(...)` 之前插入 URL 提取逻辑：
```python
# 收集 web_search 结果中的 URL（防御性处理两种返回格式）
_web_results = result if isinstance(result, list) else (result.get("results") or [] if isinstance(result, dict) else [])
for _item in _web_results:
    _url = _item.get("url") if isinstance(_item, dict) else None
    if isinstance(_url, str) and _url:
        used_sources.append(_url)
```
> URL 直接追加，不经过 `_normalize_source_path`（该函数为本地路径设计）。
> 前端 `page.tsx` 的 URL 判断逻辑（`startsWith('http')`）已在阶段一预留。

**风险：** 搜索 API 返回的 URL 字段名可能不是 `"url"`。实施前先通过 DEBUG 日志确认实际字段名（在 `chat.py` 行 1586 附近添加临时 `logger.debug` 打印 `result`）。

### 2.2 `read_page` 来源收集

**文件：** `backend/app/services/chat.py`，行 1693-1702

在 `_log_tool_summary` 调用之后，`if event_callback:` 之前插入：
```python
# 收集 read_page 成功读取的 URL
if summary_status == "ok" and isinstance(url, str) and url:
    used_sources.append(url)
```
> `url` 变量在行 1661 已从 `arguments.get("url")` 赋值，此处可直接使用。

### 阶段二验证

1. 发送需要联网检索的问题，确认来源区块中出现 URL 条目且可点击
2. 在 `tests/test_chat_stream_events.py` 中添加测试：mock `web_search` 返回带 `url` 字段的结果，验证 `sources` 事件包含对应 URL

---

## 阶段三：来源粒度提升（可选增强）

**范围：后端 + 前端，改动较大，建议独立排期。**

### 目标

将 `heading_path`（章节路径，如 `BFS求二叉树最大层和 > 🧑‍💻 User`）作为结构化字段传递，前端展示"文件 > 章节"二级定位。

### 3.1 后端：`sources` 事件增量扩展（兼容模式）

**文件：** `backend/app/services/chat.py`

采用兼容模式（不破坏现有 `sources: string[]`），在 `sources` 事件中额外增加 `source_refs` 字段。

**步骤一：** 在 `used_sources` 旁边新增 `used_source_refs`（行 1260 附近）：
```python
used_sources: List[str] = []
used_source_refs: List[dict] = []  # 新增：{path: str, heading: str}
```

**步骤二：** 在 `query_my_notes` 收集处（行 1548-1552）同时填充 `used_source_refs`：
```python
for item in result.get("results") or []:
    source_path = item.get("source_path")
    heading = item.get("heading_path", "")
    if isinstance(source_path, str) and source_path:
        normalized = _normalize_source_path(source_path)
        used_sources.append(normalized)
        used_source_refs.append({"path": normalized, "heading": heading or ""})
```

**步骤三：** `read_note_file` 收集处（行 1787-1790）`heading` 设为空字符串：
```python
if isinstance(source_file, str) and source_file:
    normalized = _normalize_source_path(source_file)
    used_sources.append(normalized)
    read_files.append(normalized)
    used_source_refs.append({"path": normalized, "heading": ""})
```

**步骤四：** 修改 `sources` 事件（行 1422-1434），增量添加 `source_refs` 字段：
```python
seen_refs = set()
deduped_refs = []
for ref in used_source_refs:
    key = (ref["path"], ref["heading"])
    if key not in seen_refs:
        seen_refs.add(key)
        deduped_refs.append(ref)

event_callback({
    "type": "sources",
    "question_id": question_id,
    "sources": sorted(set(used_sources)),   # 保持兼容
    "source_refs": deduped_refs,            # 新增
    "expected_sources": expected_sources,
    "ts": time.time(),
})
```

### 3.2 前端：消费 `source_refs`

**文件：** `frontend/src/app/lib/chat-types.ts`

新增 `SourceRef` 接口，扩展相关类型：
```typescript
export interface SourceRef {
  path: string;
  heading: string;
}
// StreamEvent sources 变体中追加：
source_refs?: SourceRef[];
// ChatMessage 中追加：
sourceRefs?: SourceRef[];
```

**文件：** `frontend/src/app/hooks/useChatSessions.ts`

`sources` 事件处理中同时保存 `sourceRefs`：
```typescript
updateAssistantMessage(..., (prev) => ({
  ...prev,
  sources: event.sources,
  sourceRefs: event.source_refs,
}));
```

**文件：** `frontend/src/app/page.tsx`

优先使用 `sourceRefs` 渲染，降级到 `sources`：
```tsx
{(message.sourceRefs ?? message.sources?.map(p => ({ path: p, heading: '' })) ?? []).map(({ path, heading }) => {
  const fileName = path.split('/').pop() ?? path;
  return (
    <li key={`${path}::${heading}`} className="source-item">
      <span className="source-filename">{fileName}</span>
      {heading && <span className="source-heading">{heading}</span>}
      <span className="source-path">{path}</span>
    </li>
  );
})}
```

在 `globals.css` 中追加：
```css
.source-heading {
  font-size: 0.75rem;
  color: var(--accent);
  font-style: italic;
}
```

### 阶段三验证

1. 发送需要检索多篇笔记的问题，确认来源区块展示"文件名 > 章节"格式
2. 验证同一文件多个章节被引用时各自独立显示
3. 验证 `source_refs` 不存在时降级为只展示路径（`message.sources` 回退逻辑）
4. 确认 eval 脚本（`run_eval_stream.py`、`grade_by_llm.py`）解析 `sources` 事件时不受 `source_refs` 新字段影响

---

## 关键文件索引

| 文件 | 改动阶段 | 关键行号 |
|------|---------|---------|
| `frontend/src/app/lib/chat-types.ts` | 一、三 | 3-12（ChatMessage）、14-34（StreamEvent） |
| `frontend/src/app/hooks/useChatSessions.ts` | 一、三 | 483-528（parseNdjsonStream 回调） |
| `frontend/src/app/page.tsx` | 一、三 | 375-487（消息渲染循环） |
| `frontend/src/app/globals.css` | 一、三 | 741-782（link-card 样式后追加） |
| `backend/app/services/chat.py` | 二、三 | 1260（变量初始化）、1422-1434（sources 事件）、1548-1552（query 收集）、1618-1635（web_search）、1693-1702（read_page）、1787-1790（read_file 收集） |

## 实施顺序

```
阶段一（纯前端，无风险）
  1.1 chat-types.ts    ← 类型基础，优先完成
  1.2 useChatSessions  ← 依赖 1.1
  1.3 page.tsx         ← 依赖 1.1，可与 1.4 并行
  1.4 globals.css      ← 独立

阶段二（纯后端，需确认 API schema）
  2.1 web_search URL   ← 先确认字段名再改
  2.2 read_page URL    ← 独立

阶段三（可选，独立排期）
  后端 source_refs     ← 依赖阶段一完成
  前端 sourceRefs      ← 依赖后端改动
```

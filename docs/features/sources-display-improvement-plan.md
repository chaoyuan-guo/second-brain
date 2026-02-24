# 来源文件展示优化计划

## Context

用户测试后发现来源文件区块存在三个问题：
1. 同一文件被多个 chunk 命中时，重复显示多行（视觉冗余）
2. 本地文件路径不可交互，用户想查看原文却无法操作
3. AI 回答与来源的关联关系不明确（暂不处理）

本计划解决问题 1（按文件分组折叠展示）和问题 2（点击展开原文预览）。

---

## 改动概览

| 文件 | 改动类型 |
|------|---------|
| `backend/app/api/routes.py` | 新增 `GET /notes/content` 端点 |
| `frontend/src/app/lib/chat-types.ts` | 新增 `NOTE_CONTENT_ENDPOINT` 常量 |
| `frontend/src/app/page.tsx` | 重写 sources-panel 渲染逻辑 + 预览面板 |
| `frontend/src/app/globals.css` | 新增分组折叠 + 预览面板样式 |

---

## 一、后端：新增文件内容读取端点

**文件：** `backend/app/api/routes.py`

在 `/notes/upload` 路由之后新增：

```python
@router.get("/notes/content")
async def get_note_content(path: str) -> dict[str, object]:
    """读取指定笔记文件的内容，供前端预览使用。"""
    from ..services.tools import read_note_file
    from ..services.exceptions import ToolExecutionError
    try:
        result = await asyncio.to_thread(read_note_file, path)
    except ToolExecutionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="读取文件失败") from exc
    return {"content": result.get("content", ""), "source_file": result.get("source_file", "")}
```

- `path` 参数为 query string，例如 `?path=data/notes/my_markdowns/foo.md`
- 内部复用 `tools.read_note_file(path)` 函数，该函数已有路径安全检查（限定在 `data/notes/my_markdowns/` 目录内）
- 返回 `{content: string, source_file: string}`

---

## 二、前端类型常量

**文件：** `frontend/src/app/lib/chat-types.ts`

在 `UPLOAD_ENDPOINT` 常量附近新增：
```typescript
export const NOTE_CONTENT_ENDPOINT = '/notes/content';
```

---

## 三、前端 page.tsx：重写 sources-panel

**文件：** `frontend/src/app/page.tsx`

### 3.1 新增 state

在 `HomePage` 组件内，现有 state 声明区域末尾新增：
```typescript
const [previewContent, setPreviewContent] = useState<string | null>(null);
const [previewTitle, setPreviewTitle] = useState<string>('');
const [previewLoading, setPreviewLoading] = useState(false);
const [expandedGroups, setExpandedGroups] = useState<Record<string, boolean>>({});
```

### 3.2 新增 handleOpenPreview 函数

```typescript
const handleOpenPreview = useCallback(async (filePath: string, title: string) => {
  setPreviewTitle(title);
  setPreviewContent('');   // 空字符串触发面板显示，同时进入 loading
  setPreviewLoading(true);
  try {
    const res = await fetch(`${getApiBaseUrl()}${NOTE_CONTENT_ENDPOINT}?path=${encodeURIComponent(filePath)}`);
    if (!res.ok) throw new Error(`读取失败: ${res.status}`);
    const data = await res.json() as { content: string };
    setPreviewContent(data.content);
  } catch (err) {
    setPreviewContent(`读取文件失败：${err instanceof Error ? err.message : '未知错误'}`);
  } finally {
    setPreviewLoading(false);
  }
}, []);
```

### 3.3 新增 toggleGroup 函数

```typescript
const toggleGroup = useCallback((key: string) => {
  setExpandedGroups((prev) => ({ ...prev, [key]: !prev[key] }));
}, []);
```

### 3.4 重写 sources-panel 渲染逻辑

将现有 sources-panel 块（第 467-509 行）替换为按文件分组的版本：

- 用 `Map<string, {path, headings: string[]}>` 对 `sourceRefs` 按 `path` 分组
- 同一 heading 去重（`!existing.headings.includes(heading)`）
- 每个文件一个 `<li className="source-group">`
- 文件名渲染为可点击的 `<button>` 触发预览
- 若有 headings，右侧显示"N 个章节"折叠按钮（`ChevronRightIcon`/`ChevronLeftIcon`）
- 展开后在文件名下方显示 heading 列表
- URL 类型来源保持原有 `<a>` 链接样式

### 3.5 新增预览面板 JSX

在 `</main>` 之前添加预览 overlay：

- `previewContent !== null` 时渲染（空字符串也触发）
- 点击 overlay 背景关闭，点击面板内部不关闭（`stopPropagation`）
- 面板内：标题栏（文件名 + × 关闭按钮）+ 内容区（loading 时显示 ThinkingDots，完成后 `<pre>` 显示原文）
- 使用 `CloseIcon`（已有 import）

### 3.6 新增 import

- `NOTE_CONTENT_ENDPOINT` 从 `./lib/chat-types`（已有 `getApiBaseUrl`、`UPLOAD_ENDPOINT` 等同路径 import，在同一行追加）

---

## 四、前端 globals.css：新增样式

在 `.source-heading` 样式（第 845 行）之后追加：

```css
/* 来源文件分组 */
.source-group {
  display: flex;
  flex-direction: column;
  border-radius: 6px;
  background: rgba(15, 23, 42, 0.03);
  border: 1px solid rgba(15, 23, 42, 0.06);
  overflow: hidden;
}

.source-group-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 4px 8px;
  gap: 8px;
}

.source-filename-btn {
  font-size: 0.82rem;
  font-weight: 500;
  color: var(--accent);
  background: none;
  border: none;
  padding: 0;
  cursor: pointer;
  text-align: left;
  text-decoration: underline;
  text-underline-offset: 2px;
  text-decoration-color: transparent;
  transition: text-decoration-color 0.15s;
}
.source-filename-btn:hover {
  text-decoration-color: var(--accent);
}

.source-expand-btn {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  font-size: 0.72rem;
  color: var(--text-muted);
  background: none;
  border: none;
  padding: 2px 4px;
  cursor: pointer;
  border-radius: 4px;
  white-space: nowrap;
  flex-shrink: 0;
}
.source-expand-btn:hover {
  background: var(--surface-strong);
  color: var(--text-strong);
}
.source-expand-btn svg {
  width: 12px;
  height: 12px;
}

.source-headings-list {
  list-style: none;
  padding: 0 8px 6px 16px;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
  border-top: 1px solid rgba(15, 23, 42, 0.06);
}

.source-heading-item {
  padding: 1px 0;
}

/* 文件预览面板 */
.preview-overlay {
  position: fixed;
  inset: 0;
  background: rgba(15, 23, 42, 0.4);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: 24px;
}

.preview-panel {
  background: var(--surface-overlay);
  border-radius: var(--radius-lg);
  box-shadow: 0 8px 32px rgba(15, 23, 42, 0.18);
  width: 100%;
  max-width: 760px;
  max-height: 80vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.preview-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 18px;
  border-bottom: 1px solid var(--panel-border);
  flex-shrink: 0;
}

.preview-title {
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--text-strong);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.preview-close {
  background: none;
  border: none;
  padding: 4px;
  cursor: pointer;
  color: var(--text-muted);
  border-radius: 6px;
  display: flex;
  align-items: center;
  flex-shrink: 0;
}
.preview-close:hover {
  background: var(--surface-strong);
  color: var(--text-strong);
}

.preview-body {
  flex: 1;
  overflow-y: auto;
  padding: 16px 18px;
}

.preview-loading {
  display: flex;
  justify-content: center;
  padding: 24px;
}

.preview-content {
  font-family: 'SF Mono', 'Fira Code', Consolas, monospace;
  font-size: 0.82rem;
  line-height: 1.6;
  color: var(--text-strong);
  white-space: pre-wrap;
  word-break: break-word;
  margin: 0;
}
```

---

## 验证方法

1. `cd frontend && npm run build` — 确认 TypeScript 无类型错误
2. 启动前后端，发送涉及笔记检索的问题
3. 确认来源区块：同一文件多个章节折叠为一行，显示"N 个章节"按钮
4. 点击章节数按钮，展开/收起章节列表
5. 点击文件名，弹出预览面板，显示 loading 后显示文件内容
6. 点击面板外部或 × 按钮，关闭预览
7. 直接 curl `GET /notes/content?path=data/notes/my_markdowns/xxx.md` 验证端点
8. 传入不存在路径时返回 404

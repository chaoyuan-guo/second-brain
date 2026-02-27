# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指导。

## 项目概述

Second Brain 是一个基于本地 Markdown 的智能检索系统。它索引个人笔记（主要是算法练习和 LeetCode 提交记录），并提供一个聊天界面，能够带引用地回答来自源文档的问题。

## 架构

```
┌──────────────────────────── Docker Container ────────────────────────────┐
│                                                                            │
│  Frontend (Next.js)  :9080                                                 │
│  - 浏览器统一访问前端                                                       │
│  - 通过 Next.js API Proxy 转发到内部服务                                   │
│                                                                            │
│  OpenCode Server (Node.js, 无头模式) :9090                                 │
│  - Agentic Loop / session API / 全局 SSE 事件流                            │
│                                                                            │
│  RAG & Data Server (FastAPI + FastMCP) :9070                               │
│  - MCP SSE `/sse`（供 OpenCode 调用）                                      │
│  - REST `/notes/content`、`/notes/upload`（供前端代理调用）                 │
│                                                                            │
│  OpenCode  ── MCP SSE ──▶  RAG & Data                                      │
│  Frontend ── API Proxy ──▶  OpenCode / RAG                                 │
└────────────────────────────────────────────────────────────────────────────┘
```

**核心数据流：**
- 浏览器请求仅进入前端 `:9080`，前端再通过 `/api/*` 代理到内部 `:9090/:9070`
- 对话通过 OpenCode session API（`/session`、`/session/:id/prompt_async`）与全局事件流 `/event` 完成
- source refs 从 `ToolPart.state.completed.metadata.source_refs` 透传给前端来源面板
- 笔记检索与上传建索引由同一 RAG & Data Server 进程处理，避免跨进程状态不一致

## 开发命令

### 服务管理（通过 start_services.sh）

```bash
# 启动所有服务（后端 + 前端 + MCP 解释器）
./start_services.sh start all

# 启动单个服务
./start_services.sh start backend    # uvicorn 运行在端口 9000
./start_services.sh start frontend   # next dev 运行在端口 9080
./start_services.sh start mcp        # MCP 解释器运行在端口 9070

# 停止/重启/状态
./start_services.sh stop backend
./start_services.sh restart all
./start_services.sh status
```

健康检查成功时输出 `服务健康检查：OK`。

### 手动启动后端

```bash
./.venv/bin/python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 9000
```

### 前端

```bash
cd frontend
npm install
npm run dev      # 开发服务器运行在端口 9080
npm run build    # 生产构建输出到 frontend/out/
```

### 测试

```bash
# 运行所有测试
./.venv/bin/python -m pytest -q

# 运行单个测试文件
./.venv/bin/python -m pytest tests/test_chat_stream_events.py -v

# 运行特定测试
./.venv/bin/python -m pytest tests/test_chat_stream_events.py::test_stream_events_basic -v
```

### 评估

```bash
# 针对测试集运行评估（需要后端运行中，LLM-as-Judge 评分）
./.venv/bin/python -u eval/scripts/run_eval_stream.py \
  --base-url http://127.0.0.1:9000 \
  --concurrency 10 \
  --report eval/reports/report.json \
  2>&1 | tee eval/reports/eval.log

# 只运行指定题目（按 ID 前缀匹配，逗号分隔）
./.venv/bin/python -u eval/scripts/run_eval_stream.py \
  --question-ids Q01,Q14 \
  --base-url http://127.0.0.1:9000 \
  --report eval/reports/report.json

# 仅评分已有答案
./.venv/bin/python eval/scripts/grade_by_llm.py \
  --answers eval/reports/answers.json \
  --output eval/reports/report.json
```

### Docker

```bash
docker build -t second_brain:opencode -f docker/Dockerfile.opencode .
docker rm -f second_brain_opencode 2>/dev/null || true

# 推荐：仅暴露前端端口（浏览器访问）
docker run -d --name second_brain_opencode \
  -p 9080:9080 \
  --env-file .env.docker \
  -v "$PWD/data:/app/data" \
  second_brain:opencode

# 调试：需要直连 OpenCode/RAG 时再额外暴露
# docker run -d --name second_brain_opencode \
#   -p 9080:9080 -p 9090:9090 -p 9070:9070 \
#   --env-file .env.docker \
#   -v "$PWD/data:/app/data" \
#   second_brain:opencode

# 健康检查
curl -I "http://127.0.0.1:9080"
```

## 关键目录

| 路径 | 用途 |
|------|------|
| `backend/app/services/chat.py` | 核心聊天编排，包含工具调用循环 |
| `backend/app/services/tools.py` | 工具实现（查询、读取、代码解释器） |
| `backend/app/core/config.py` | 配置、系统提示词、常量 |
| `data/notes/my_markdowns/` | 用于索引的源 Markdown 文件 |
| `data/indexes/` | FAISS 索引文件 |
| `skills/` | 技能定义（SKILL.md 文件），用于专门查询 |
| `eval/` | 评估脚本和测试集 |

## 环境变量

必需：
- `SUPER_MIND_API_KEY` 或 `AI_BUILDER_TOKEN` - LLM 后端的 API 密钥（用于 embedding 模型）

可选：
- `CHAT_ALLOWED_ORIGINS` - CORS 来源（默认：localhost:9080）
- `MCP_SSE_ENDPOINT` - MCP 解释器端点（默认：http://127.0.0.1:9070/sse/）
- `MCP_INTERPRETER_BACKEND` - 强制使用 `embedded` 或 `mcp` 解释器模式
- `OPENCODE_SELF_CHECK` - 容器启动后是否执行 OpenCode 自检（默认 `1`，设为 `0` 可跳过）
- `.env.docker` - 容器推荐环境文件，统一维护 ai-builder/Azure 变量

### Chat 模型端点配置

系统根据运行环境自动选择 chat 模型端点：

**本地开发环境**（默认使用 Azure）：
- 需要配置以下 Azure 相关环境变量：
  - `azure_base_url` - Azure OpenAI 端点（例如：https://td06.openai.azure.com/openai/v1）
  - `azure_api_key` - Azure API 密钥
  - `azure_api-version` - Azure API 版本（例如：2025-04-01-preview）
  - `azure_use_model` - Azure 部署的模型名（例如：gpt-52）
- 如需强制使用 ai-builder 端点，设置 `use_azure=False`

**容器服务环境**（默认使用 ai-builder）：
- 使用 `SUPER_MIND_API_BASE_URL`（默认：https://space.ai-builders.com/backend/v1，必须带 `/v1`）
- 使用 `SUPER_MIND_CHAT_MODEL`（默认：gpt-5）
- 如需强制使用 Azure 端点，设置 `use_azure=True` 并配置相应的 Azure 环境变量

**Embedding 模型**：
- 所有环境下都使用 ai-builder 端点（https://space.ai-builders.com/backend/v1）
- 通过 `SUPER_MIND_API_KEY` 或 `AI_BUILDER_TOKEN` 认证

**评估脚本**（eval/scripts/grade_by_llm.py）：
- 默认使用 Azure 端点的 gpt-52 模型进行 LLM-as-Judge 评分
- 可通过环境变量覆盖：`azure_base_url`、`azure_api_key`、`azure_use_model`

## 编码规范

- 后端：PEP 8、类型注解、4 空格缩进、函数使用 snake_case
- 前端：TypeScript、Next.js App Router、组件使用 PascalCase、hooks 使用 camelCase
- 提交：Conventional Commits 格式（`feat:`、`fix:`、`chore:`）
- 语言：文档和讨论使用中文，代码和命令使用英文

## 设计理念

本项目遵循"少结构，多智能"原则——基于模型智能构建能力，而非过度工程化的约束。在提出解决方案时，优先考虑通用性和可复用性。当需求模糊或存在更好的替代方案时，请提出澄清问题。

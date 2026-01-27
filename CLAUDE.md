# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指导。

## 项目概述

Second Brain 是一个基于本地 Markdown 的智能检索系统。它索引个人笔记（主要是算法练习和 LeetCode 提交记录），并提供一个聊天界面，能够带引用地回答来自源文档的问题。

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (Next.js)                       │
│                    localhost:9080 / frontend/                    │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend (FastAPI)                           │
│                   localhost:9000 / backend/app/                  │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────────────┐ │
│  │ Routes   │→ │ Chat Service │→ │ Tools (query, read, code)  │ │
│  │ /chat/*  │  │ (chat.py)    │  │ (tools.py)                 │ │
│  └──────────┘  └──────────────┘  └────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
┌─────────────────────────────────┐   ┌─────────────────────────────┐
│   FAISS Index + Metadata        │   │    MCP Python Interpreter   │
│   data/indexes/                 │   │    localhost:9070 (SSE)     │
│   data/notes/                   │   │    or embedded interpreter  │
└─────────────────────────────────┘   └─────────────────────────────┘
```

**核心数据流：**
- 聊天请求访问 `/chat/stream`（NDJSON 流式）或 `/chat`（阻塞式）
- 后端使用兼容 OpenAI 的 API 进行工具调用（query_my_notes、read_note_file、run_code_interpreter、load_skill）
- 笔记通过 FAISS 进行语义搜索索引；元数据存储在 `data/notes/my_notes_metadata.json`
- 代码执行使用 MCP SSE 服务器（本地开发）或内嵌解释器（容器环境）

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
source .venv/bin/activate
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 9000
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
pytest -q

# 运行单个测试文件
pytest tests/test_chat_stream_events.py -v

# 运行特定测试
pytest tests/test_chat_stream_events.py::test_stream_events_basic -v
```

### 评估

```bash
# 针对测试集运行评估（需要后端运行中）
python eval/scripts/run_eval_stream.py \
  --testset eval/testsets/testset.json \
  --base-url http://127.0.0.1:9000 \
  --strict-sources \
  --recall-k 5 \
  --concurrency 5 \
  --report eval/reports/report.json

# 评分答案
python eval/scripts/grade_testset.py \
  --testset eval/testsets/testset.json \
  --answers eval/reports/answers.json \
  --output eval/reports/report.json
```

### Docker

```bash
docker build -t second_brain:local .
docker run -d --name second_brain_18000 -p 18000:8000 -e AI_BUILDER_TOKEN second_brain:local
curl "http://127.0.0.1:18000/hello?input=test"
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
- `SUPER_MIND_API_KEY` 或 `AI_BUILDER_TOKEN` - LLM 后端的 API 密钥

可选：
- `CHAT_ALLOWED_ORIGINS` - CORS 来源（默认：localhost:9080）
- `MCP_SSE_ENDPOINT` - MCP 解释器端点（默认：http://127.0.0.1:9070/sse/）
- `MCP_INTERPRETER_BACKEND` - 强制使用 `embedded` 或 `mcp` 解释器模式

## 编码规范

- 后端：PEP 8、类型注解、4 空格缩进、函数使用 snake_case
- 前端：TypeScript、Next.js App Router、组件使用 PascalCase、hooks 使用 camelCase
- 提交：Conventional Commits 格式（`feat:`、`fix:`、`chore:`）
- 语言：文档和讨论使用中文，代码和命令使用英文

## 设计理念

本项目遵循"少结构，多智能"原则——基于模型智能构建能力，而非过度工程化的约束。在提出解决方案时，优先考虑通用性和可复用性。当需求模糊或存在更好的替代方案时，请提出澄清问题。

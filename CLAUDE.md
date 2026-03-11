# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指导。

## 项目概述

Second Brain 是一个基于本地 Markdown 的智能检索系统。它索引个人笔记（主要是算法练习和 LeetCode 提交记录），并提供一个聊天界面，能够带引用地回答来自源文档的问题。

## 项目结构与模块组织

- 后端代码位于 `backend/app`，`backend/app/main.py` 通过 FastAPI + OpenAI 工具链暴露 API，运行日志统一写入 `runtime/logs/backend.log`，而自定义笔记资源集中于 `data/notes/my_markdowns/` 方便复用。
- Next.js 前端置于 `frontend/src/app`，生产运行由容器内 `next start` 提供，运行入口统一为 OpenCode 一体化 Docker 容器。

## 开发命令

### Docker（OpenCode 一体化容器）

```bash
docker build -t second_brain:opencode -f docker/Dockerfile.opencode .
docker rm -f second_brain_opencode 2>/dev/null || true

# 推荐：仅暴露前端端口（浏览器访问）
docker run -d --name second_brain_opencode --restart unless-stopped \
  -p 9080:9080 \
  --env-file .env.docker \
  -v "$PWD/data:/app/data" \
  second_brain:opencode

# 调试：需要直连 OpenCode/RAG 时再额外暴露
# docker run -d --name second_brain_opencode --restart unless-stopped \
#   -p 9080:9080 -p 9090:9090 -p 9070:9070 \
#   --env-file .env.docker \
#   -v "$PWD/data:/app/data" \
#   second_brain:opencode

# 健康检查
curl -I "http://127.0.0.1:9080"

# 查看日志
docker logs -f second_brain_opencode

# 停止容器
docker stop second_brain_opencode
```

## 日志规范

- 容器部署仅输出 stdout（供 `docker logs`），后端/工具不写 `runtime/logs/*.log`；如需在容器运行前端，同样保持 stdout 输出。
- 后端/工具输出可用 `LOG_TO_STDOUT` 与 `LOG_TO_FILE` 覆盖默认行为；如显式启用双写，请确保 stdout 不再重定向回同一文件，避免重复。

## 编码规范

- **后端**：遵循 PEP 8 与类型注解优先原则，保持 4 空格缩进、短小协程、集中式异常 `ToolExecutionError`；配置项放在 `BASE_DIR` 旁的常量里，新增工具函数时请以 `snake_case` 命名并补充 docstring。
- **前端**：使用 TypeScript + Next.js App Router，组件命名采用 `PascalCase`，hooks/工具使用 `camelCase`；尽量将 UI 状态封装为客户端组件，网络请求统一调用 `/api` 代理或现有 FastAPI 端点。
- **提交**：遵循 Conventional Commits 格式（如 `feat: add note upload validation`、`fix: guard empty filename`），保持可读性；单次提交聚焦单一功能或缺陷修复。
- **语言**：文档和讨论使用中文，代码和命令使用英文。

## 测试指南

- 后端测试已位于根目录 `tests/`（`test_<module>.py`），默认执行命令为 `pytest -q`；新增后端测试时请覆盖成功与失败分支，目标覆盖率 ≥80%。
- 前端测试位于 `frontend/src/__tests__/`，采用 React Testing Library + Vitest；执行 `npm run test`（`vitest run`）并通过 `npm run lint`（`tsc --noEmit`）保证 TS/JSX 规范，命名遵循 `<Component>.test.tsx`。

## 评估指南

评估采用 LLM-as-Judge 方法，使用 Azure 端点的 `gpt-52` 模型，围绕个性化命中、精准简洁、诚实性、可追溯性四个维度评分：

```bash
# 针对测试集运行评估（需要后端运行中，LLM-as-Judge 评分）
./.venv/bin/python -u eval/scripts/run_eval_stream.py \
  --base-url http://127.0.0.1:9090 \
  --concurrency 10 \
  --report eval/reports/report.json \
  2>&1 | tee eval/reports/eval.log

# 只运行指定题目（按 ID 前缀匹配，逗号分隔）
./.venv/bin/python -u eval/scripts/run_eval_stream.py \
  --question-ids Q01,Q14 \
  --base-url http://127.0.0.1:9090 \
  --report eval/reports/report.json

# 仅评分已有答案
./.venv/bin/python eval/scripts/grade_by_llm.py \
  --answers eval/reports/answers.json \
  --output eval/reports/report.json
```

## 提交与 Pull Request 规范

- 仓库已初始化 Git，请继续遵循 Conventional Commits 保持可读性；单次提交聚焦单一功能或缺陷修复。
- PR 需包含变更摘要、验证方式（命令输出或截图）、相关 Issue 链接以及潜在风险；若触及容器启动参数或环境变量，请同步更新 Docker 启动说明以免部署偏差。

## 安全与配置提示

- `.env` 必须提供 `SUPER_MIND_API_KEY` 与可选 `CHAT_ALLOWED_ORIGINS`；不要将密钥写入日志或前端 bundle，可通过 `os.getenv` 访问并在启动时校验。
- 生产部署建议优先使用容器镜像发布，默认仅暴露 9080 端口；外部请求保持 20s 超时并保留错误日志，避免工具链卡死。

## 模型端点使用规则

- **Chat 模型**：系统根据运行环境自动选择端点（通过 `running_in_container()` 判断）：
  - **非容器环境**：默认使用 Azure 端点（`azure_base_url`、`azure_api_key`、`azure_api-version`、`azure_use_model`），可通过 `use_azure=False` 覆盖。
  - **容器服务**：默认使用 ai-builder 端点（`SUPER_MIND_API_BASE_URL`、`SUPER_MIND_CHAT_MODEL`），可通过 `use_azure=True` 覆盖。
- **Embedding 模型**：所有环境统一使用 ai-builder 端点（https://space.ai-builders.com/backend/v1），通过 `SUPER_MIND_API_KEY` 或 `AI_BUILDER_TOKEN` 认证。
- **评估评分**（`eval/scripts/grade_by_llm.py`）：默认使用 Azure 端点的 `gpt-52` 模型进行 LLM-as-Judge 评分，可通过 `azure_base_url`、`azure_api_key`、`azure_use_model` 环境变量覆盖。

## 关键目录

| 路径 | 用途 |
|------|------|
| `backend/app/services/tools.py` | 工具实现（检索、读取、query rewrite、技能加载） |
| `backend/app/core/config.py` | 配置、系统提示词、常量 |
| `data/notes/my_markdowns/` | 用于索引的源 Markdown 文件 |
| `data/indexes/` | FAISS 索引文件 |
| `eval/` | 评估脚本和测试集 |
| `runtime/logs/` | 运行时日志文件 |

## 环境变量

必需：
- `SUPER_MIND_API_KEY` 或 `AI_BUILDER_TOKEN` - LLM 后端的 API 密钥（用于 embedding 模型）

可选：
- `CHAT_ALLOWED_ORIGINS` - CORS 来源（默认：localhost:9080）
- `OPENCODE_SELF_CHECK` - 容器启动后是否执行 OpenCode 自检（默认 `1`，设为 `0` 可跳过）
- `.env.docker` - 容器推荐环境文件，统一维护 ai-builder/Azure 变量

## 协作与设计原则

- 讨论技术实现方案时，请优先从更通用、可复用、泛化能力更强的角度思考与给出建议。
- Agentic 系统倾向遵循 "less structure, more intelligence" 的设计哲学：尽量让系统能力建立在模型能力之上，避免过度结构化/过度工程化的约束，从而在模型变强时让系统能力能够同步“水涨船高”。
- 当我下达任务或提出请求时，如果你认为信息不足、表述不清晰、存在隐含前提，或你有更优方案/替代路径，请主动提问、澄清并阐述你的思考与权衡。

## 其他说明

默认使用中文进行说明与讨论，除非内容为代码片段、命令或规范要求英文表述。

## 运行规范补充

- 服务运行统一通过 Docker 容器管理：修改后请按需重建镜像并重启容器。
- 推荐重启流程：`docker rm -f second_brain_opencode 2>/dev/null || true && docker build -t second_brain:opencode -f docker/Dockerfile.opencode . && docker run -d --name second_brain_opencode --restart unless-stopped -p 9080:9080 --env-file .env.docker -v "$PWD/data:/app/data" second_brain:opencode`
- 调试场景需要直连 OpenCode/RAG 时，使用调试端口映射 `-p 9080:9080 -p 9090:9090 -p 9070:9070`。

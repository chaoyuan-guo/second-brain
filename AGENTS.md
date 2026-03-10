# Repository Guidelines

## Project Structure & Module Organization
- 后端代码位于 `backend/app`，`backend/app/main.py` 通过 FastAPI + OpenAI 工具链暴露 API，运行日志统一写入 `runtime/logs/backend.log`，而自定义笔记资源集中于 `data/notes/my_markdowns/` 方便复用。 Backend logic lives under `backend/app`, logs to `runtime/logs/backend.log`, while knowledge snippets live in `data/notes/my_markdowns/` for reuse.
- Next.js 前端置于 `frontend/src/app`，静态导出产物写入 `frontend/out`，运行入口统一为 OpenCode 一体化 Docker 容器。 The UI resides under `frontend/src/app`, static export lands in `frontend/out`, and services are started via the integrated Docker container.

## Build, Test, and Development Commands
- Docker（OpenCode 一体化容器）：
  - 构建镜像：`docker build -t second_brain:opencode -f docker/Dockerfile.opencode .`
  - 推荐环境变量文件：`.env.docker`（示例：`SUPER_MIND_API_BASE_URL=https://space.ai-builders.com/backend/v1`，必须带 `/v1`）
  - 启动前校验：确保 `.env.docker` 中 `SUPER_MIND_API_KEY`（或 `AI_BUILDER_TOKEN`）非空，避免后端因缺少 token 启动失败。
  - 默认启动（仅暴露前端端口）：`docker rm -f second_brain_opencode 2>/dev/null || true && docker run -d --name second_brain_opencode --restart unless-stopped -p 9080:9080 --env-file .env.docker -v "$PWD/data:/app/data" second_brain:opencode`
  - 调试启动（额外暴露 OpenCode/RAG）：`docker rm -f second_brain_opencode 2>/dev/null || true && docker run -d --name second_brain_opencode --restart unless-stopped -p 9080:9080 -p 9090:9090 -p 9070:9070 --env-file .env.docker -v "$PWD/data:/app/data" second_brain:opencode`
  - 验证：`curl -I "http://127.0.0.1:9080"`
  - 日志/停止：`docker logs -f second_brain_opencode` / `docker stop second_brain_opencode`

## Logging Guidelines
- 容器部署仅输出 stdout（供 `docker logs`），后端/工具不写 `runtime/logs/*.log`；如需在容器运行前端，同样保持 stdout 输出。
- 后端/工具输出可用 `LOG_TO_STDOUT` 与 `LOG_TO_FILE` 覆盖默认行为；如显式启用双写，请确保 stdout 不再重定向回同一文件，避免重复。

## Coding Style & Naming Conventions
- 后端遵循 PEP 8 与类型注解优先原则，保持 4 空格缩进、短小协程、集中式异常 `ToolExecutionError`；配置项放在 `BASE_DIR` 旁的常量里，新增工具函数时请以 `snake_case` 命名并补充 docstring。 Backend code should remain type-hinted, 4-space indented, and keep tool helpers in snake_case with docstrings and cohesive logging keys.
- 前端使用 TypeScript + Next.js App Router，组件命名采用 `PascalCase`，hooks/工具使用 `camelCase`；尽量将 UI 状态封装为客户端组件，网络请求统一调用 `/api` 代理或现有 FastAPI 端点。 Keep styles colocated via CSS modules or inline Tailwind classes for consistency.

## Testing Guidelines
- 后端测试已位于根目录 `tests/`（`test_<module>.py`），默认执行命令为 `pytest -q`；新增后端测试时请覆盖成功与失败分支，目标覆盖率 ≥80%。 Add fixtures for external HTTP clients and patch network calls to keep tests deterministic.
- 前端测试位于 `frontend/src/__tests__/`，采用 React Testing Library + Vitest；执行 `npm run test`（`vitest run`）并通过 `npm run lint`（`tsc --noEmit`）保证 TS/JSX 规范，命名遵循 `<Component>.test.tsx`。 Snapshot tests should be paired with meaningful interaction assertions.

## Evaluation Guidelines
- 评估采用 LLM-as-Judge 方法，使用 Azure 端点的 `gpt-52` 模型，围绕个性化命中、精准简洁、诚实性、可追溯性四个维度评分：`./.venv/bin/python eval/scripts/run_eval_stream.py --base-url http://127.0.0.1:9090 --concurrency 10 --report eval/reports/report.json`（默认写 `eval/reports/answers.json`）。

## Commit & Pull Request Guidelines
- 仓库已初始化 Git，请继续遵循 Conventional Commits（如 `feat: add web_search retries`、`fix: guard empty query`）保持可读性；单次提交聚焦单一功能或缺陷修复。 Commits should stay atomic on the `main` branch unless stated otherwise.
- PR 需包含变更摘要、验证方式（命令输出或截图）、相关 Issue 链接以及潜在风险；若触及容器启动参数或环境变量，请同步更新 Docker 启动说明以免部署偏差。 Request reviewers familiar with both FastAPI and Next.js when changes cross the stack.

## Security & Configuration Tips
- `.env` 必须提供 `SUPER_MIND_API_KEY` 与可选 `CHAT_ALLOWED_ORIGINS`；不要将密钥写入日志或前端 bundle，可通过 `os.getenv` 访问并在启动时校验。 Keep the `.env` file out of version control.
- 生产部署建议优先使用容器镜像发布，默认仅暴露 9080 端口；外部请求保持 20s 超时并保留错误日志，避免工具链卡死。 Rotate API tokens regularly and scrub logs before sharing.

## 模型端点使用规则
- **Chat 模型**：系统根据运行环境自动选择端点（通过 `running_in_container()` 判断）：
  - **非容器环境**：默认使用 Azure 端点（`azure_base_url`、`azure_api_key`、`azure_api-version`、`azure_use_model`），可通过 `use_azure=False` 覆盖。
  - **容器服务**：默认使用 ai-builder 端点（`SUPER_MIND_API_BASE_URL`、`SUPER_MIND_CHAT_MODEL`），可通过 `use_azure=True` 覆盖。
- **Embedding 模型**：所有环境统一使用 ai-builder 端点（https://space.ai-builders.com/backend/v1），通过 `SUPER_MIND_API_KEY` 或 `AI_BUILDER_TOKEN` 认证。
- **评估评分**（`eval/scripts/grade_by_llm.py`）：默认使用 Azure 端点的 `gpt-52` 模型进行 LLM-as-Judge 评分，可通过 `azure_base_url`、`azure_api_key`、`azure_use_model` 环境变量覆盖。

## 协作与设计原则
- 讨论技术实现方案时，请优先从更通用、可复用、泛化能力更强的角度思考与给出建议。
- Agentic 系统倾向遵循 “less structure, more intelligence” 的设计哲学：尽量让系统能力建立在模型能力之上，避免过度结构化/过度工程化的约束，从而在模型变强时让系统能力能够同步“水涨船高”。
- **OpenCode 原生优先原则**：凡是涉及 OpenCode 一体化容器内的会话链路、事件流、工具调用、Agent 行为约束、回答收口策略等改造，必须优先基于 OpenCode 原生能力实现。
- OpenCode 相关改造的默认实施顺序应为：先调整 OpenCode agent prompt、原生工具使用策略、session/event 主链路与原生事件语义；前端展示层与自建后端代理层只负责最小必要的适配，不应默认承接主逻辑。
- 只有在 OpenCode 原生能力经过验证后仍无法满足需求时，才允许在应用层增加补偿逻辑；此时必须明确说明原生能力的限制、补偿逻辑的边界，以及为什么不能继续在 OpenCode 层解决。
- 未经用户明确确认，不得把前端或自建后端中的兜底/补偿逻辑升级为默认主链路，也不得用应用层重写去替代 OpenCode 原生链路。
- 当我下达任务或提出请求时，如果你认为信息不足、表述不清晰、存在隐含前提，或你有更优方案/替代路径，请主动提问、澄清并阐述你的思考与权衡。

## 其他说明
默认使用中文进行说明与讨论，除非内容为代码片段、命令或规范要求英文表述。

## 运行规范补充
- 服务运行统一通过 Docker 容器管理：修改后请按需重建镜像并重启容器。
- 推荐重启流程：`docker rm -f second_brain_opencode 2>/dev/null || true && docker build -t second_brain:opencode -f docker/Dockerfile.opencode . && docker run -d --name second_brain_opencode --restart unless-stopped -p 9080:9080 --env-file .env.docker -v "$PWD/data:/app/data" second_brain:opencode`
- 调试场景需要直连 OpenCode/RAG 时，使用调试端口映射 `-p 9080:9080 -p 9090:9090 -p 9070:9070`。

# Repository Guidelines

## Project Structure & Module Organization
- 后端代码位于 `backend/app`，`backend/app/main.py` 通过 FastAPI + OpenAI 工具链暴露 API，运行日志统一写入 `runtime/logs/backend.log`，而自定义笔记资源集中于 `data/notes/my_markdowns/` 方便复用。 Backend logic lives under `backend/app`, logs to `runtime/logs/backend.log`, while knowledge snippets live in `data/notes/my_markdowns/` for reuse.
- Next.js 前端置于 `frontend/src/app`，静态导出产物写入 `frontend/out`，`start_services.sh` 负责同时拉起 `uvicorn` 与 `npm run dev`，常态开发请从该脚本或分别在两个终端启动服务。 The UI resides under `frontend/src/app`, static export lands in `frontend/out`, and `start_services.sh` orchestrates both uvicorn and npm dev servers.

## Build, Test, and Development Commands
- Python 依赖需在虚拟环境中手动安装：`python -m venv .venv && source .venv/bin/activate && pip install fastapi uvicorn httpx beautifulsoup4 python-dotenv openai`，随后使用 `uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 9000` 作为本地后端入口。 Set up a venv, install the listed packages, then run `uvicorn backend.app.main:app` for hot reload.
- 前端工作流：`cd frontend && npm install && npm run dev` 提供本地调试，`npm run build` 生成产物供 `npm run start` 或静态部署使用。Use `start_services.sh` when you need both tiers plus synchronized log files.
- Docker 镜像构建与容器启动（不读取项目 `.env`、不挂载 `data/runtime`，token 走当前 shell 环境变量）：
  - 构建镜像：`docker build -t second_brain:local .`
  - 启动容器（宿主机 18000 → 容器 8000）：`docker rm -f second_brain_18000 2>/dev/null || true && docker run -d --name second_brain_18000 --restart unless-stopped -p 18000:8000 -e AI_BUILDER_TOKEN second_brain:local`
  - 验证：`curl "http://127.0.0.1:18000/hello?input=test"`
  - 日志/停止：`docker logs -f second_brain_18000` / `docker stop second_brain_18000`

## Logging Guidelines
- 本地开发仅写日志文件：后端 `runtime/logs/backend.log`、前端 `runtime/logs/frontend.log`、工具输出 `runtime/logs/tool_output.log`。
- 容器部署仅输出 stdout（供 `docker logs`），后端/工具不写 `runtime/logs/*.log`；如需在容器运行前端，同样保持 stdout 输出。
- 后端/工具输出可用 `LOG_TO_STDOUT` 与 `LOG_TO_FILE` 覆盖默认行为；如显式启用双写，请确保 stdout 不再重定向回同一文件，避免重复。

## Coding Style & Naming Conventions
- 后端遵循 PEP 8 与类型注解优先原则，保持 4 空格缩进、短小协程、集中式异常 `ToolExecutionError`；配置项放在 `BASE_DIR` 旁的常量里，新增工具函数时请以 `snake_case` 命名并补充 docstring。 Backend code should remain type-hinted, 4-space indented, and keep tool helpers in snake_case with docstrings and cohesive logging keys.
- 前端使用 TypeScript + Next.js App Router，组件命名采用 `PascalCase`，hooks/工具使用 `camelCase`；尽量将 UI 状态封装为客户端组件，网络请求统一调用 `/api` 代理或现有 FastAPI 端点。 Keep styles colocated via CSS modules or inline Tailwind classes for consistency.

## Testing Guidelines
- 仓库尚未附带后端测试，请在根目录建立 `tests/`，以 `test_<module>.py` 命名，使用 `pytest -q` 作为默认命令，并针对每个工具函数提供成功与失败分支的协程测试，目标覆盖率 ≥80%。 Add fixtures for external HTTP clients and patch network calls to keep tests deterministic.
- 前端建议在 `frontend/src/__tests__/` 下采用 React Testing Library；新增 `npm run test`（映射至 `next test` 或 `vitest run`）后，命名遵循 `<Component>.test.tsx`，同时通过 `npm run lint`（Next 自带）保证 JSX/TS 规范。 Snapshot tests should be paired with meaningful interaction assertions.

## Evaluation Guidelines
- 评估相关内容集中在 `eval/` 下，评估集与模板在 `eval/testsets/`，脚本在 `eval/scripts/`，报告建议输出到 `eval/reports/`。
- 运行自动评分：`python eval/scripts/grade_testset_v2.py --answers path/to/answers.json`；如需指定评估集或输出报告，可用 `--testset eval/testsets/testset_v2.json --output eval/reports/report.json`。
- 运行在线评估（默认流式 `/chat/stream`）：`python eval/scripts/run_eval_stream.py --base-url http://127.0.0.1:9000 --report eval/reports/report.json`，默认输出 `eval/reports/answers.json`。
- 如需非流式接口：`--mode chat --endpoint /chat`（当前评估无需鉴权）。

## Commit & Pull Request Guidelines
- 仓库已初始化 Git，请继续遵循 Conventional Commits（如 `feat: add web_search retries`、`fix: guard empty query`）保持可读性；单次提交聚焦单一功能或缺陷修复。 Commits should stay atomic on the `main` branch unless stated otherwise.
- PR 需包含变更摘要、验证方式（命令输出或截图）、相关 Issue 链接以及潜在风险；若触及 env/脚本，请同时更新 `start_services.sh` 或 README 片段以免部署偏差。 Request reviewers familiar with both FastAPI and Next.js when changes cross the stack.

## Security & Configuration Tips
- `.env` 必须提供 `SUPER_MIND_API_KEY` 与可选 `CHAT_ALLOWED_ORIGINS`；不要将密钥写入日志或前端 bundle，可通过 `os.getenv` 访问并在启动时校验。 Keep the `.env` file out of version control.
- 生产部署需将 `frontend/out` 置于受控 CDN，并以 `uvicorn main:app --proxy-headers --forwarded-allow-ips="*"` 运行后端；任何外部请求都应保持 20s 超时与错误日志，以免工具链卡死。 Rotate API tokens regularly and scrub `backend.log` before sharing.

## 其他说明
默认使用中文进行说明与讨论，除非内容为代码片段、命令或规范要求英文表述。

## 运行规范补充
- `start_services.sh` 现支持 `./start_services.sh [start|stop|restart|status] [all|backend|frontend|mcp]`，可按需单独管理各服务；`status` 仍会附带端口与孤儿进程检测。
- 修改不同模块时请按以下策略重启并等待脚本输出 `服务健康检查：OK`：
  - 仅改动后端（如 `main.py`、工具脚本等）：执行 `./start_services.sh restart backend`
  - 仅改动前端（`frontend/` 范围）：执行 `./start_services.sh restart frontend`
  - 同时影响前后端、MCP 或更新 `start_services.sh` 本身：执行 `./start_services.sh restart all`（或省略目标，默认 all）
  - 若脚本返回非 0 或显示 `FAILED`，需立即查看对应日志（`backend.log` / `frontend.log` / `mcp_interpreter.log`）定位问题并修复后再重启。
- 脚本健康检查逻辑：
  - 正常：后端打印 `OK (端口 9000, PID xxxx)`，前端打印 `OK (HTTP 检测通过，端口 9080)`，MCP 打印 `OK (端口 9070, PID xxxx)`。
  - 异常：会提示 `FAILED - 端口 xxxx 未监听` 或列出匹配进程，此时请按提示检查对应日志及 PID。

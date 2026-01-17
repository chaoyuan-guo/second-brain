FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend

ENV NEXT_TELEMETRY_DISABLED=1

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend ./
RUN npm run build


FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 安装 Python 依赖
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 复制后端、脚本及数据
COPY backend ./backend
COPY scripts ./scripts
COPY data ./data

# 复制前端静态导出产物（在构建期生成，避免依赖仓库内产物）
COPY --from=frontend-builder /app/frontend/out ./frontend/out

# 创建运行时必需目录
RUN mkdir -p runtime/logs data/notes/indexes

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

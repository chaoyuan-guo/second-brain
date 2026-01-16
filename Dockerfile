FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

# 仅复制 package 配置以利用缓存
COPY frontend/package*.json ./
RUN npm install --production=false

# 复制前端源码并构建静态导出
COPY frontend ./
RUN npm run build


FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 安装运行所需基础依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 复制后端、脚本及数据
COPY backend ./backend
COPY scripts ./scripts
COPY data ./data

# 复制前端编译产物
COPY --from=frontend-builder /app/frontend/out ./frontend/out

# 创建运行时必需目录
RUN mkdir -p runtime/logs data/notes/indexes

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

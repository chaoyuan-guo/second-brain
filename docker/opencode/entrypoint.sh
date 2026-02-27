#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/app"
cd "$APP_DIR"

OPENCODE_HOST="${OPENCODE_HOST:-0.0.0.0}"
OPENCODE_PORT="${OPENCODE_PORT:-9090}"
RAG_HOST="${RAG_HOST:-0.0.0.0}"
RAG_PORT="${RAG_PORT:-9070}"
FRONTEND_HOST="${FRONTEND_HOST:-0.0.0.0}"
FRONTEND_PORT="${FRONTEND_PORT:-9080}"

cleanup() {
  local code=$?
  jobs -p | xargs -r kill 2>/dev/null || true
  wait || true
  exit "$code"
}
trap cleanup EXIT INT TERM

echo "[entrypoint] starting rag-data server on ${RAG_HOST}:${RAG_PORT}"
python -m uvicorn backend.app.main:app --host "$RAG_HOST" --port "$RAG_PORT" &

echo "[entrypoint] starting opencode server on ${OPENCODE_HOST}:${OPENCODE_PORT}"
opencode serve --hostname "$OPENCODE_HOST" --port "$OPENCODE_PORT" &

echo "[entrypoint] starting frontend on ${FRONTEND_HOST}:${FRONTEND_PORT}"
./frontend/node_modules/.bin/next start frontend -H "$FRONTEND_HOST" -p "$FRONTEND_PORT" &

wait -n

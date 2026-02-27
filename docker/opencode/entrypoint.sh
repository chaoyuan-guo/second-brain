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
OPENCODE_LOG_LEVEL="${OPENCODE_LOG_LEVEL:-WARN}"
OPENCODE_PRINT_LOGS="${OPENCODE_PRINT_LOGS:-0}"
OPENCODE_SELF_CHECK="${OPENCODE_SELF_CHECK:-1}"

require_api_token() {
  if [ -n "${SUPER_MIND_API_KEY:-}" ] || [ -n "${AI_BUILDER_TOKEN:-}" ]; then
    return 0
  fi
  echo "[entrypoint] ERROR: missing API token." >&2
  echo "[entrypoint] Set SUPER_MIND_API_KEY (preferred) or AI_BUILDER_TOKEN in .env.docker." >&2
  exit 1
}

normalize_base_url() {
  local raw="${1:-}"
  raw="${raw%/}"
  if [ -z "$raw" ]; then
    echo ""
    return 0
  fi
  if [[ "$raw" == */v1 ]]; then
    echo "$raw"
  else
    echo "$raw/v1"
  fi
}

SUPER_MIND_API_BASE_URL="$(normalize_base_url "${SUPER_MIND_API_BASE_URL:-https://space.ai-builders.com/backend/v1}")"
export SUPER_MIND_API_BASE_URL
require_api_token

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
OPENCODE_LOG_ARGS=(--log-level "$OPENCODE_LOG_LEVEL")
if [ "$OPENCODE_PRINT_LOGS" = "1" ]; then
  OPENCODE_LOG_ARGS+=(--print-logs)
fi
opencode serve "${OPENCODE_LOG_ARGS[@]}" --hostname "$OPENCODE_HOST" --port "$OPENCODE_PORT" &

echo "[entrypoint] starting frontend on ${FRONTEND_HOST}:${FRONTEND_PORT}"
./frontend/node_modules/.bin/next start frontend -H "$FRONTEND_HOST" -p "$FRONTEND_PORT" &

if [ "$OPENCODE_SELF_CHECK" = "1" ]; then
  echo "[entrypoint] running opencode startup self-check"
  ./docker/opencode/selfcheck.sh
fi

wait -n

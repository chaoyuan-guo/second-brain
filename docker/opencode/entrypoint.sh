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
OPENCODE_LOG_LEVEL="${OPENCODE_LOG_LEVEL:-INFO}"
OPENCODE_PRINT_LOGS="${OPENCODE_PRINT_LOGS:-1}"
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

# OpenCode provider 配置默认读取 AI_BUILDER_TOKEN；若仅提供 SUPER_MIND_API_KEY，则自动复用。
if [ -z "${AI_BUILDER_TOKEN:-}" ] && [ -n "${SUPER_MIND_API_KEY:-}" ]; then
  export AI_BUILDER_TOKEN="${SUPER_MIND_API_KEY}"
fi
# 反向兜底，避免仅设置 AI_BUILDER_TOKEN 时其他组件读取不到 SUPER_MIND_API_KEY。
if [ -z "${SUPER_MIND_API_KEY:-}" ] && [ -n "${AI_BUILDER_TOKEN:-}" ]; then
  export SUPER_MIND_API_KEY="${AI_BUILDER_TOKEN}"
fi
export SUPER_MIND_API_KEY AI_BUILDER_TOKEN

# 兼容 .env 中历史键名 azure_api-version（带短横线）
if [ -z "${azure_api_version:-}" ]; then
  azure_api_version_legacy="$(printenv 'azure_api-version' 2>/dev/null || true)"
  if [ -n "$azure_api_version_legacy" ]; then
    export azure_api_version="$azure_api_version_legacy"
  fi
fi

# Azure baseURL 归一化（给 OpenCode 的 openai-compatible provider 使用）：
# 目标为 https://<host>/openai/v1
if [ -n "${azure_base_url:-}" ]; then
  azure_base_url="${azure_base_url%/}"
  lowered_base="$(printf '%s' "$azure_base_url" | tr '[:upper:]' '[:lower:]')"
  if [[ "$lowered_base" == */openai/v1 ]]; then
    :
  elif [[ "$lowered_base" == */openai ]]; then
    azure_base_url="${azure_base_url}/v1"
  elif [[ "$lowered_base" == */v1 ]]; then
    azure_base_url="${azure_base_url%/v1}/openai/v1"
  else
    azure_base_url="${azure_base_url}/openai/v1"
  fi
  export azure_base_url
fi
export OPENCODE_AZURE_BASE_URL="${azure_base_url:-}"

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

echo "[entrypoint] logging: opencode_level=${OPENCODE_LOG_LEVEL} opencode_print_logs=${OPENCODE_PRINT_LOGS} backend_log_to_stdout=${LOG_TO_STDOUT:-auto} backend_log_to_file=${LOG_TO_FILE:-auto}"
echo "[entrypoint] auth: super_mind_api_key_len=${#SUPER_MIND_API_KEY} ai_builder_token_len=${#AI_BUILDER_TOKEN} base_url=${SUPER_MIND_API_BASE_URL}"
AZURE_API_VERSION_LEN=0
if [ -n "${azure_api_version:-}" ]; then
  AZURE_API_VERSION_LEN=${#azure_api_version}
fi
echo "[entrypoint] azure: base_url=${azure_base_url:-<empty>} opencode_base_url=${OPENCODE_AZURE_BASE_URL:-<empty>} api_version_len=${AZURE_API_VERSION_LEN} model=${azure_use_model:-<empty>}"

if [ "$OPENCODE_SELF_CHECK" = "1" ]; then
  echo "[entrypoint] running opencode startup self-check"
  ./docker/opencode/selfcheck.sh
fi

wait -n

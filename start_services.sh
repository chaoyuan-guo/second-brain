#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
RUNTIME_DIR="$PROJECT_ROOT/runtime"
LOG_DIR="$RUNTIME_DIR/logs"
mkdir -p "$LOG_DIR"
BACKEND_LOG="$LOG_DIR/backend.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"
BACKEND_PID_FILE="$PROJECT_ROOT/.backend.pid"
FRONTEND_PID_FILE="$PROJECT_ROOT/.frontend.pid"
BACKEND_PORT=9000
FRONTEND_PORT=9080
MCP_LOG="$LOG_DIR/mcp_interpreter.log"
MCP_PID_FILE="$PROJECT_ROOT/.mcp_interpreter.pid"
MCP_PORT=9070
MCP_ENDPOINT="http://127.0.0.1:${MCP_PORT}/sse/"
MCP_ENV_DIR="$PROJECT_ROOT/.mcp_env"
MCP_BIN_DIR="$MCP_ENV_DIR/bin"
MCP_FASTMCP="$MCP_BIN_DIR/fastmcp"
MCP_PYTHON="$MCP_BIN_DIR/python"
MCP_WORKDIR="$RUNTIME_DIR/mcp_workspace"
mkdir -p "$MCP_WORKDIR"
MCP_SYNC_SOURCE_DIRS=("data/notes/my_markdowns")

sync_mcp_workspace() {
  mkdir -p "$MCP_WORKDIR"
  for rel in "${MCP_SYNC_SOURCE_DIRS[@]}"; do
    local src="$PROJECT_ROOT/$rel"
    local dest="$MCP_WORKDIR/$rel"
    if [ ! -d "$src" ]; then
      continue
    fi
    mkdir -p "$(dirname "$dest")"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --delete "$src/" "$dest/"
    else
      rm -rf "$dest"
      mkdir -p "$dest"
      cp -a "$src/." "$dest/"
    fi
  done
}

command -v lsof >/dev/null 2>&1 && HAVE_LSOF=1 || HAVE_LSOF=0
command -v fuser >/dev/null 2>&1 && HAVE_FUSER=1 || HAVE_FUSER=0

usage() {
  cat <<'EOF'
用法: ./start_services.sh [start|stop|restart|status] [all|backend|frontend|mcp]

start    启动指定服务（默认 all 同时启动前后端与 MCP）
stop     停止指定服务（默认 all 停止前后端与 MCP）
restart  重启指定服务（默认 all）
status   查看指定服务状态（默认 all）

示例:
  ./start_services.sh start frontend   # 仅启动前端
  ./start_services.sh stop backend     # 仅停止后端
EOF
}

ensure_command() {
  local cmd="$1"
  local hint="$2"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "缺少依赖命令: $cmd ($hint)" >&2
    exit 1
  fi
}

ensure_backend_requirements() {
  ensure_command uvicorn "请先安装 'uvicorn' (pip install uvicorn)。"
}

ensure_frontend_requirements() {
  ensure_command npm "请先安装 Node.js 与 npm。"
}

ensure_mcp_requirements() {
  ensure_command "$MCP_FASTMCP" "请先在 .mcp_env 中安装 fastmcp。"
}

service_label() {
  case "$1" in
    backend)
      echo "后端服务"
      ;;
    frontend)
      echo "前端服务"
      ;;
    mcp)
      echo "MCP 解释器"
      ;;
    *)
      echo "$1"
      ;;
  esac
}

describe_services() {
  if [ $# -eq 0 ]; then
    echo "无"
    return
  fi
  local labels=()
  local svc
  for svc in "$@"; do
    labels+=("$(service_label "$svc")")
  done
  local IFS='、'
  echo "${labels[*]}"
}

verify_target() {
  case "$1" in
    backend)
      verify_service_started \
        "后端服务" "$BACKEND_PORT" "$BACKEND_LOG" "uvicorn backend.app.main:app --port $BACKEND_PORT"
      ;;
    frontend)
      verify_service_started \
        "前端服务" "$FRONTEND_PORT" "$FRONTEND_LOG" "next dev -p $FRONTEND_PORT"
      ;;
    mcp)
      verify_service_started \
        "MCP 解释器" "$MCP_PORT" "$MCP_LOG" "fastmcp run"
      ;;
    *)
      return 0
      ;;
  esac
}

pids_alive() {
  local pid="$1"
  if [ -z "$pid" ]; then
    return 1
  fi
  if ! ps -p "$pid" >/dev/null 2>&1; then
    return 1
  fi
  local state
  state=$(ps -p "$pid" -o stat= 2>/dev/null | awk '{print $1}')
  if [[ "$state" == Z* ]]; then
    return 1
  fi
  return 0
}

pids_on_port() {
  local port="$1"
  local result=""
  if [ "$HAVE_LSOF" -eq 1 ]; then
    result="$(lsof -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | tr '\n' ' ')"
  elif [ "$HAVE_FUSER" -eq 1 ]; then
    result="$(fuser "${port}/tcp" 2>/dev/null | tr ' ' '\n' | grep -Eo '^[0-9]+$' | tr '\n' ' ')"
  fi
  if [ -z "$result" ] && command -v python3 >/dev/null 2>&1; then
    result="$(
      PORT="$port" python3 - <<'PY'
import os
import sys

port = int(os.environ.get("PORT", "0"))
if port <= 0:
    sys.exit(0)
target = f"{port:04X}"
inodes = set()
for table in ("/proc/net/tcp", "/proc/net/tcp6"):
    try:
        with open(table) as fh:
            next(fh)
            for line in fh:
                parts = line.split()
                if len(parts) < 10:
                    continue
                if parts[3] != "0A":
                    continue
                local = parts[1].split(':')[1]
                if local == target:
                    inodes.add(parts[9])
    except FileNotFoundError:
        continue

if not inodes:
    sys.exit(0)

pids = []
for pid in os.listdir('/proc'):
    if not pid.isdigit():
        continue
    fd_dir = os.path.join('/proc', pid, 'fd')
    try:
        for fd in os.listdir(fd_dir):
            path = os.path.join(fd_dir, fd)
            try:
                link = os.readlink(path)
            except OSError:
                continue
            if link.startswith('socket:['):
                inode = link[8:-1]
                if inode in inodes:
                    pids.append(pid)
                    break
    except PermissionError:
        continue

if pids:
    print(' '.join(sorted(set(pids), key=lambda x: int(x))))
PY
    )"
  fi
  echo "$result"
}

wait_for_exit() {
  local pids="$1"
  local attempts=0
  while [ $attempts -lt 10 ]; do
    local all_down=1
    for pid in $pids; do
      if pids_alive "$pid"; then
        all_down=0
        break
      fi
    done
    if [ $all_down -eq 1 ]; then
      return 0
    fi
    sleep 0.5
    attempts=$((attempts + 1))
  done
  kill -9 $pids 2>/dev/null || true
}

stop_service() {
  local name="$1"
  local pid_file="$2"
  local port="$3"
  local signature="${4:-}"

  local killed=0
  while true; do
    local pids=""
    if [ -f "$pid_file" ]; then
      local file_pid
      file_pid="$(cat "$pid_file")"
      if pids_alive "$file_pid"; then
        pids="$file_pid"
      else
        rm -f "$pid_file"
      fi
    fi

    if [ -z "$pids" ] && [ -n "$port" ]; then
      pids="$(pids_on_port "$port")"
    fi

    if [ -z "$pids" ] && [ -n "$signature" ]; then
      pids="$(pgrep -f "$signature" 2>/dev/null | tr '\n' ' ' || true)"
    fi

    if [ -z "$pids" ]; then
      if [ $killed -eq 0 ]; then
        echo "$name 未在运行"
      fi
      break
    fi

    killed=1
    echo "正在停止 $name: $pids"
    kill $pids 2>/dev/null || true
    wait_for_exit "$pids"

    sleep 0.5
  done

  rm -f "$pid_file"
}

status_service() {
  local name="$1"
  local pid_file="$2"
  local port="$3"
  local signature="${4:-}"
  local status="未在运行"
  local pid=""

  if [ -f "$pid_file" ]; then
    pid="$(cat "$pid_file")"
    if pids_alive "$pid"; then
      status="运行中 (PID $pid)"
    fi
  fi

  if [ "$status" = "未在运行" ] && [ -n "$port" ]; then
    local port_pids
    port_pids="$(pids_on_port "$port")"
    if [ -n "$port_pids" ]; then
      status="端口 ${port} 被进程 ${port_pids} 占用"
    fi
  fi

  if [ "$status" = "未在运行" ] && [ -n "$signature" ]; then
    local sig_pids
    sig_pids="$(pgrep -f "$signature" 2>/dev/null | tr '\n' ' ' || true)"
    if [ -n "$sig_pids" ]; then
      status="检测到匹配进程 ${sig_pids} (signature: $signature)"
    fi
  fi

  echo "$name: $status"
}

verify_service_started() {
  local name="$1"
  local port="$2"
  local log_path="$3"
  local signature="${4:-}"
  local attempts=0
  local max_attempts=60 # 30 秒
  local curl_url="http://127.0.0.1:$port"

  if [ -z "$port" ]; then
    echo "$name: 未配置端口，跳过健康检查"
    return 0
  fi

  while [ $attempts -lt $max_attempts ]; do
    local port_pids
    port_pids="$(pids_on_port "$port")"
    if [ -n "$port_pids" ]; then
      echo "$name: OK (端口 $port, PID $port_pids)"
      return 0
    fi
    sleep 0.5
    attempts=$((attempts + 1))
  done

  local sig_pids=""
  if [ -n "$signature" ]; then
    sig_pids="$(pgrep -f "$signature" 2>/dev/null | tr '\n' ' ' || true)"
  fi

  if [ -n "$port" ]; then
    if curl --noproxy '*' --max-time 3 -s -o /dev/null "$curl_url" >/dev/null 2>&1; then
      echo "$name: OK (HTTP 检测通过，端口 $port)"
      return 0
    fi
  fi

  echo "$name: FAILED - 端口 $port 未监听，查看日志 $log_path" >&2
  if [ -n "$sig_pids" ]; then
    echo "$name: 匹配进程: $sig_pids" >&2
  fi
  return 1
}

start_backend() {
  stop_service "后端服务" "$BACKEND_PID_FILE" "$BACKEND_PORT" "uvicorn backend.app.main:app --port $BACKEND_PORT"
  : > "$BACKEND_LOG"
  cd "$PROJECT_ROOT"
  MCP_SSE_ENDPOINT="${MCP_SSE_ENDPOINT:-$MCP_ENDPOINT}" \
    nohup uvicorn backend.app.main:app --host 0.0.0.0 --port "$BACKEND_PORT" >> "$BACKEND_LOG" 2>&1 &
  local pid=$!
  echo "$pid" > "$BACKEND_PID_FILE"
  echo "后端服务已启动 (PID $pid) 日志: $BACKEND_LOG"
}

start_frontend() {
  stop_service "前端服务" "$FRONTEND_PID_FILE" "$FRONTEND_PORT" "next dev -p $FRONTEND_PORT"
  : > "$FRONTEND_LOG"
  cd "$PROJECT_ROOT/frontend"
  local node_options_default="--max_old_space_size=6144"
  local node_options="${FRONTEND_NODE_OPTIONS:-${NODE_OPTIONS:-$node_options_default}}"
  NEXT_FORCE_WEBPACK="${NEXT_FORCE_WEBPACK:-1}" \
    NODE_OPTIONS="$node_options" nohup npm run dev >> "$FRONTEND_LOG" 2>&1 &
  local pid=$!
  echo "$pid" > "$FRONTEND_PID_FILE"
  echo "前端服务已启动 (PID $pid) 日志: $FRONTEND_LOG"
}

start_mcp() {
  if [ ! -x "$MCP_FASTMCP" ]; then
    echo "缺少 fastmcp，可执行文件 $MCP_FASTMCP 不存在" >&2
    exit 1
  fi

  stop_service "MCP 解释器" "$MCP_PID_FILE" "$MCP_PORT" "fastmcp run"
  : > "$MCP_LOG"
  sync_mcp_workspace
  nohup "$MCP_FASTMCP" run -t sse "$MCP_BIN_DIR/../lib/python3.10/site-packages/mcp_python_interpreter/server.py" \
    --host 127.0.0.1 --port "$MCP_PORT" -- --dir "$MCP_WORKDIR" >> "$MCP_LOG" 2>&1 &
  local pid=$!
  echo "$pid" > "$MCP_PID_FILE"
  echo "MCP 解释器已启动 (PID $pid) 日志: $MCP_LOG"
}

start_all() {
  ensure_backend_requirements
  ensure_frontend_requirements
  ensure_mcp_requirements
  start_backend
  start_frontend
  start_mcp
  check_services backend frontend mcp
}

stop_backend() {
  stop_service "后端服务" "$BACKEND_PID_FILE" "$BACKEND_PORT" "uvicorn backend.app.main:app --port $BACKEND_PORT"
}

stop_frontend() {
  stop_service "前端服务" "$FRONTEND_PID_FILE" "$FRONTEND_PORT" "next dev -p $FRONTEND_PORT"
}

stop_mcp() {
  stop_service "MCP 解释器" "$MCP_PID_FILE" "$MCP_PORT" "fastmcp run"
}

status_backend() {
  status_service "后端服务" "$BACKEND_PID_FILE" "$BACKEND_PORT" "uvicorn backend.app.main:app --port $BACKEND_PORT"
}

status_frontend() {
  status_service "前端服务" "$FRONTEND_PID_FILE" "$FRONTEND_PORT" "next dev -p $FRONTEND_PORT"
}

status_mcp() {
  status_service "MCP 解释器" "$MCP_PID_FILE" "$MCP_PORT" "fastmcp run"
}

START_ACTION="${1:-start}"
TARGET="${2:-all}"

resolve_targets() {
  local selection="$1"
  case "$selection" in
    all)
      echo "backend frontend mcp"
      ;;
    backend|frontend|mcp)
      echo "$selection"
      ;;
    *)
      echo "未知服务: $selection" >&2
      usage >&2
      exit 1
      ;;
  esac
}

check_services() {
  local services=("$@")
  local errors=0
  local svc
  for svc in "${services[@]}"; do
    if ! verify_target "$svc"; then
      errors=1
    fi
  done

  if [ $errors -eq 0 ]; then
    echo "服务健康检查：OK"
    echo "$(describe_services "${services[@]}") 已完成启动。"
  else
    echo "服务健康检查：FAILED (请根据上述提示查看日志)" >&2
    exit 1
  fi
}

run_action() {
  local action="$1"
  shift
  local services=("$@")
  case "$action" in
    start)
      local svc
      for svc in "${services[@]}"; do
        case "$svc" in
          backend)
            ensure_backend_requirements
            start_backend
            ;;
          frontend)
            ensure_frontend_requirements
            start_frontend
            ;;
          mcp)
            ensure_mcp_requirements
            start_mcp
            ;;
        esac
      done
      check_services "${services[@]}"
      ;;
    stop)
      local svc
      for svc in "${services[@]}"; do
        case "$svc" in
          backend)
            stop_backend
            ;;
          frontend)
            stop_frontend
            ;;
          mcp)
            stop_mcp
            ;;
        esac
      done
      ;;
    restart)
      run_action stop "${services[@]}"
      run_action start "${services[@]}"
      ;;
    status)
      local svc
      for svc in "${services[@]}"; do
        case "$svc" in
          backend)
            status_backend
            ;;
          frontend)
            status_frontend
            ;;
          mcp)
            status_mcp
            ;;
        esac
      done
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

TARGET_SERVICES=($(resolve_targets "$TARGET"))

run_action "$START_ACTION" "${TARGET_SERVICES[@]}"

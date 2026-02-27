#!/usr/bin/env bash
set -euo pipefail

OPENCODE_URL="${OPENCODE_URL:-http://127.0.0.1:9090}"
OPENCODE_SELF_CHECK_TIMEOUT="${OPENCODE_SELF_CHECK_TIMEOUT:-45}"
OPENCODE_SELF_CHECK_PATH="${OPENCODE_SELF_CHECK_PATH:-/app}"
SELF_CHECK_PROMPT="${OPENCODE_SELF_CHECK_PROMPT:-Please reply with pong only.}"

log() {
  printf '[selfcheck] %s\n' "$*"
}

EVENT_PID=""
EVENT_FILE=""

cleanup() {
  if [ -n "$EVENT_PID" ]; then
    kill "$EVENT_PID" >/dev/null 2>&1 || true
    wait "$EVENT_PID" >/dev/null 2>&1 || true
  fi
  if [ -n "$EVENT_FILE" ]; then
    rm -f "$EVENT_FILE"
  fi
}

trap cleanup EXIT

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "missing required command: $1"
    exit 1
  fi
}

wait_for_opencode() {
  local deadline=$((SECONDS + OPENCODE_SELF_CHECK_TIMEOUT))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if curl -fsS "$OPENCODE_URL/session" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

main() {
  require_cmd curl
  require_cmd jq
  require_cmd python

  log "checking opencode API at ${OPENCODE_URL}"
  if ! wait_for_opencode; then
    log "opencode API not ready within ${OPENCODE_SELF_CHECK_TIMEOUT}s"
    exit 1
  fi

  local session_json
  session_json="$(curl -fsS -X POST "$OPENCODE_URL/session" \
    -H 'content-type: application/json' \
    -d "{\"path\":\"${OPENCODE_SELF_CHECK_PATH}\"}")"

  local session_id
  session_id="$(printf '%s' "$session_json" | jq -r '.id // empty')"
  if [ -z "$session_id" ]; then
    log "failed to create session: $session_json"
    exit 1
  fi
  log "created session: ${session_id}"

  EVENT_FILE="$(mktemp)"
  curl -sN "$OPENCODE_URL/event" >"$EVENT_FILE" &
  EVENT_PID=$!

  sleep 1
  local prompt_status
  prompt_status="$(curl -sS -o /tmp/opencode_selfcheck_prompt.body -w '%{http_code}' \
    -X POST "$OPENCODE_URL/session/${session_id}/prompt_async" \
    -H 'content-type: application/json' \
    -d "{\"parts\":[{\"type\":\"text\",\"text\":\"${SELF_CHECK_PROMPT}\"}]}")"

  if [ "$prompt_status" != "204" ]; then
    log "prompt_async failed with status=$prompt_status"
    cat /tmp/opencode_selfcheck_prompt.body || true
    exit 1
  fi

  if ! SESSION_ID="$session_id" EVENT_FILE="$EVENT_FILE" TIMEOUT="$OPENCODE_SELF_CHECK_TIMEOUT" python - <<'PY'
import json
import os
import time

sid = os.environ["SESSION_ID"]
path = os.environ["EVENT_FILE"]
timeout = int(os.environ["TIMEOUT"])

assistant_id = None
saw_finish = False
assistant_text = ""
pos = 0
deadline = time.time() + timeout

while time.time() < deadline:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        f.seek(pos)
        chunk = f.read()
        pos = f.tell()

    for line in chunk.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        try:
            obj = json.loads(payload)
        except Exception:
            continue

        if obj.get("type") == "session.error":
            print("session.error", obj)
            raise SystemExit(2)

        if obj.get("type") != "message.part.updated":
            continue

        props = obj.get("properties") or {}
        part = props.get("part") or {}
        if part.get("sessionID") != sid:
            continue

        part_type = part.get("type")
        message_id = part.get("messageID")
        if part_type == "step-start" and message_id and not assistant_id:
            assistant_id = message_id
            continue

        if part_type == "text":
            text = part.get("text")
            if assistant_id and message_id != assistant_id:
                continue
            if isinstance(text, str) and text.strip():
                assistant_text = text
            continue

        if part_type == "step-finish":
            if assistant_id and message_id != assistant_id:
                continue
            saw_finish = True

    if saw_finish and assistant_text.strip():
        print(assistant_text.strip())
        raise SystemExit(0)

    time.sleep(0.3)

print("timeout waiting for assistant reply")
raise SystemExit(3)
PY
  then
    log "failed to observe valid assistant response"
    exit 1
  fi

  log "opencode session/prompt/event selfcheck passed"
}

main "$@"

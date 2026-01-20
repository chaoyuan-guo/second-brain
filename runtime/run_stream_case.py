import json
import sys
import time
from typing import Optional

import httpx

URL = "http://127.0.0.1:9000/chat/stream"
HEADERS = {"accept": "application/x-ndjson"}
TIMEOUT = httpx.Timeout(120.0, read=120.0)


def stream_chat(query: str, *, max_seconds: float = 180.0) -> tuple[str, Optional[str]]:
    payload = {"user_message": query}
    content = ""
    start = time.monotonic()
    try:
        with httpx.Client(timeout=TIMEOUT, trust_env=False) as client:
            with client.stream("POST", URL, headers=HEADERS, json=payload) as resp:
                resp.raise_for_status()
                line_iter = resp.iter_lines()
                while True:
                    if time.monotonic() - start > max_seconds:
                        return content, "TimeoutExceeded"
                    try:
                        line = next(line_iter)
                    except StopIteration:
                        break
                    if not line:
                        continue
                    event = json.loads(line)
                    if event.get("type") == "delta":
                        content += event.get("delta", "")
                    elif event.get("type") == "done":
                        break
    except Exception as exc:  # noqa: BLE001
        return content, f"{type(exc).__name__}: {exc}"
    return content, None


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: run_stream_case.py <query>")
        return 2
    query = sys.argv[1]
    answer, error = stream_chat(query)
    result = {"query": query, "answer": answer, "error": error}
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

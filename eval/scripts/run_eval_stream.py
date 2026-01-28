#!/usr/bin/env python3
"""Run evaluation by calling the streaming chat endpoint and collecting answers.

Usage:
  python eval/scripts/run_eval_stream.py --base-url http://127.0.0.1:9000
  python eval/scripts/run_eval_stream.py --testset eval/testsets/testset.json --out eval/reports/answers.json
  python eval/scripts/run_eval_stream.py --report eval/reports/report.json
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest


def load_testset(path: Path) -> List[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("questions", [])


def parse_headers(values: List[str]) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    for item in values:
        if ":" not in item:
            continue
        key, val = item.split(":", 1)
        key = key.strip()
        val = val.strip()
        if key:
            headers[key] = val
    return headers


def stream_chat(
    url: str,
    payload: dict,
    timeout: int,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """发送流式聊天请求并收集答案和工具事件。

    Returns:
        (答案文本, 工具调用事件列表)
    """
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urlrequest.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/x-ndjson")
    req.add_header("x-stream-format", "ndjson")
    if extra_headers:
        for key, val in extra_headers.items():
            req.add_header(key, val)

    answer_parts: List[str] = []
    tool_events: List[Dict[str, Any]] = []

    with urlrequest.urlopen(req, timeout=timeout) as resp:
        while True:
            raw = resp.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type")

            if event_type == "delta":
                delta = event.get("delta")
                if isinstance(delta, str):
                    answer_parts.append(delta)
            elif event_type == "tool":
                # 收集工具调用事件
                tool_events.append({
                    "stage": event.get("stage"),
                    "tool_name": event.get("tool_name"),
                    "tool_call_id": event.get("tool_call_id"),
                    "latency_ms": event.get("latency_ms"),
                    "error": event.get("error"),
                })
            elif event_type == "done":
                break

    return "".join(answer_parts).strip(), tool_events


def chat_once(
    url: str,
    payload: dict,
    timeout: int,
    extra_headers: Optional[Dict[str, str]] = None,
) -> str:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urlrequest.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if extra_headers:
        for key, val in extra_headers.items():
            req.add_header(key, val)

    with urlrequest.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    try:
        payload_json = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip()
    response = payload_json.get("response")
    if isinstance(response, str):
        return response.strip()
    return ""


def run_eval(
    questions: Iterable[dict],
    base_url: str,
    endpoint: str,
    timeout: int,
    pause: float,
    mode: str,
    headers: Dict[str, str],
    strict_sources: bool,
    concurrency: int,
    limit: Optional[int] = None,
) -> Tuple[Dict[str, str], Dict[str, List[Dict[str, Any]]]]:
    """运行评估，返回答案和工具追踪。

    Returns:
        (answers: Dict[question_id, answer], tool_traces: Dict[question_id, tool_events])
    """
    answers: Dict[str, str] = {}
    tool_traces: Dict[str, List[Dict[str, Any]]] = {}
    url = base_url.rstrip("/") + endpoint

    def _run_single(q: dict) -> Tuple[str, str, List[Dict[str, Any]]]:
        qid = q["id"]
        query = q["query"]
        payload = {"user_message": query}
        request_headers = dict(headers)
        if strict_sources:
            request_headers["X-Eval-Strict"] = "1"
            request_headers["X-Eval-Question-Id"] = qid
            sources = q.get("expected_sources") or q.get("sources") or []
            if sources:
                request_headers["X-Eval-Expected-Sources"] = ",".join(
                    quote(source, safe="/._-") for source in sources
                )
        tool_events: List[Dict[str, Any]] = []
        try:
            if mode == "chat":
                answer = chat_once(url, payload, timeout, request_headers)
            else:
                answer, tool_events = stream_chat(url, payload, timeout, request_headers)
        except (TimeoutError, socket.timeout):
            answer = "[timeout]"
        except urlerror.HTTPError as exc:
            answer = f"[http_error] {exc.code} {exc.reason}"
        except urlerror.URLError as exc:
            answer = f"[url_error] {exc.reason}"
        if pause > 0:
            time.sleep(pause)
        return qid, answer, tool_events

    queued: List[dict] = []
    for idx, q in enumerate(questions, start=1):
        if limit and idx > limit:
            break
        queued.append(q)

    worker_count = max(1, concurrency)
    total = len(queued)
    completed = 0

    print(f"开始评估: 共 {total} 题，并发数 {worker_count}")

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_run_single, q) for q in queued]
        for future in as_completed(futures):
            qid, answer, tool_events = future.result()
            answers[qid] = answer
            tool_traces[qid] = tool_events
            completed += 1
            print(f"  [{completed}/{total}] {qid} 完成")

    print(f"✓ 评估完成，已生成 {len(answers)} 个答案")
    return answers, tool_traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Run streaming eval against chat API")
    parser.add_argument("--testset", default="eval/testsets/testset.json")
    parser.add_argument("--base-url", default="http://127.0.0.1:9000")
    parser.add_argument("--endpoint", default="/chat/stream")
    parser.add_argument("--mode", choices=["stream", "chat"], default="stream")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--pause", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("-H", "--header", action="append", default=[], help="Extra header, e.g. 'Authorization: Bearer x'")
    parser.add_argument("--out", default="eval/reports/answers.json")
    parser.add_argument("--report", help="Optional report JSON path; runs grader after answering")
    parser.add_argument("--strict-sources", action="store_true", help="Require answers to include note sources")
    parser.add_argument("--recall-k", default="1,3,5,10", help="Compute recall@k (comma-separated integers, default: 1,3,5,10)")
    args = parser.parse_args()

    questions = load_testset(Path(args.testset))
    headers = parse_headers(args.header)

    answers, tool_traces = run_eval(
        questions,
        base_url=args.base_url,
        endpoint=args.endpoint,
        timeout=args.timeout,
        pause=args.pause,
        mode=args.mode,
        headers=headers,
        strict_sources=args.strict_sources,
        concurrency=args.concurrency,
        limit=args.limit,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ 答案已保存至: {out_path}")

    # 保存工具追踪
    tool_traces_path = out_path.parent / (out_path.stem + "_tool_traces.json")
    tool_traces_path.write_text(json.dumps(tool_traces, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ 工具追踪已保存至: {tool_traces_path}")

    if args.report:
        print(f"\n开始评分...")
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_cmd = [
            "python",
            "eval/scripts/grade_testset.py",
            "--answers",
            str(out_path),
            "--output",
            str(report_path),
            "--tool-traces",
            str(tool_traces_path),
        ]
        if args.strict_sources:
            report_cmd.append("--require-sources")
        if args.recall_k:
            report_cmd += ["--recall-k", args.recall_k]
        if args.testset:
            report_cmd += ["--testset", args.testset]
        subprocess.run(
            report_cmd,
            check=False,
        )

    print(
        json.dumps(
            {
                "out": str(out_path),
                "report": args.report or "",
                "mode": args.mode,
                "endpoint": args.endpoint,
                "base_url": args.base_url,
                "strict_sources": args.strict_sources,
                "concurrency": args.concurrency,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

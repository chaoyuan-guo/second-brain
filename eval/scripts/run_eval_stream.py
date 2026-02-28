#!/usr/bin/env python3
"""Run evaluation by calling chat/OpenCode endpoints and collecting answers.

Usage:
  python eval/scripts/run_eval_stream.py --base-url http://127.0.0.1:9090
  python eval/scripts/run_eval_stream.py --testset eval/testsets/testset.json --out eval/reports/answers.json
  python eval/scripts/run_eval_stream.py --report eval/reports/report.json
  python eval/scripts/run_eval_stream.py --question-ids Q01,Q14
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
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


def _build_request(
    url: str,
    method: str,
    data: bytes | None = None,
    extra_headers: Optional[Dict[str, str]] = None,
) -> urlrequest.Request:
    req = urlrequest.Request(url, data=data, method=method)
    if extra_headers:
        for key, val in extra_headers.items():
            req.add_header(key, val)
    return req


def _request_json(
    url: str,
    method: str,
    payload: Optional[dict],
    timeout: int,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Tuple[int, Any]:
    data = None
    headers = dict(extra_headers or {})
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = _build_request(url=url, method=method, data=data, extra_headers=headers)
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            if not raw:
                return resp.status, {}
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urlerror.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = body
        return exc.code, parsed


def stream_chat(
    url: str,
    payload: dict,
    timeout: int,
    extra_headers: Optional[Dict[str, str]] = None,
    stage_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """旧 NDJSON /chat/stream 协议。"""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = _build_request(url=url, method="POST", data=data, extra_headers=extra_headers)
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/x-ndjson")
    req.add_header("x-stream-format", "ndjson")

    answer_parts: List[str] = []
    tool_events: List[Dict[str, Any]] = []

    if stage_cb:
        stage_cb("connect_start")
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        if stage_cb:
            stage_cb("response_opened")
        saw_first_chunk = False
        while True:
            raw = resp.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            if stage_cb and not saw_first_chunk:
                stage_cb("first_chunk")
                saw_first_chunk = True
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
                tool_event = {
                    "stage": event.get("stage"),
                    "tool_name": event.get("tool_name"),
                    "tool_call_id": event.get("tool_call_id"),
                    "arguments": event.get("arguments"),
                    "latency_ms": event.get("latency_ms"),
                    "error": event.get("error"),
                    "output": event.get("result"),
                }
                if event.get("tool_name") == "query_my_notes" and event.get("stage") == "end":
                    result = event.get("result")
                    if isinstance(result, dict):
                        tool_event["retrieved_sources"] = result.get("sources", [])
                        tool_event["retrieved_chunks"] = result.get("chunks", [])
                tool_events.append(tool_event)
            elif event_type == "done":
                break

    if stage_cb:
        stage_cb("response_done")
    return "".join(answer_parts).strip(), tool_events


def stream_opencode(
    base_url: str,
    query: str,
    timeout: int,
    session_path: str,
    extra_headers: Optional[Dict[str, str]] = None,
    stage_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """OpenCode 会话协议：POST /session + POST /session/:id/prompt_async + GET /event。"""
    headers = dict(extra_headers or {})

    if stage_cb:
        stage_cb("session_create_start")
    session_status, session_payload = _request_json(
        url=base_url.rstrip("/") + "/session",
        method="POST",
        payload={"path": session_path},
        timeout=timeout,
        extra_headers=headers,
    )
    if session_status // 100 != 2 or not isinstance(session_payload, dict):
        raise RuntimeError(f"create session failed: status={session_status} payload={session_payload}")

    session_id = str(session_payload.get("id") or "").strip()
    if not session_id:
        raise RuntimeError(f"create session missing id: payload={session_payload}")
    if stage_cb:
        stage_cb("session_created")

    event_req = _build_request(
        url=base_url.rstrip("/") + "/event",
        method="GET",
        extra_headers={
            **headers,
            "Accept": "text/event-stream",
        },
    )

    answer_parts: List[str] = []
    tool_events: List[Dict[str, Any]] = []
    tool_status_by_call: Dict[str, str] = {}
    assistant_message_id: Optional[str] = None
    saw_step_finish = False
    saw_first_chunk = False

    def _active_tool_count() -> int:
        active = 0
        for status in tool_status_by_call.values():
            if status in {"pending", "running"}:
                active += 1
        return active

    deadline = time.time() + timeout

    if stage_cb:
        stage_cb("event_connect_start")
    with urlrequest.urlopen(event_req, timeout=timeout) as event_resp:
        if stage_cb:
            stage_cb("event_opened")

        prompt_status, prompt_payload = _request_json(
            url=base_url.rstrip("/") + f"/session/{session_id}/prompt_async",
            method="POST",
            payload={"parts": [{"type": "text", "text": query}]},
            timeout=timeout,
            extra_headers=headers,
        )
        if prompt_status not in {200, 202, 204}:
            raise RuntimeError(
                f"prompt_async failed: status={prompt_status} payload={prompt_payload}"
            )
        if stage_cb:
            stage_cb("prompt_sent")

        while True:
            if time.time() > deadline:
                raise TimeoutError(
                    f"timeout waiting OpenCode response (session={session_id})"
                )

            raw = event_resp.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line or line.startswith(":") or not line.startswith("data:"):
                continue

            payload_raw = line[5:].strip()
            if not payload_raw:
                continue

            try:
                event = json.loads(payload_raw)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type")
            if event_type == "session.error":
                raise RuntimeError(f"session.error: {event}")
            if event_type != "message.part.updated":
                continue

            properties = event.get("properties") or {}
            part = properties.get("part") or event.get("part") or {}
            if not isinstance(part, dict):
                continue

            if part.get("sessionID") != session_id:
                continue

            if stage_cb and not saw_first_chunk:
                stage_cb("first_chunk")
                saw_first_chunk = True

            part_type = part.get("type")
            part_message_id = part.get("messageID")

            if part_type == "step-start":
                if not assistant_message_id and isinstance(part_message_id, str) and part_message_id:
                    assistant_message_id = part_message_id
                continue

            if assistant_message_id and part_message_id and part_message_id != assistant_message_id:
                continue

            if part_type == "text":
                delta = properties.get("delta")
                if isinstance(delta, str) and delta:
                    answer_parts.append(delta)
                    continue
                text = part.get("text")
                if isinstance(text, str) and text:
                    if answer_parts:
                        answer_parts[-1] = text
                    else:
                        answer_parts.append(text)
                continue

            if part_type == "tool":
                tool_name = str(part.get("tool") or "tool")
                call_id = str(part.get("callID") or f"{tool_name}-{len(tool_status_by_call)}")
                state = part.get("state")
                if not isinstance(state, dict):
                    continue
                status = str(state.get("status") or "")
                if not status:
                    continue

                prev_status = tool_status_by_call.get(call_id)
                tool_status_by_call[call_id] = status

                if status in {"pending", "running"}:
                    if prev_status not in {"pending", "running"}:
                        tool_events.append(
                            {
                                "stage": "start",
                                "tool_name": tool_name,
                                "tool_call_id": call_id,
                                "arguments": state.get("input"),
                                "error": None,
                            }
                        )
                    continue

                if status == "completed":
                    tool_event: Dict[str, Any] = {
                        "stage": "end",
                        "tool_name": tool_name,
                        "tool_call_id": call_id,
                        "arguments": state.get("input"),
                        "output": state.get("output"),
                        "metadata": state.get("metadata"),
                        "error": None,
                    }
                    metadata = state.get("metadata")
                    if isinstance(metadata, dict):
                        tool_event["source_refs"] = metadata.get("source_refs", [])
                    if tool_name == "query_my_notes":
                        output = state.get("output")
                        if isinstance(output, dict):
                            tool_event["retrieved_sources"] = output.get("sources", [])
                            tool_event["retrieved_chunks"] = output.get("chunks", [])
                    tool_events.append(tool_event)
                    continue

                if status == "error":
                    tool_events.append(
                        {
                            "stage": "error",
                            "tool_name": tool_name,
                            "tool_call_id": call_id,
                            "arguments": state.get("input"),
                            "error": state.get("error"),
                        }
                    )
                    continue

            if part_type == "step-finish":
                saw_step_finish = True
                if _active_tool_count() == 0:
                    break

    if stage_cb:
        stage_cb("response_done")
    return "".join(answer_parts).strip(), tool_events


def chat_once(
    url: str,
    payload: dict,
    timeout: int,
    extra_headers: Optional[Dict[str, str]] = None,
    stage_cb: Optional[Callable[[str], None]] = None,
) -> str:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = _build_request(url=url, method="POST", data=data, extra_headers=extra_headers)
    req.add_header("Content-Type", "application/json")

    if stage_cb:
        stage_cb("connect_start")
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        if stage_cb:
            stage_cb("response_opened")
        raw = resp.read().decode("utf-8", errors="ignore")
    if stage_cb:
        stage_cb("response_done")
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
    concurrency: int,
    limit: Optional[int] = None,
    stage_log: bool = True,
    session_path: str = "/app",
) -> Tuple[Dict[str, str], Dict[str, List[Dict[str, Any]]]]:
    """运行评估，返回答案和工具追踪。"""
    answers: Dict[str, str] = {}
    tool_traces: Dict[str, List[Dict[str, Any]]] = {}
    url = base_url.rstrip("/") + endpoint

    def _run_single(q: dict) -> Tuple[str, str, List[Dict[str, Any]]]:
        qid = q["id"]
        query = q["query"]
        payload = {"user_message": query}
        request_headers = dict(headers)
        tool_events: List[Dict[str, Any]] = []
        stage = "init"

        def _stage_cb(next_stage: str) -> None:
            nonlocal stage
            stage = next_stage
            if stage_log:
                print(f"[stage] {qid} {next_stage}", file=sys.stderr)

        try:
            if mode == "chat":
                if stage_log:
                    print(f"[stage] {qid} request_start", file=sys.stderr)
                answer = chat_once(
                    url,
                    payload,
                    timeout,
                    request_headers,
                    stage_cb=_stage_cb,
                )
            elif mode == "stream":
                if stage_log:
                    print(f"[stage] {qid} request_start", file=sys.stderr)
                answer, tool_events = stream_chat(
                    url,
                    payload,
                    timeout,
                    request_headers,
                    stage_cb=_stage_cb,
                )
            else:
                if stage_log:
                    print(f"[stage] {qid} request_start", file=sys.stderr)
                answer, tool_events = stream_opencode(
                    base_url=base_url,
                    query=query,
                    timeout=timeout,
                    session_path=session_path,
                    extra_headers=request_headers,
                    stage_cb=_stage_cb,
                )
        except (TimeoutError, socket.timeout):
            if stage_log:
                print(f"[stage] {qid} timeout stage={stage}", file=sys.stderr)
            answer = "[timeout]"
        except urlerror.HTTPError as exc:
            if stage_log:
                print(f"[stage] {qid} http_error stage={stage}", file=sys.stderr)
            answer = f"[http_error] {exc.code} {exc.reason}"
        except urlerror.URLError as exc:
            if stage_log:
                print(f"[stage] {qid} url_error stage={stage}", file=sys.stderr)
            answer = f"[url_error] {exc.reason}"
        except RuntimeError as exc:
            if stage_log:
                print(f"[stage] {qid} runtime_error stage={stage}", file=sys.stderr)
            answer = f"[runtime_error] {exc}"
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

    print(f"开始评估: 共 {total} 题，并发数 {worker_count}，模式 {mode}")

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
    parser.add_argument("--base-url", default="http://127.0.0.1:9090")
    parser.add_argument("--endpoint", default="/chat/stream")
    parser.add_argument("--mode", choices=["opencode", "stream", "chat"], default="opencode")
    parser.add_argument("--session-path", default="/app", help="OpenCode session path")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--pause", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--question-ids",
        help="Comma-separated question ID prefixes to run, e.g. 'Q01,Q14'. "
        "Matches by prefix so 'Q01' matches 'Q01_coin_change_error_pattern'.",
    )
    parser.add_argument(
        "-H",
        "--header",
        action="append",
        default=[],
        help="Extra header, e.g. 'Authorization: Bearer x'",
    )
    parser.add_argument(
        "--no-stage-log",
        action="store_false",
        dest="stage_log",
        help="Disable stage markers",
    )
    parser.set_defaults(stage_log=True)
    parser.add_argument("--out", default="eval/reports/answers.json")
    parser.add_argument("--report", help="Optional report JSON path; runs grade_by_llm.py after answering")
    args = parser.parse_args()

    questions = load_testset(Path(args.testset))
    headers = parse_headers(args.header)

    if args.question_ids:
        prefixes = tuple(p.strip() for p in args.question_ids.split(",") if p.strip())
        before = len(questions)
        questions = [q for q in questions if q["id"].startswith(prefixes)]
        print(f"按 ID 过滤: {before} → {len(questions)} 题 (匹配: {', '.join(prefixes)})")
        if not questions:
            print("⚠ 没有匹配的题目，请检查 --question-ids 参数", file=sys.stderr)
            sys.exit(1)

    answers, tool_traces = run_eval(
        questions,
        base_url=args.base_url,
        endpoint=args.endpoint,
        timeout=args.timeout,
        pause=args.pause,
        mode=args.mode,
        headers=headers,
        concurrency=args.concurrency,
        limit=args.limit,
        stage_log=args.stage_log,
        session_path=args.session_path,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ 答案已保存至: {out_path}")

    tool_traces_path = out_path.parent / (out_path.stem + "_tool_traces.json")
    tool_traces_path.write_text(json.dumps(tool_traces, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ 工具追踪已保存至: {tool_traces_path}")

    os.sync()

    try:
        with open(out_path, "r", encoding="utf-8") as f:
            json.load(f)
        with open(tool_traces_path, "r", encoding="utf-8") as f:
            loaded_traces = json.load(f)
        print(f"✓ 文件完整性验证通过 (tool_traces: {len(loaded_traces)} 条)")
    except Exception as exc:
        print(f"⚠ 文件完整性验证失败: {exc}", file=sys.stderr)

    if args.report:
        print("\n开始评分...")
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        report_cmd = [
            "python",
            "eval/scripts/grade_by_llm.py",
            "--testset",
            args.testset,
            "--answers",
            str(out_path),
            "--tool-traces",
            str(tool_traces_path),
            "--output",
            str(report_path),
        ]

        env = os.environ.copy()
        project_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run(
            report_cmd,
            check=False,
            env=env,
        )

    print(
        json.dumps(
            {
                "out": str(out_path),
                "report": args.report or "",
                "mode": args.mode,
                "endpoint": args.endpoint,
                "base_url": args.base_url,
                "concurrency": args.concurrency,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

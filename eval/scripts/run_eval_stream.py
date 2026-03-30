#!/usr/bin/env python3
"""Run evaluation against the OpenCode session/event API and collect answers.

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
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import urlsplit

_DIRECT_OPENER = urlrequest.build_opener(urlrequest.ProxyHandler({}))


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


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def _is_failure_answer(answer: str) -> bool:
    prefixes = ("[timeout]", "[http_error]", "[url_error]", "[runtime_error]")
    return answer.strip().startswith(prefixes)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


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


def _candidate_base_urls(base_url: str) -> List[str]:
    raw = base_url.rstrip("/")
    parsed = urlsplit(raw)
    host = parsed.hostname or ""
    scheme = parsed.scheme or "http"
    port = f":{parsed.port}" if parsed.port else ""

    ordered: List[str] = []
    if host in {"localhost", "127.0.0.1"}:
        ordered.extend(
            [
                f"{scheme}://[::1]{port}",
                f"{scheme}://localhost{port}",
                f"{scheme}://127.0.0.1{port}",
            ]
        )
    elif host == "::1":
        ordered.extend(
            [
                f"{scheme}://[::1]{port}",
                f"{scheme}://localhost{port}",
                f"{scheme}://127.0.0.1{port}",
            ]
        )
    else:
        ordered.append(raw)

    deduped: List[str] = []
    for item in ordered:
        normalized = item.rstrip("/")
        if normalized not in deduped:
            deduped.append(normalized)
    return deduped


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
        with _DIRECT_OPENER.open(req, timeout=timeout) as resp:
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


def probe_base_url(
    base_url: str,
    timeout: int = 5,
) -> Tuple[bool, str]:
    """Probe whether the endpoint is reachable via HTTP.

    401/403/404/405 all count as "reachable": they indicate the server responded.
    This avoids treating "auth required" or "wrong path" as "service down".
    """
    candidates = [base_url.rstrip("/"), base_url.rstrip("/") + "/session"]
    tried: List[str] = []
    for candidate in candidates:
        if candidate in tried:
            continue
        tried.append(candidate)
        req = _build_request(url=candidate, method="GET")
        try:
            with _DIRECT_OPENER.open(req, timeout=timeout) as resp:
                status = getattr(resp, "status", 200)
                return True, f"{candidate} -> HTTP {status}"
        except urlerror.HTTPError as exc:
            return True, f"{candidate} -> HTTP {exc.code}"
        except urlerror.URLError as exc:
            reason = exc.reason
            if isinstance(reason, socket.timeout):
                return False, f"{candidate} -> timeout"
            continue
        except Exception:
            continue
    return False, f"{base_url.rstrip('/')} -> no HTTP response"


def stream_opencode(
    base_url: str,
    query: str,
    timeout: int,
    session_path: str,
    extra_headers: Optional[Dict[str, str]] = None,
    stage_cb: Optional[Callable[[str], None]] = None,
    trace: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """OpenCode 会话协议：POST /session + POST /session/:id/prompt_async + GET /event。"""
    headers = dict(extra_headers or {})

    if stage_cb:
        stage_cb("session_create_start")

    resolved_base_url = ""
    last_error = ""
    for candidate in _candidate_base_urls(base_url):
        session_status, session_payload = _request_json(
            url=candidate + "/session",
            method="POST",
            payload={"path": session_path},
            timeout=timeout,
            extra_headers=headers,
        )
        if session_status // 100 == 2 and isinstance(session_payload, dict):
            resolved_base_url = candidate
            break
        last_error = f"candidate={candidate} status={session_status} payload={session_payload}"

    if not resolved_base_url:
        raise RuntimeError(f"create session failed: {last_error}")

    session_id = str(session_payload.get("id") or "").strip()
    if not session_id:
        raise RuntimeError(f"create session missing id: payload={session_payload}")

    if trace is not None:
        trace["resolved_base_url"] = resolved_base_url
        trace["session_id"] = session_id
        trace.setdefault("raw_events", [])
        trace.setdefault("timeline", [])

    if stage_cb:
        stage_cb("session_created")

    event_req = _build_request(
        url=resolved_base_url + "/event",
        method="GET",
        extra_headers={
            **headers,
            "Accept": "text/event-stream",
        },
    )

    answer_parts: List[str] = []
    tool_events: List[Dict[str, Any]] = []
    tool_status_by_call: Dict[str, str] = {}
    message_roles: Dict[str, str] = {}
    known_parts: Dict[str, Dict[str, str]] = {}
    current_step_message_id: Optional[str] = None
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
    with _DIRECT_OPENER.open(event_req, timeout=timeout) as event_resp:
        if stage_cb:
            stage_cb("event_opened")

        prompt_status, prompt_payload = _request_json(
            url=resolved_base_url + f"/session/{session_id}/prompt_async",
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
            if trace is not None:
                trace["raw_events"].append(
                    {
                        "ts": datetime.utcnow().isoformat() + "Z",
                        "event_type": event_type,
                        "payload": event,
                    }
                )

            if event_type == "session.error":
                if trace is not None:
                    trace["timeline"].append(
                        {
                            "ts": datetime.utcnow().isoformat() + "Z",
                            "kind": "session.error",
                            "payload": event.get("properties", {}),
                        }
                    )
                raise RuntimeError(f"session.error: {event}")

            if event_type == "message.updated":
                info = (event.get("properties") or {}).get("info") or {}
                if isinstance(info, dict) and info.get("sessionID") == session_id:
                    message_id = info.get("id")
                    role = info.get("role")
                    if isinstance(message_id, str) and isinstance(role, str):
                        message_roles[message_id] = role
                continue

            if event_type == "message.part.delta":
                properties = event.get("properties") or {}
                if not isinstance(properties, dict):
                    continue

                part_id = properties.get("partID")
                field = properties.get("field")
                delta = properties.get("delta")
                meta = known_parts.get(str(part_id), {}) if isinstance(part_id, str) else {}
                part_message_id = properties.get("messageID") or meta.get("messageID")
                part_session_id = properties.get("sessionID") or meta.get("sessionID")
                part_type = meta.get("type")

                if part_session_id != session_id:
                    continue
                if field != "text" or not isinstance(delta, str) or not delta:
                    continue
                if part_type != "text":
                    continue
                if not isinstance(part_message_id, str):
                    continue
                if message_roles.get(part_message_id) != "assistant":
                    continue
                if current_step_message_id and part_message_id != current_step_message_id:
                    continue

                if stage_cb and not saw_first_chunk:
                    stage_cb("first_chunk")
                    saw_first_chunk = True
                answer_parts.append(delta)
                if trace is not None:
                    trace["timeline"].append(
                        {
                            "ts": datetime.utcnow().isoformat() + "Z",
                            "kind": "part.delta",
                            "part_type": "text",
                            "message_id": part_message_id,
                            "delta_len": len(delta),
                        }
                    )
                continue

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
            part_id = part.get("id")
            if (
                isinstance(part_id, str)
                and isinstance(part_type, str)
                and isinstance(part_message_id, str)
            ):
                known_parts[part_id] = {
                    "type": part_type,
                    "messageID": part_message_id,
                    "sessionID": session_id,
                }
            if trace is not None:
                trace["timeline"].append(
                    {
                        "ts": datetime.utcnow().isoformat() + "Z",
                        "kind": "part.updated",
                        "part_type": part_type,
                        "message_id": part_message_id,
                    }
                )

            if part_type == "step-start":
                if (
                    isinstance(part_message_id, str)
                    and part_message_id
                    and message_roles.get(part_message_id) == "assistant"
                ):
                    current_step_message_id = part_message_id
                continue

            if not isinstance(part_message_id, str):
                continue
            if message_roles.get(part_message_id) != "assistant":
                continue
            if current_step_message_id and part_message_id != current_step_message_id:
                continue

            if part_type == "text":
                text = part.get("text")
                if isinstance(text, str):
                    if answer_parts:
                        answer_parts = [text]
                    elif text:
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
                    if trace is not None:
                        trace["timeline"].append(
                            {
                                "ts": datetime.utcnow().isoformat() + "Z",
                                "kind": "tool",
                                "status": status,
                                "tool_name": tool_name,
                                "tool_call_id": call_id,
                            }
                        )
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
                    if trace is not None:
                        trace["timeline"].append(
                            {
                                "ts": datetime.utcnow().isoformat() + "Z",
                                "kind": "tool",
                                "status": status,
                                "tool_name": tool_name,
                                "tool_call_id": call_id,
                            }
                        )
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
                    tool_events.append(tool_event)
                    continue

                if status == "error":
                    if trace is not None:
                        trace["timeline"].append(
                            {
                                "ts": datetime.utcnow().isoformat() + "Z",
                                "kind": "tool",
                                "status": status,
                                "tool_name": tool_name,
                                "tool_call_id": call_id,
                                "error": state.get("error"),
                            }
                        )
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
                reason = part.get("reason")
                if trace is not None:
                    trace["timeline"].append(
                        {
                            "ts": datetime.utcnow().isoformat() + "Z",
                            "kind": "step-finish",
                            "message_id": part_message_id,
                            "reason": reason,
                        }
                    )
                if reason != "tool-calls" and _active_tool_count() == 0:
                    break

    if stage_cb:
        stage_cb("response_done")
    return "".join(answer_parts).strip(), tool_events


def run_eval(
    questions: Iterable[dict],
    base_url: str,
    timeout: int,
    pause: float,
    headers: Dict[str, str],
    concurrency: int,
    limit: Optional[int] = None,
    stage_log: bool = True,
    session_path: str = "/app",
    trace_dir: Optional[Path] = None,
    trace_run_id: str = "",
) -> Tuple[Dict[str, str], Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    """运行评估，返回答案和工具追踪。"""
    answers: Dict[str, str] = {}
    tool_traces: Dict[str, List[Dict[str, Any]]] = {}
    trace_summaries: Dict[str, Dict[str, Any]] = {}

    def _run_single(q: dict) -> Tuple[str, str, List[Dict[str, Any]], Dict[str, Any]]:
        qid = q["id"]
        query = q["query"]
        request_headers = dict(headers)
        eval_trace_id = request_headers.get("x-request-id") or f"eval-{_safe_name(qid)}-{uuid.uuid4().hex[:8]}"
        request_headers["x-request-id"] = eval_trace_id
        request_headers.setdefault("x-eval-question-id", qid)
        tool_events: List[Dict[str, Any]] = []
        stage = "init"
        case_trace: Dict[str, Any] = {
            "trace_id": eval_trace_id,
            "question_id": qid,
            "query": query,
            "mode": "opencode",
            "started_at": datetime.utcnow().isoformat() + "Z",
            "stages": [],
            "raw_events": [],
            "timeline": [],
        }

        def _stage_cb(next_stage: str) -> None:
            nonlocal stage
            stage = next_stage
            case_trace["stages"].append(
                {
                    "ts": datetime.utcnow().isoformat() + "Z",
                    "stage": next_stage,
                }
            )
            if stage_log:
                print(f"[stage] {qid} {next_stage}", file=sys.stderr)

        try:
            if stage_log:
                print(f"[stage] {qid} request_start", file=sys.stderr)
            answer, tool_events = stream_opencode(
                base_url=base_url,
                query=query,
                timeout=timeout,
                session_path=session_path,
                extra_headers=request_headers,
                stage_cb=_stage_cb,
                trace=case_trace,
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
            case_trace["error"] = str(exc)
        except Exception as exc:  # noqa: BLE001
            if stage_log:
                print(f"[stage] {qid} unexpected_error stage={stage}", file=sys.stderr)
            answer = f"[runtime_error] unexpected: {exc}"
            case_trace["error"] = str(exc)

        case_trace["finished_at"] = datetime.utcnow().isoformat() + "Z"
        case_trace["answer"] = answer
        case_trace["tool_event_count"] = len(tool_events)
        case_trace["failed"] = _is_failure_answer(answer)
        case_trace["error_stage"] = stage

        if trace_dir is not None and trace_run_id:
            case_dir = trace_dir / trace_run_id / _safe_name(qid)
            _write_jsonl(case_dir / "events.jsonl", case_trace.get("raw_events", []))
            _write_json(case_dir / "timeline.json", case_trace.get("timeline", []))
            summary = {
                "trace_id": eval_trace_id,
                "question_id": qid,
                "failed": case_trace["failed"],
                "error_stage": stage,
                "stage_count": len(case_trace.get("stages", [])),
                "tool_event_count": len(tool_events),
                "session_id": case_trace.get("session_id", ""),
                "resolved_base_url": case_trace.get("resolved_base_url", ""),
                "started_at": case_trace.get("started_at"),
                "finished_at": case_trace.get("finished_at"),
            }
            _write_json(case_dir / "summary.json", summary)
            if case_trace["failed"]:
                failure_bundle = {
                    "trace_id": eval_trace_id,
                    "question_id": qid,
                    "query": query,
                    "answer": answer,
                    "error_stage": stage,
                    "error": case_trace.get("error", ""),
                    "last_stages": case_trace.get("stages", [])[-20:],
                    "last_timeline": case_trace.get("timeline", [])[-50:],
                    "last_events": case_trace.get("raw_events", [])[-50:],
                    "tool_events": tool_events[-20:],
                }
                _write_json(case_dir / "failure_bundle.json", failure_bundle)

        if pause > 0:
            time.sleep(pause)
        return qid, answer, tool_events, case_trace

    queued: List[dict] = []
    for idx, q in enumerate(questions, start=1):
        if limit and idx > limit:
            break
        queued.append(q)

    worker_count = max(1, concurrency)
    total = len(queued)
    completed = 0

    print(f"开始评估: 共 {total} 题，并发数 {worker_count}，base_url {base_url}")

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_run_single, q) for q in queued]
        for future in as_completed(futures):
            qid, answer, tool_events, case_trace = future.result()
            answers[qid] = answer
            tool_traces[qid] = tool_events
            trace_summaries[qid] = {
                "trace_id": case_trace.get("trace_id"),
                "failed": case_trace.get("failed", False),
                "error_stage": case_trace.get("error_stage"),
                "tool_event_count": len(tool_events),
            }
            completed += 1
            print(f"  [{completed}/{total}] {qid} 完成")

    print(f"✓ 评估完成，已生成 {len(answers)} 个答案")
    return answers, tool_traces, trace_summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation against the OpenCode API")
    parser.add_argument("--testset", default="eval/testsets/testset.json")
    parser.add_argument("--base-url", default="http://127.0.0.1:9090")
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
    parser.add_argument(
        "--trace-dir",
        default="eval/reports/traces",
        help="Directory for per-case raw SSE traces/timelines/failure bundles",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Disable writing per-case traces",
    )
    parser.add_argument("--report", help="Optional report JSON path; runs grade_by_llm.py after answering")
    args = parser.parse_args()

    reachable, probe_detail = probe_base_url(args.base_url)
    if reachable:
        print(
            f"✓ OpenCode 连通性探测通过: {probe_detail} "
            f"(注意：401/403/404/405 说明服务有响应，不等于 down)"
        )
    else:
        print(f"✗ OpenCode 连通性探测失败: {probe_detail}", file=sys.stderr)
        sys.exit(1)

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

    trace_run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    effective_trace_dir = None if args.no_trace else Path(args.trace_dir)

    answers, tool_traces, trace_summaries = run_eval(
        questions,
        base_url=args.base_url,
        timeout=args.timeout,
        pause=args.pause,
        headers=headers,
        concurrency=args.concurrency,
        limit=args.limit,
        stage_log=args.stage_log,
        session_path=args.session_path,
        trace_dir=effective_trace_dir,
        trace_run_id=trace_run_id,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ 答案已保存至: {out_path}")

    tool_traces_path = out_path.parent / (out_path.stem + "_tool_traces.json")
    tool_traces_path.write_text(json.dumps(tool_traces, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ 工具追踪已保存至: {tool_traces_path}")

    if effective_trace_dir is not None:
        trace_index = {
            "run_id": trace_run_id,
            "trace_dir": str((effective_trace_dir / trace_run_id).resolve()),
            "cases": trace_summaries,
        }
        trace_index_path = out_path.parent / (out_path.stem + "_trace_index.json")
        trace_index_path.write_text(json.dumps(trace_index, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✓ 追踪索引已保存至: {trace_index_path}")

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
                "mode": "opencode",
                "base_url": args.base_url,
                "concurrency": args.concurrency,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

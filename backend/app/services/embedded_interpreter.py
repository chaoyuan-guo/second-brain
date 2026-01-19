"""进程内 Python 代码解释器。

用于容器部署场景：
- 避免启动独立的 MCP SSE 服务（满足单端口/单进程约束）。
- 避免每次调用都拉起 mcp-python-interpreter 子进程（提升高频调用性能）。

安全策略（偏保守）：
- 默认只允许导入 Python 标准库；阻止网络/子进程/多进程等高风险模块。
- 文件读写限制在指定 workdir 下（通过替换 builtins.open）。

稳定性策略：
- 通过 sys.settrace 进行软超时控制（可打断纯 Python 无限循环）。
- 由于 stdout/stderr 重定向是全局的，解释器执行采用全局锁串行化。
"""

from __future__ import annotations

import builtins
import contextlib
import importlib.util
import io
import os
import sys
import sysconfig
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional


class EmbeddedInterpreterError(RuntimeError):
    """解释器内部错误。"""


_EXEC_LOCK = threading.Lock()


def _resolve_within_workdir(
    workdir: Path,
    raw_path: str | os.PathLike[str],
    allow_system_access: bool,
) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = workdir / path

    resolved = path.expanduser().resolve()
    if allow_system_access:
        return resolved

    try:
        resolved.relative_to(workdir.resolve())
    except ValueError as exc:
        raise PermissionError(
            f"Access denied: path must be under workdir ({workdir})."
        ) from exc
    return resolved


def _make_restricted_open(
    workdir: Path,
    allow_system_access: bool,
) -> Callable[..., Any]:
    original_open = builtins.open

    def _open(file: Any, *args: Any, **kwargs: Any) -> Any:
        # 兼容 open(fd, ...) 场景（整数文件描述符不做路径限制）。
        if isinstance(file, int):
            return original_open(file, *args, **kwargs)
        resolved = _resolve_within_workdir(workdir, file, allow_system_access)
        return original_open(resolved, *args, **kwargs)

    return _open


_DENY_IMPORT_PREFIXES = {
    "subprocess",
    "multiprocessing",
    "socket",
    "ssl",
    "http",
    "urllib",
    "asyncio",
    "threading",
    "ctypes",
}


def _is_denied_module(module_name: str) -> bool:
    normalized = module_name.split(".")[0]
    return normalized in _DENY_IMPORT_PREFIXES


def _is_stdlib_module(module_name: str) -> bool:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return False
    if spec.origin in (None, "built-in", "frozen"):
        return True

    origin = Path(spec.origin).resolve()
    stdlib_root = Path(sysconfig.get_paths()["stdlib"]).resolve()
    purelib_root = Path(sysconfig.get_paths()["purelib"]).resolve()
    platlib_root = Path(sysconfig.get_paths()["platlib"]).resolve()

    # 许多 Python 发行版会把 site-packages 放在 stdlib 目录下的子目录中，
    # 这里需要显式排除第三方包安装目录。
    for third_party_root in (purelib_root, platlib_root):
        try:
            origin.relative_to(third_party_root)
        except ValueError:
            continue
        return False
    try:
        origin.relative_to(stdlib_root)
        return True
    except ValueError:
        return False


def _make_restricted_import() -> Callable[..., Any]:
    original_import = builtins.__import__

    def _restricted_import(
        name: str,
        globals: Optional[Dict[str, Any]] = None,
        locals: Optional[Dict[str, Any]] = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if level != 0:
            raise ImportError("Relative imports are disabled in the embedded interpreter")
        if not name:
            raise ImportError("Empty module name")
        if _is_denied_module(name):
            raise ImportError(f"Module '{name}' is disabled in the embedded interpreter")
        if not _is_stdlib_module(name):
            raise ImportError(
                f"Only Python stdlib imports are allowed (blocked: '{name}')"
            )
        return original_import(name, globals, locals, fromlist, level)

    return _restricted_import


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    status: int
    elapsed_ms: float


class InterpreterSession:
    def __init__(
        self,
        *,
        workdir: Path,
        allow_system_access: bool,
    ) -> None:
        self.workdir = workdir
        self.allow_system_access = allow_system_access
        self.history: list[dict[str, Any]] = []
        self.locals: dict[str, Any] = {
            "__name__": "__main__",
            "__doc__": None,
            "__package__": None,
        }

        safe_builtins = dict(builtins.__dict__)
        safe_builtins["open"] = _make_restricted_open(
            workdir, allow_system_access=allow_system_access
        )
        safe_builtins["__import__"] = _make_restricted_import()
        safe_builtins["input"] = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("input() is disabled in the embedded interpreter")
        )

        self.locals["__builtins__"] = safe_builtins

    def execute(self, code: str, *, timeout_seconds: int) -> ExecutionResult:
        started_at = time.perf_counter()
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        def _make_trace() -> Callable[..., Any]:
            deadline = time.monotonic() + max(timeout_seconds, 1)

            def _trace(frame, event, arg):
                # 单行 while True: pass 这类循环不会频繁触发 line 事件，
                # 需要开启 opcode 级别事件才能可靠打断。
                if event == "call":
                    frame.f_trace_opcodes = True
                if event in {"line", "opcode"} and time.monotonic() > deadline:
                    raise TimeoutError(
                        f"Execution timed out after {timeout_seconds} seconds"
                    )
                return _trace

            return _trace

        status = 0
        old_cwd = os.getcwd()
        old_trace = sys.gettrace()

        try:
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(
                stderr_capture
            ):
                os.chdir(self.workdir)
                sys.settrace(_make_trace())
                try:
                    try:
                        compiled = compile(code, "<embedded>", "eval")
                    except SyntaxError:
                        compiled = compile(code, "<embedded>", "exec")
                        exec(compiled, self.locals)
                    else:
                        result_value = eval(compiled, self.locals)
                        if result_value is not None:
                            print(repr(result_value))
                except TimeoutError:
                    status = -1
                    traceback.print_exc()
                except Exception:
                    status = 1
                    traceback.print_exc()
        finally:
            sys.settrace(old_trace)
            os.chdir(old_cwd)

        elapsed_ms = (time.perf_counter() - started_at) * 1000
        stdout_text = stdout_capture.getvalue()
        stderr_text = stderr_capture.getvalue()
        self.history.append(
            {
                "code": code,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "status": status,
                "elapsed_ms": elapsed_ms,
            }
        )
        if len(self.history) > 50:
            self.history = self.history[-50:]
        return ExecutionResult(
            stdout=stdout_text,
            stderr=stderr_text,
            status=status,
            elapsed_ms=elapsed_ms,
        )


class EmbeddedPythonInterpreter:
    def __init__(self) -> None:
        self._sessions: dict[str, InterpreterSession] = {}

    def _get_session(
        self,
        session_id: str,
        *,
        workdir: Path,
        allow_system_access: bool,
    ) -> InterpreterSession:
        session = self._sessions.get(session_id)
        if session is None:
            session = InterpreterSession(
                workdir=workdir, allow_system_access=allow_system_access
            )
            self._sessions[session_id] = session
        return session

    def run(
        self,
        *,
        code: str,
        session_id: str,
        execution_mode: str,
        timeout: int,
        workdir: Path,
        allow_system_access: bool = False,
    ) -> dict[str, Any]:
        if execution_mode and execution_mode != "inline":
            return {
                "ok": True,
                "content": f"Unsupported execution_mode: {execution_mode}. Only 'inline' is available.",
                "raw": {
                    "status": 1,
                    "execution_mode": execution_mode,
                },
            }

        if not isinstance(timeout, int):
            timeout = 300
        timeout = max(timeout, 1)

        workdir.mkdir(parents=True, exist_ok=True)

        session = self._get_session(
            session_id or "default",
            workdir=workdir,
            allow_system_access=allow_system_access,
        )

        with _EXEC_LOCK:
            try:
                result = session.execute(code, timeout_seconds=timeout)
            except Exception as exc:
                return {
                    "ok": False,
                    "error": f"Embedded interpreter crashed: {exc}",
                }

        content = result.stdout
        if result.status != 0:
            content = result.stderr or result.stdout
            if result.stderr and result.stdout:
                content = f"{result.stderr}\n\n{result.stdout}"

        return {
            "ok": True,
            "content": content.strip(),
            "raw": {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "status": result.status,
                "elapsed_ms": result.elapsed_ms,
                "session_id": session_id,
                "execution_mode": "inline",
                "timeout": timeout,
                "workdir": str(workdir),
                "history_size": len(session.history),
            },
        }


embedded_python_interpreter = EmbeddedPythonInterpreter()

import os
from pathlib import Path

import pytest


def test_embedded_interpreter_blocks_non_stdlib_import(monkeypatch, tmp_path: Path) -> None:
    from backend.app.services.embedded_interpreter import embedded_python_interpreter

    result = embedded_python_interpreter.run(
        code="import numpy\nprint('x')",
        session_id="t1",
        execution_mode="inline",
        timeout=2,
        workdir=tmp_path,
        allow_system_access=False,
    )
    assert result["ok"] is True
    assert "Only Python stdlib imports are allowed" in result["content"]


def test_embedded_interpreter_restricts_file_access(tmp_path: Path) -> None:
    from backend.app.services.embedded_interpreter import embedded_python_interpreter

    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    result = embedded_python_interpreter.run(
        code=f"open(r'{outside}', 'r').read()",
        session_id="t2",
        execution_mode="inline",
        timeout=2,
        workdir=tmp_path,
        allow_system_access=False,
    )
    assert result["ok"] is True
    assert "Access denied" in result["content"]


def test_embedded_interpreter_timeout(tmp_path: Path) -> None:
    from backend.app.services.embedded_interpreter import embedded_python_interpreter

    # 纯 Python 无限循环应被软超时打断。
    result = embedded_python_interpreter.run(
        code="while True: pass",
        session_id="t3",
        execution_mode="inline",
        timeout=1,
        workdir=tmp_path,
        allow_system_access=False,
    )
    assert result["ok"] is True
    assert "Execution timed out" in result["content"]


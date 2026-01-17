import subprocess


def test_call_mcp_python_interpreter_aligns_process_timeout(monkeypatch) -> None:
    from backend.app.services import tools

    def fake_ensure_ready() -> None:
        return None

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["timeout"] = kwargs.get("timeout")
        return subprocess.CompletedProcess(cmd, 0, stdout=b'{"ok": true}', stderr=b"")

    monkeypatch.setattr(tools, "ensure_mcp_ready", fake_ensure_ready)
    monkeypatch.setattr(subprocess, "run", fake_run)

    tools.call_mcp_python_interpreter({"code": "print(1)", "timeout": 10})

    cmd = captured["cmd"]
    assert "--process-timeout" in cmd
    idx = cmd.index("--process-timeout")
    assert int(cmd[idx + 1]) >= 420
    assert captured["timeout"] >= int(cmd[idx + 1])


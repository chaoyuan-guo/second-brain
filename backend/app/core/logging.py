"""日志初始化逻辑。"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

from .config import is_truthy, running_in_container, settings


class _ContextFilter(logging.Filter):
    """为日志记录注入缺省的上下文字段，避免 formatter 缺 KeyError。"""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        defaults = {
            "trace_id": "-",
            "turn": "-",
            "attempt": "-",
            "tool_call_id": "-",
            "tool": "-",
            "stream": "-",
        }
        for key, value in defaults.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


def _resolve_log_sinks() -> tuple[bool, bool]:
    """解析日志输出渠道配置（文件/标准输出）。"""

    stdout_env = os.getenv("LOG_TO_STDOUT")
    file_env = os.getenv("LOG_TO_FILE")

    if stdout_env is None:
        stdout_enabled = running_in_container()
    else:
        stdout_enabled = is_truthy(stdout_env)

    if file_env is None:
        file_enabled = not running_in_container()
    else:
        file_enabled = is_truthy(file_env)

    return file_enabled, stdout_enabled


def _configure_logging() -> tuple[logging.Logger, logging.Logger]:
    """初始化主日志与工具输出日志。"""

    context_filter = _ContextFilter()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s "
        "[trace_id=%(trace_id)s turn=%(turn)s attempt=%(attempt)s]: %(message)s"
    )

    file_enabled, stdout_enabled = _resolve_log_sinks()
    file_handler = None
    if file_enabled:
        file_handler = TimedRotatingFileHandler(
            settings.log_path,
            when="midnight",
            backupCount=6,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(context_filter)

    stream_handler = None
    if stdout_enabled:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.addFilter(context_filter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if file_handler is not None and not any(
        isinstance(handler, logging.FileHandler)
        and getattr(handler, "baseFilename", None) == file_handler.baseFilename
        for handler in root_logger.handlers
    ):
        root_logger.addHandler(file_handler)

    if stream_handler is not None and not any(
        isinstance(handler, logging.StreamHandler)
        and getattr(handler, "stream", None) is sys.stdout
        for handler in root_logger.handlers
    ):
        root_logger.addHandler(stream_handler)

    app_logger = logging.getLogger("super-mind-app")

    tool_output_logger = logging.getLogger("super-mind-tool-output")
    tool_output_logger.setLevel(logging.INFO)
    tool_output_logger.propagate = False
    tool_handler = TimedRotatingFileHandler(
        settings.tool_log_path,
        when="midnight",
        backupCount=6,
        encoding="utf-8",
    )
    tool_handler.setFormatter(formatter)
    tool_handler.addFilter(context_filter)
    if not any(
        isinstance(handler, TimedRotatingFileHandler)
        and getattr(handler, "baseFilename", None) == tool_handler.baseFilename
        for handler in tool_output_logger.handlers
    ):
        tool_output_logger.addHandler(tool_handler)

    return app_logger, tool_output_logger


app_logger, tool_output_logger = _configure_logging()


__all__ = ["app_logger", "tool_output_logger"]

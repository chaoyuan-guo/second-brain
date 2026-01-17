"""日志初始化逻辑。"""

from __future__ import annotations

import logging
from logging.handlers import TimedRotatingFileHandler

from .config import settings


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


def _configure_logging() -> tuple[logging.Logger, logging.Logger]:
    """初始化主日志与工具输出日志。"""

    context_filter = _ContextFilter()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s "
        "[trace_id=%(trace_id)s turn=%(turn)s attempt=%(attempt)s]: %(message)s"
    )

    file_handler = logging.FileHandler(settings.log_path)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(context_filter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not any(
        isinstance(handler, logging.FileHandler)
        and getattr(handler, "baseFilename", None) == file_handler.baseFilename
        for handler in root_logger.handlers
    ):
        root_logger.addHandler(file_handler)

    app_logger = logging.getLogger("super-mind-app")

    tool_output_logger = logging.getLogger("super-mind-tool-output")
    tool_output_logger.setLevel(logging.INFO)
    tool_output_logger.propagate = False

    tool_handler = TimedRotatingFileHandler(
        settings.tool_log_path,
        when="midnight",
        backupCount=30,
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

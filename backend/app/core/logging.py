"""日志初始化逻辑。"""

from __future__ import annotations

import logging
from logging.handlers import TimedRotatingFileHandler

from .config import settings


def _configure_logging() -> tuple[logging.Logger, logging.Logger]:
    """初始化主日志与工具输出日志。"""

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    file_handler = logging.FileHandler(settings.log_path)
    file_handler.setFormatter(formatter)

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
    if not any(
        isinstance(handler, TimedRotatingFileHandler)
        and getattr(handler, "baseFilename", None) == tool_handler.baseFilename
        for handler in tool_output_logger.handlers
    ):
        tool_output_logger.addHandler(tool_handler)

    return app_logger, tool_output_logger


app_logger, tool_output_logger = _configure_logging()


__all__ = ["app_logger", "tool_output_logger"]

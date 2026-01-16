"""自定义服务层异常。"""


class ToolExecutionError(Exception):
    """当工具调用失败或返回非法结果时抛出。"""


__all__ = ["ToolExecutionError"]


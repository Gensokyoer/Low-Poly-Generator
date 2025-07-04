from utils.stratergy_pattern.tool_stratergy import ToolStrategy
from PySide6.QtGui import QPixmap, QImage, QImageReader, QWheelEvent, QMouseEvent, QPainter


class ToolManager:
    """工具管理器，负责切换不同工具策略"""

    def __init__(self, viewer):
        self._viewer = viewer
        self._current_tool = None
        self._tools = {}  # 工具名称到策略的映射

    def register_tool(self, name: str, tool: ToolStrategy):
        """注册工具"""
        self._tools[name] = tool

    def activate_tool(self, name: str):
        """激活指定工具"""
        if self._current_tool:
            self._current_tool.deactivate(self._viewer)

        if name in self._tools:
            self._current_tool = self._tools[name]
            self._current_tool.activate(self._viewer)

    def deactivate_all(self):
        """停用所有工具"""
        if self._current_tool:
            self._current_tool.deactivate(self._viewer)
            self._current_tool = None

    def mouse_press(self, event: QMouseEvent):
        if self._current_tool:
            self._current_tool.mouse_press(self._viewer, event)

    def mouse_move(self, event: QMouseEvent):
        if self._current_tool:
            self._current_tool.mouse_move(self._viewer, event)

    def mouse_release(self, event: QMouseEvent):
        if self._current_tool:
            self._current_tool.mouse_release(self._viewer, event)
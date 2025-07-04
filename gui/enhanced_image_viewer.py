from PySide6.QtGui import QMouseEvent

from gui.image_viewer import ImageViewer
from utils.stratergy_pattern.tool_manager import ToolManager
from utils.stratergy_pattern.stratergy.lasso_stratergy import LassoStrategy
from utils.stratergy_pattern.stratergy.rectangle_stratergy import RectangleStrategy
from utils.stratergy_pattern.stratergy.pan_stratergy import PanStrategy
from utils.stratergy_pattern.stratergy.magicwand_stratergy import MagicWandStrategy

class EnhancedImageViewer(ImageViewer):
    """增强版图像查看器，添加工具支持"""

    def __init__(self, parent=None):
        super().__init__(parent)
        # 初始化工具管理
        self._tool_manager = ToolManager(self)

        # 注册默认工具
        self._tool_manager.register_tool("lasso", LassoStrategy())
        self._tool_manager.register_tool("rectangle", RectangleStrategy())
        self._tool_manager.register_tool("pan", PanStrategy())
        # self._tool_manager.register_tool("magic", MagicWandStrategy())


    def set_tool(self, tool_name: str):
        """设置当前工具"""
        self._tool_manager.activate_tool(tool_name)

    def clear_tool(self):
        """清除当前工具"""
        self._tool_manager.deactivate_all()

    def mousePressEvent(self, event: QMouseEvent):
        self._tool_manager.mouse_press(event)
        if not event.isAccepted():
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        self._tool_manager.mouse_move(event)
        if not event.isAccepted():
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._tool_manager.mouse_release(event)
        if not event.isAccepted():
            super().mouseReleaseEvent(event)

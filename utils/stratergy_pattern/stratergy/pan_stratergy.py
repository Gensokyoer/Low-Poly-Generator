from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QMouseEvent
from utils.stratergy_pattern.tool_stratergy import ToolStrategy

class PanStrategy(ToolStrategy):
    """图像拖动策略（左键拖动）"""
    def __init__(self):
        super().__init__()

        # 鼠标拖动相关变量
        self._pan_start = QPoint(0, 0)
        self._panning = False

    def activate(self, viewer):
        viewer.setCursor(Qt.ArrowCursor)  # 设置手掌形光标

    def mouse_press(self, viewer, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._pan_start = event.pos()
            self._panning = True
            viewer.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()

    def mouse_move(self, viewer, event: QMouseEvent):
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()

            # 移动滚动条实现拖动效果
            h_bar = viewer.horizontalScrollBar()
            v_bar = viewer.verticalScrollBar()
            h_bar.setValue(h_bar.value() - delta.x())
            v_bar.setValue(v_bar.value() - delta.y())
            event.accept()

    def mouse_release(self, viewer, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._panning = False
            viewer.setCursor(Qt.ArrowCursor)
            event.accept()

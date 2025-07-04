from PySide6.QtCore import Qt, QPointF, QRectF, QObject, Signal
from PySide6.QtGui import QPainterPath, QPen, QColor
from PySide6.QtWidgets import QGraphicsPathItem
from PySide6.QtGui import QPixmap, QImage, QImageReader, QWheelEvent, QMouseEvent, QPainter
from PySide6.QtWidgets import QGraphicsView


class ToolStrategy(QObject):
    """工具策略接口"""
    finished = Signal(object)  # 传递完成后的图形对象
    be_reset = Signal()

    def reset(self, viewer):
        pass

    def mouse_press(self, viewer, event: QMouseEvent):
        pass

    def mouse_move(self, viewer, event: QMouseEvent):
        pass

    def mouse_release(self, viewer, event: QMouseEvent):
        pass

    def activate(self, viewer):
        """激活工具时的初始化"""
        viewer.setDragMode(QGraphicsView.DragMode.NoDrag)
        viewer.setCursor(Qt.CrossCursor)

    def deactivate(self, viewer):
        """停用工具时的清理"""
        viewer.setCursor(Qt.ArrowCursor)
        self.reset(viewer)




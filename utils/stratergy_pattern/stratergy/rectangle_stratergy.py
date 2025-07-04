from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QPen, QColor
from PySide6.QtWidgets import QGraphicsRectItem
from PySide6.QtGui import QMouseEvent
from utils.stratergy_pattern.tool_stratergy import ToolStrategy

class RectangleStrategy(ToolStrategy):
    """矩形选框工具策略"""

    def __init__(self):
        super().__init__()
        self._start_point = None
        self._rect_item = None
        self._pen = QPen(QColor(255, 0, 0), 2, Qt.DashLine)

    def reset(self, viewer):
        if self._rect_item is not None:
            viewer.scene().removeItem(self._rect_item)
            self._rect_item = None
            self.be_reset.emit()

    def mouse_press(self, viewer, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            # 移除之前的矩形
            self.reset(viewer)

            # 记录起始点
            self._start_point = viewer.mapToScene(event.pos())

            # 创建新的矩形项
            self._rect_item = QGraphicsRectItem()
            self._rect_item.setPen(self._pen)
            viewer.scene().addItem(self._rect_item)

    def mouse_move(self, viewer, event: QMouseEvent):
        # 确保起始点和矩形项存在
        if self._start_point is None or self._rect_item is None:
            return

        # 获取当前鼠标位置
        end_point = viewer.mapToScene(event.pos())
        if not isinstance(end_point, QPointF):
            return

        # 更新矩形
        rect = QRectF(self._start_point, end_point).normalized()
        self._rect_item.setRect(rect)

    def mouse_release(self, viewer, event: QMouseEvent):
        if self._rect_item is not None:
            # 发射完成信号
            self.finished.emit(self._rect_item)

            # 重置状态（但保留矩形项，以便下次按下时移除）
            self._start_point = None
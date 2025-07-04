from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainterPath, QPen, QColor, QMouseEvent
from PySide6.QtWidgets import QGraphicsPathItem
from utils.stratergy_pattern import tool_stratergy

class LassoStrategy(tool_stratergy.ToolStrategy):
    """套索工具策略（自动清除旧选区）"""

    def __init__(self):
        super().__init__()
        self._path = None
        self._path_item = None
        self._points = []

    def reset(self, viewer):
        if self._path_item is not None:
            viewer.scene().removeItem(self._path_item)
            self._path_item = None
            self.be_reset.emit()


    def mouse_press(self, viewer, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            # 清除旧套索（如果存在）
            self.reset(viewer)

            # 初始化新路径
            scene_pos = viewer.mapToScene(event.pos())
            self._points = [scene_pos]

            self._path = QPainterPath()
            self._path.moveTo(scene_pos)

            # 创建新路径图形项
            self._path_item = QGraphicsPathItem()
            self._path_item.setPath(self._path)
            self._path_item.setPen(QPen(QColor(0, 255, 0), 1, Qt.DashLine))
            viewer.scene().addItem(self._path_item)

    def mouse_move(self, viewer, event: QMouseEvent):
        # 确保路径和图形项已初始化
        if self._path is None or self._path_item is None:
            return

        scene_pos = viewer.mapToScene(event.pos())
        if not isinstance(scene_pos, QPointF):
            return

        # 更新路径
        self._points.append(scene_pos)
        self._path.lineTo(scene_pos)
        self._path_item.setPath(self._path)

    def mouse_release(self, viewer, event: QMouseEvent):
        # 确保路径闭合且有效
        if self._path is None or self._path_item is None or len(self._points) < 3:
            return

        # 闭合路径
        self._path.closeSubpath()
        self._path_item.setPath(self._path)

        # 发射完成信号（传递最终路径）
        self.finished.emit(self._path_item)

        # 重置状态（但保留_path_item以便下次按下时清除）
        self._path = None
        self._points = []
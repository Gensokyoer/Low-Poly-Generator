from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QPainterPath, QPen, QColor, QImage, QMouseEvent
from PySide6.QtWidgets import QGraphicsPathItem
from utils.stratergy_pattern import tool_stratergy
from collections import deque

class MagicWandStrategy(tool_stratergy.ToolStrategy):
    """魔棒工具策略"""

    def __init__(self):
        super().__init__()
        self._tolerance = 30  # 默认容差
        self._contiguous = True  # 默认仅选择连续区域
        self._path_item = None

    def set_tolerance(self, value: int):
        """设置颜色容差（0-255）"""
        self._tolerance = max(0, min(value, 255))

    def set_contiguous(self, enabled: bool):
        """设置是否仅选择连续区域"""
        self._contiguous = enabled

    def mouse_press(self, viewer, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            # 获取点击位置的图像像素颜色
            scene_pos = viewer.mapToScene(event.pos())
            image = viewer.get_image()
            if image is None:
                return

            x, y = int(scene_pos.x()), int(scene_pos.y())
            if not (0 <= x < image.width() and 0 <= y < image.height()):
                return

            target_color = QColor(image.pixel(x, y))
            selected_pixels = self._flood_fill(image, x, y, target_color) if self._contiguous \
                else self._global_select(image, target_color)

            # 将选中的像素转换为路径
            path = self._pixels_to_path(selected_pixels)
            self._path_item = QGraphicsPathItem()
            self._path_item.setPath(path)
            self._path_item.setPen(QPen(QColor(0, 255, 0), 1, Qt.DashLine))
            viewer.scene().addItem(self._path_item)

            # 发射完成信号
            self.finished.emit(self._path_item)

    def _flood_fill(self, image: QImage, x: int, y: int, target_color: QColor) -> set:
        """区域生长算法（连续模式）"""
        selected = set()
        queue = deque([(x, y)])
        width, height = image.width(), image.height()

        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) in selected or not (0 <= cx < width and 0 <= cy < height):
                continue

            current_color = QColor(image.pixel(cx, cy))
            if self._color_distance(target_color, current_color) <= self._tolerance:
                selected.add((cx, cy))
                # 8邻域扩展
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                    queue.append((cx + dx, cy + dy))

        return selected

    def _global_select(self, image: QImage, target_color: QColor) -> set:
        """全局选择（非连续模式）"""
        selected = set()
        width, height = image.width(), image.height()

        for x in range(width):
            for y in range(height):
                current_color = QColor(image.pixel(x, y))
                if self._color_distance(target_color, current_color) <= self._tolerance:
                    selected.add((x, y))

        return selected

    def _color_distance(self, c1: QColor, c2: QColor) -> float:
        """计算颜色距离（简化曼哈顿距离）"""
        return abs(c1.red() - c2.red()) + abs(c1.green() - c2.green()) + abs(c1.blue() - c2.blue())

    def _pixels_to_path(self, pixels: set) -> QPainterPath:
        """将像素集合转换为QPainterPath（简化版：使用边界矩形）"""
        if not pixels:
            return QPainterPath()

        # 计算选中区域的边界矩形
        min_x = min(p[0] for p in pixels)
        max_x = max(p[0] for p in pixels)
        min_y = min(p[1] for p in pixels)
        max_y = max(p[1] for p in pixels)

        path = QPainterPath()
        path.addRect(QRectF(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1))
        return path

from PySide6.QtCore import Qt, Signal, QObject, QPoint, QRect
from PySide6.QtWidgets import (QApplication, QWidget, QFileDialog, QGraphicsPixmapItem,
                               QMessageBox, QGraphicsScene, QGraphicsView)
from PySide6.QtGui import QPixmap, QImage, QImageReader, QWheelEvent, QMouseEvent, QPainter

class ImageViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

    # def wheelEvent(self, event: QWheelEvent):
    #     """鼠标滚轮缩放"""
    #     zoom_factor = 1.2
    #     if event.angleDelta().y() > 0:
    #         # 放大
    #         self.scale(zoom_factor, zoom_factor)
    #     else:
    #         # 缩小
    #         self.scale(1.0 / zoom_factor, 1.0 / zoom_factor)

    def wheelEvent(self, event: QWheelEvent):
        """鼠标滚轮缩放，限制缩放范围在0.1-10倍之间"""
        zoom_factor = 1.2

        # 获取当前缩放比例
        current_scale = self.transform().m11()  # 获取x方向的缩放比例

        if event.angleDelta().y() > 0:
            # 放大 - 检查是否超过最大限制
            if current_scale * zoom_factor <= 10:
                self.scale(zoom_factor, zoom_factor)
        else:
            # 缩小 - 检查是否低于最小限制
            if current_scale / zoom_factor >= 0.1:
                self.scale(1.0 / zoom_factor, 1.0 / zoom_factor)

    def fit_view(self):
        """适应视图大小"""
        if self.scene() and self.scene().items():
            self.fitInView(self.scene().itemsBoundingRect(), Qt.KeepAspectRatio)

    def get_image(self) -> QImage:
        """Returns the currently displayed image as a QImage.

        Returns:
            QImage: The current image being displayed, or None if no image is present.
        """
        if not self.scene() or not self.scene().items():
            return None

        # Get the first item from the scene (assuming it's the image)
        item = self.scene().items()[0]
        if isinstance(item, QGraphicsPixmapItem):
            return item.pixmap().toImage()
        return None

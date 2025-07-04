
import cv2
import numpy as np
from PySide6.QtGui import QImage, QPainter, QPainterPath, QBrush
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGraphicsPathItem
from PySide6.QtCore import QBuffer


def lasso_path_to_mask(path_item, image_shape):
    height, width = image_shape

    mask_image = QImage(width, height, QImage.Format_ARGB32)
    mask_image.fill(Qt.transparent)

    painter = QPainter(mask_image)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(Qt.white))
    painter.drawPath(path_item.path())
    painter.end()

    # 将QImage保存为内存中的PNG数据
    buffer = QBuffer()
    buffer.open(QBuffer.ReadWrite)
    mask_image.save(buffer, "PNG")
    png_data = buffer.data()
    buffer.close()

    # 使用OpenCV解码PNG数据
    arr = cv2.imdecode(np.frombuffer(png_data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        return np.zeros((height, width), dtype=np.uint8)

    # 确保尺寸与原始图像一致
    if arr.shape != (height, width):
        arr = cv2.resize(arr, (width, height))

    return arr
import cv2
import numpy as np
from PySide6.QtGui import QImage, QPainter, QBrush
from PySide6.QtCore import Qt
from PySide6.QtCore import QBuffer


def rect_item_to_mask(rect_item, image_shape):
    """将QGraphicsRectItem转换为二值掩膜数组

    Args:
        rect_item: QGraphicsRectItem对象
        image_shape: 目标图像的形状(height, width)

    Returns:
        numpy.ndarray: 二值掩膜数组(0和255)
    """
    height, width = image_shape

    # 创建透明背景的QImage
    mask_image = QImage(width, height, QImage.Format_ARGB32)
    mask_image.fill(Qt.transparent)

    # 绘制矩形到QImage上
    painter = QPainter(mask_image)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(Qt.white))

    # 获取矩形的位置和尺寸
    rect = rect_item.rect()
    painter.drawRect(rect)
    painter.end()

    # 将QImage转换为PNG格式的内存数据
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
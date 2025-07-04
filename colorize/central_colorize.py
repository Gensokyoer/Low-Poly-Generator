import cv2
import numpy as np


def colorize_delaunay(mesh, original_image):
    height, width, channels = original_image.shape
    colorized_image = np.ones((height, width, channels), dtype=np.uint8) * 255
    triangles = mesh['triangles']
    vertices = mesh['vertices']

    for triangle in triangles:
        # 1. 获取三角形三个顶点的坐标（格式为 (x, y)）
        pts_coords = vertices[triangle].astype(np.int32)  # 形状 (3, 2)

        # 2. 转换为 OpenCV 所需的填充格式
        pts = pts_coords.reshape((-1, 1, 2))  # 形状 (3, 1, 2)

        # 3. 计算质心坐标（基于顶点坐标）
        centroid_x = np.mean(pts_coords[:, 0])  # x坐标均值
        centroid_y = np.mean(pts_coords[:, 1])  # y坐标均值

        # 4. 约束坐标范围（确保不越界）
        x = np.clip(int(centroid_x), 0, original_image.shape[1] - 1)
        y = np.clip(int(centroid_y), 0, original_image.shape[0] - 1)

        # 5. 获取颜色（注意 OpenCV 使用 BGR 格式）
        color = original_image[y, x].tolist()

        # 6. 填充三角形
        cv2.fillPoly(colorized_image, [pts], color)

    return colorized_image
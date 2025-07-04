import cv2
import numpy as np


def colorize_delaunay(mesh, original_image):
    colorized_image = original_image.copy()
    triangles = mesh['triangles']
    vertices = mesh['vertices']
    for triangle in triangles:
        # 获取三角形的顶点坐标
        pts = vertices[triangle].astype(np.int32)  # 形状为 (3, 2)
        pts = pts.reshape((-1, 1, 2))  # 转换为 OpenCV 所需的格式

        # 利用掩码提取三角形区域所有像素
        mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        ys, xs = np.where(mask == 255)

        if len(xs) == 0:
            # 如果区域内没有像素，则退回到使用质心采样
            centroid = np.mean(triangle, axis=0).astype(np.int32)
            centroid[0] = np.clip(centroid[0], 0, original_image.shape[1] - 1)
            centroid[1] = np.clip(centroid[1], 0, original_image.shape[0] - 1)
            median_color = original_image[centroid[1], centroid[0]].tolist()
        else:
            # 提取该区域内所有像素的 BGR 颜色
            colors = original_image[ys, xs]  # shape: (n, 3)
            # 将 BGR 颜色转换到 Lab 颜色空间
            lab_colors = cv2.cvtColor(colors.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
            lab_colors = lab_colors.reshape(-1, 3)
            # 根据 Lab 空间中的 L 分量（亮度）对颜色进行排序
            L_values = lab_colors[:, 0].astype(np.float32)
            sorted_indices = np.argsort(L_values)

            n = len(sorted_indices)
            # 选择 40% 到 60% 之间的像素索引，避免过暗或过亮的极端值
            low_idx = int(n * 0.4)
            high_idx = int(n * 0.6)
            if high_idx <= low_idx:
                selected_indices = sorted_indices
            else:
                selected_indices = sorted_indices[low_idx:high_idx]

            selected_colors = colors[selected_indices]
            # 分别求出 B, G, R 三个通道在所选区域内的中位值作为最终颜色
            median_B = int(np.median(selected_colors[:, 0]))
            median_G = int(np.median(selected_colors[:, 1]))
            median_R = int(np.median(selected_colors[:, 2]))
            median_color = [median_B, median_G, median_R]

        # 用选定的中位数颜色填充三角形
        cv2.fillPoly(colorized_image, [pts], median_color)

    return colorized_image

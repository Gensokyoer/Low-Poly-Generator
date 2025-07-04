import cv2
import numpy as np
import random


def draw_edge(points, img):
    # 生成随机颜色，确保颜色不为黑色
    color = (random.randint(0, 255),
             random.randint(0, 255),
             random.randint(0, 255))
    while color == (0, 0, 0):
        color = (random.randint(0, 255),
                 random.randint(0, 255),
                 random.randint(0, 255))

    # 按顺序将所有相邻的点连接起来，绘制线段
    i = 0
    start_pt = points[i]
    start_pt = start_pt[::-1]
    end_pt = points[i + 1]
    end_pt = end_pt[::-1]
    cv2.circle(img, start_pt, radius=1, color=color, thickness=2)
    for i in range(len(points) - 1):
        start_pt = points[i]
        start_pt = start_pt[::-1]
        end_pt = points[i + 1]
        end_pt = end_pt[::-1]

        cv2.line(img, start_pt, end_pt, color, thickness=1)
        cv2.circle(img, end_pt, radius=1, color=color, thickness=2)

    return img


# 测试代码示例
if __name__ == "__main__":
    points = [(274, 330), (283, 318), (286, 305), (281, 287), (268, 271), (273, 330), (266, 334)]
    # 定义画布尺寸，例如400 x 400像素，3个颜色通道（BGR）
    img_shape = (400, 400, 3)

    canvas = draw_edge(points, img_shape)

    # 显示绘制后的画布
    cv2.imshow("Edge Drawing", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

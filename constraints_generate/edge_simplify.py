import numpy as np
from constraints_generate.douglas_peuker import rdp
from utils.edge_drawer import draw_edge
from constraints_generate.extend_edges import extend_constrained_edges

def genetrate_simpl_edges(contours, img, eta, epsilon):
    rows, cols = img.shape[:2]
    #maxLen = 99999
    maxLen = eta * (rows + cols)
    corners = [
        (0, 0),  # 左上
        (cols - 1, 0),  # 右上
        (cols - 1, rows - 1),  # 右下
        (0, rows - 1)  # 左下
    ]

    # 约束点和约束边初始化
    constraint_points = corners
    constraint_edges = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0]
    ])
    num = 4  # 点集大小

    # 绘制用的图像副本
    constraint_edges_img = img.copy()

    # 处理每个轮廓
    for contour in contours:
        sp = rdp(contour, epsilon, maxLen)
        # 绘制图像
        constraint_edges_img = draw_edge(sp, constraint_edges_img)

        # 转化点集
        valid_points = np.array([(y, x) for x, y in sp], dtype=np.int32)

        # 处理点集
        constraint_points.extend(valid_points)

        # 生成边集
        constraint_edges = extend_constrained_edges(constraint_edges, valid_points, num)
        num += len(sp)



    return constraint_points, constraint_edges, constraint_edges_img
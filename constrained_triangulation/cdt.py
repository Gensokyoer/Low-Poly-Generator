import numpy as np
import cv2
import triangle as tr

def generate_constrained_delaunay(points_arr, constrained_edges, image_shape):
    """
    生成带约束边的约束Delaunay三角剖分，并在图像上绘制三角网格

    参数:
        points_arr: 原始点集，格式为Nx2的NumPy数组
        constraint_edges: 约束边的索引列表，格式为Kx2的NumPy数组
        image_shape: 输出图像尺寸

    返回:
        tri_img: 绘制三角网格的图像
        cdt: 约束Delaunay剖分结果（包含顶点和三角形信息）
    """
    if len(points_arr) < 3:
        return tri_img, None

    # 构建输入数据结构
    data = {
        'vertices': points_arr,
        'segments': constrained_edges
    }
    # 执行约束Delaunay三角剖分
    mesh = tr.triangulate(data, 'pD')

    # 提取顶点和三角形
    vertices = mesh['vertices']
    triangles = mesh['triangles'].astype(int)

    # 在图像上绘制三角网格
    tri_img = np.zeros(image_shape, dtype=np.uint8)
    for triangle in triangles:
        pts = vertices[triangle].astype(int)
        pts = pts.reshape(-1, 1, 2)
        cv2.polylines(tri_img, [pts], isClosed=True, color=(255, 255, 255), thickness=1)

    return tri_img, mesh
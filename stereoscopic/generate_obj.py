import numpy as np
import trimesh
from PIL import Image

def normalize_depth(depth, depth_max=20):
    """
    将深度图按指定上限归一化，增强凹凸效果。

    参数:
        depth: 输入深度图，值范围不限，形状为 (H, W)。
        depth_max: 深度上限参数，控制归一化后的取值范围。
                  值越大，凹凸差异越明显（默认1.0）。

    返回:
        normalized_depth: 归一化后的深度图，值范围 [0, depth_max]。
    """
    # 将深度图缩放到 [0, 1] 范围
    depth_min = depth.min()
    depth_range = depth.max() - depth_min
    if depth_range > 0:
        depth_normalized = (depth - depth_min) / depth_range
    else:
        depth_normalized = np.zeros_like(depth)  # 避免除以零

    # 按深度上限参数拉伸
    normalized_depth = depth_normalized * depth_max

    return normalized_depth

def mesh_depth_to_obj(mesh, depth, low_poly_img, depth_max = 20, output_path="output.obj"):
    """
    将2D三角网格 + 深度图转换为3D OBJ模型，并从Low Poly图像上取样颜色应用于每个面片。

    参数:
        mesh: triangle库生成的三角网格对象，需包含以下属性:
            - vertices: 2D顶点数组，形状为 (N, 2)
            - triangles: 三角形面片索引数组，形状为 (M, 3)
        depth: 深度图（与mesh同尺寸），值范围建议归一化到[0,1]，
               形状为 (H, W) 或 (H, W, 1)，深度值越大表示Z坐标越高。
        low_poly_img: Low Poly图像，用于取样颜色，形状为 (H, W, 3) 或 (H, W, 4)
        output_path: 输出的OBJ文件路径。
    """
    # 确保深度图是二维的
    if depth.ndim == 3:
        depth = depth.squeeze()

    # 确保Low Poly图像是RGB格式（去除alpha通道如果存在）
    if low_poly_img.ndim == 3 and low_poly_img.shape[2] == 4:
        low_poly_img = low_poly_img[:, :, :3]
    elif low_poly_img.ndim == 2:
        low_poly_img = np.stack([low_poly_img] * 3, axis=-1)

    depth = normalize_depth(depth, depth_max)

    # 获取网格的2D顶点和三角形面片
    vertices_2d = mesh['vertices']
    triangles = mesh['triangles'].astype(int)
    num_faces = len(triangles)

    # 将2D顶点坐标转换为图像像素坐标
    height, width = depth.shape
    pixel_coords = vertices_2d.copy()

    # 如果mesh坐标是归一化的，转换为像素坐标
    if np.max(pixel_coords) <= 1.0:
        pixel_coords[:, 0] *= (width - 1)
        pixel_coords[:, 1] *= (height - 1)
    pixel_coords = np.round(pixel_coords).astype(int)

    # 确保坐标不越界
    pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, width - 1)
    pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, height - 1)

    # 从深度图中提取Z坐标（注意：图像坐标系Y轴向下）
    z_values = depth[pixel_coords[:, 1], pixel_coords[:, 0]]  # (N,)
    # z_values = depth[height - 1 - pixel_coords[:, 1], pixel_coords[:, 0]]  # 翻转Y坐标

    # 合并为3D顶点 (X, Y, Z)
    vertices_3d = np.column_stack([vertices_2d, z_values])  # (N, 3)

    # 计算每个三角形的中心点坐标（在原始图像空间）
    triangle_centers = np.zeros((num_faces, 2))
    for i, tri in enumerate(triangles):
        # 获取三角形的三个顶点
        v0 = vertices_2d[tri[0]]
        v1 = vertices_2d[tri[1]]
        v2 = vertices_2d[tri[2]]
        # 计算中心点
        center = (v0 + v1 + v2) / 3.0
        triangle_centers[i] = center

    # 将中心点坐标转换为图像像素坐标
    img_height, img_width = low_poly_img.shape[:2]
    center_pixel_coords = triangle_centers.copy()

    # 如果顶点坐标是归一化的，转换为像素坐标
    if np.max(vertices_2d) <= 1.0:
        center_pixel_coords[:, 0] *= (img_width - 1)
        center_pixel_coords[:, 1] *= (img_height - 1)

    center_pixel_coords = np.round(center_pixel_coords).astype(int)
    center_pixel_coords[:, 0] = np.clip(center_pixel_coords[:, 0], 0, img_width - 1)
    center_pixel_coords[:, 1] = np.clip(center_pixel_coords[:, 1], 0, img_height - 1)

    # 从Low Poly图像中取样颜色（注意：图像坐标系Y轴向下）
    face_colors = low_poly_img[center_pixel_coords[:, 1], center_pixel_coords[:, 0]]  # (M, 3)
    # face_colors = low_poly_img[img_height - 1 - center_pixel_coords[:, 1], center_pixel_coords[:, 0]]

    # 生成材质文件
    mtl_filename = output_path.replace('.obj', '.mtl')

    # 先写入MTL材质文件
    with open(mtl_filename, 'w') as f:
        for i in range(num_faces):
            r, g, b = face_colors[i] / 255.0  # 假设图像是0-255范围，转换为0-1
            f.write(f"newmtl material_{i}\n")
            f.write(f"Kd {b:.6f} {g:.6f} {r:.6f}\n")
            f.write("illum 1\n\n")

    # 写入OBJ文件
    with open(output_path, 'w') as f:
        # 写入材质库引用
        f.write(f"mtllib {mtl_filename.split('/')[-1]}\n\n")


        for v in vertices_3d:
            f.write(f"v {v[0]} {-v[1]} {v[2]}\n")

        # 写入面片并应用材质
        for i, face in enumerate(triangles):
            f.write(f"usemtl material_{i}\n")
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    print(f"OBJ模型已保存到: {output_path}")
    print(f"材质文件已保存到: {mtl_filename}")


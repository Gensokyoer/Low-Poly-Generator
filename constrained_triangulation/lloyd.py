import numpy as np
from scipy.spatial import cKDTree

def lloyd_relaxation(free_points, feature_flow_map, image_size, constraints=None, iterations=10):
    """
    利用 Lloyd 松弛算法对自由点进行迭代优化，
    兼顾约束点和加权质心计算（基于特征流图）。

    参数:
        free_points : numpy 数组, 形状 (N, 2)，初始自由点坐标
        feature_flow_map : 二维 numpy 数组 (height, width)，用于加权计算质心的特征流图
        image_size : (height, width)，图像尺寸（用于确定边界）
        constraints : numpy 数组, 形状 (M, 2)，约束点（不参与更新，仅用于参与 Voronoi 划分），其中已经包含边界上的四个角点
        iterations : 整数，迭代次数，默认为 10

    返回:
        更新后的自由点坐标（numpy 数组，形状为 (N, 2)）
    """
    free_points = free_points.copy()
    height, width = image_size

    # 预生成像素坐标网格（优化性能）
    x_coords = np.arange(width)
    y_coords = np.arange(height)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    pixel_coords = np.column_stack([grid_x.ravel(), grid_y.ravel()])  # (H*W, 2)

    for _ in range(iterations):
        # 合并自由点和约束点
        all_points = np.vstack([free_points, constraints]) if constraints is not None else free_points.copy()

        # 构建KDTree加速最近邻搜索
        tree = cKDTree(all_points)
        _, labels = tree.query(pixel_coords, k=1)

        new_positions = np.zeros_like(free_points)

        for i in range(len(free_points)):
            # 获取当前点的Voronoi区域像素
            mask = labels == i
            region_pixels = pixel_coords[mask]

            if len(region_pixels) == 0:
                new_pos = free_points[i]
            else:
                # 提取特征流权重（注意坐标转换）
                x_pix = region_pixels[:, 0].astype(int)
                y_pix = region_pixels[:, 1].astype(int)
                weights = feature_flow_map[y_pix, x_pix] / 255.0

                # 计算加权质心
                total_weight = np.sum(weights)
                if total_weight > 1e-8:
                    weighted_x = np.dot(x_pix, weights) / total_weight
                    weighted_y = np.dot(y_pix, weights) / total_weight
                    new_pos = np.array([weighted_x, weighted_y])
                else:
                    new_pos = free_points[i]

            # 边界约束处理
            original = free_points[i]
            on_vertical = np.isclose(original[0], 0, atol=1e-4) or np.isclose(original[0], width - 1, atol=1e-4)
            on_horizontal = np.isclose(original[1], 0, atol=1e-4) or np.isclose(original[1], height - 1, atol=1e-4)

            if on_vertical: new_pos[0] = original[0]
            if on_horizontal: new_pos[1] = original[1]

            # 坐标裁剪
            new_pos = np.clip(new_pos, [0, 0], [width - 1, height - 1])
            new_positions[i] = new_pos

        free_points = new_positions

    return free_points
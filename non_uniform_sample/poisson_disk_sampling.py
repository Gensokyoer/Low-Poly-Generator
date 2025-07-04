import numpy as np
# import cv2
# import random
# import math
from scipy.spatial import KDTree
from scipy.ndimage import label


def poisson(mask, num_points, min_dist):
    """
    在给定的区域掩码内执行泊松盘采样，支持多连通区域。

    参数:
        mask (numpy.ndarray): 二值化的区域掩码，True 表示可采样区域。
        num_points (int): 采样点的数量期望。
        min_dist (float): 采样点之间的绝对最小距离限制。

    返回:
        points (list of tuple): 生成的采样点列表，每个元素为 (x, y) 坐标。
    """
    # 参数校验
    if num_points <= 0:
        return []
    if min_dist <= 0:
        raise ValueError("min_dist 必须大于 0")

    # 预处理：识别所有连通分量
    labeled_mask, num_features = label(mask)
    regions = [np.column_stack(np.where(labeled_mask == i)) for i in range(1, num_features + 1)]

    # 计算动态距离
    valid_pixels = np.sum(mask)
    area_ratio = valid_pixels / mask.size
    effective_area = area_ratio * mask.shape[0] * mask.shape[1]
    avg_area_per_point = effective_area / num_points
    dynamic_dist = np.sqrt(avg_area_per_point / np.pi)
    actual_min_dist = max(dynamic_dist, min_dist)

    # 初始化数据结构
    cell_size = actual_min_dist / np.sqrt(2)
    grid = np.full((int(np.ceil(mask.shape[0] / cell_size)) + 1,
                    int(np.ceil(mask.shape[1] / cell_size)) + 1), -1)
    samples = []
    active = []

    # 第一阶段：为每个连通分量初始化至少一个点
    for region in regions:
        if len(region) == 0:
            continue
        # 随机选择区域内的点
        idx = np.random.choice(len(region))
        x, y = region[idx][1], region[idx][0]
        # 检查是否满足最小距离
        valid = True
        grid_x, grid_y = int(x / cell_size), int(y / cell_size)
        for i in range(max(0, grid_y - 2), min(grid.shape[0], grid_y + 3)):
            for j in range(max(0, grid_x - 2), min(grid.shape[1], grid_x + 3)):
                if grid[i, j] != -1:
                    if np.linalg.norm(np.array([x, y]) - samples[grid[i, j]]) < actual_min_dist:
                        valid = False
                        break
            if not valid: break
        if valid:
            samples.append((x, y))
            active.append(len(samples) - 1)
            grid[grid_y, grid_x] = len(samples) - 1

    # 第二阶段：改进的扩散采样
    kd_tree = KDTree(np.column_stack(np.where(mask))[::-1])  # (x,y)格式
    while active and len(samples) < num_points * 1.5:
        idx = np.random.choice(active)
        point = samples[idx]
        found = False

        # 生成候选点
        for _ in range(30):
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(actual_min_dist, 2 * actual_min_dist)
            new_x = point[0] + radius * np.cos(angle)
            new_y = point[1] + radius * np.sin(angle)

            # 边界检查
            if not (0 <= new_x < mask.shape[1] and 0 <= new_y < mask.shape[0]):
                continue

            # 有效性检查（精确到像素级）
            _, nearest_idx = kd_tree.query([new_x, new_y])
            nearest_valid = kd_tree.data[nearest_idx]
            if np.linalg.norm([new_x - nearest_valid[0], new_y - nearest_valid[1]]) > 1.0:
                continue

            # 冲突检查
            grid_x, grid_y = int(new_x / cell_size), int(new_y / cell_size)
            valid = True
            for i in range(max(0, grid_y - 2), min(grid.shape[0], grid_y + 3)):
                for j in range(max(0, grid_x - 2), min(grid.shape[1], grid_x + 3)):
                    if grid[i, j] != -1:
                        if np.linalg.norm(np.array([new_x, new_y]) - samples[grid[i, j]]) < actual_min_dist:
                            valid = False
                            break
                if not valid: break
            if valid:
                samples.append((new_x, new_y))
                active.append(len(samples) - 1)
                grid[grid_y, grid_x] = len(samples) - 1
                found = True
                break

        if not found:
            active.remove(idx)

    # 第三阶段：确保所有连通分量都有采样点
    final_tree = KDTree(samples) if samples else None
    for region in regions:
        if len(region) == 0:
            continue
        region_points = [(p[1], p[0]) for p in region]  # 转换为(x,y)格式
        if not samples:
            samples.append(region_points[np.random.choice(len(region_points))])
            continue
        # 检查该区域是否已有采样点
        distances, _ = final_tree.query(region_points)
        if np.min(distances) > actual_min_dist * 2:
            # 该区域未被覆盖，强制添加一个点
            new_point = region_points[np.random.choice(len(region_points))]
            samples.append(new_point)

    # 最终采样数量控制
    # if len(samples) > num_points:
    #     return [samples[i] for i in np.random.choice(len(samples), num_points, replace=False)]
    return samples
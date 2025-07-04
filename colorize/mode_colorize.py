import cv2
import numpy as np


def quantize_colors(colors, bucket_size):
    """
    将 colors (N×3 uint8) 按 bucket_size 在每个通道上量化，
    返回量化后的桶索引和每个桶对应的中心色。
    """
    # 量化索引 0,1,2,... e.g. bucket_size=16 时，255//16 = 15 最大桶号
    idx = (colors // bucket_size).astype(np.int32)
    # 将三通道索引合并为一个一维键： i = r_idx*桶数^2 + g_idx*桶数 + b_idx
    max_bucket = 256 // bucket_size + (1 if 256 % bucket_size else 0)
    keys = idx[:, 0] * (max_bucket ** 2) + idx[:, 1] * max_bucket + idx[:, 2]
    # 计算每个桶的中心色（可选，用于填充），中心 = 桶号*bucket_size + bucket_size//2，且裁剪到 [0,255]
    centers = (idx * bucket_size + bucket_size // 2).clip(0, 255).astype(np.uint8)
    return keys, centers


def colorize_delaunay_mode(mesh, original_image, bucket_size=32):
    """
    使用“众数（Mode）入桶”思路对 Delaunay 三角网着色。
    bucket_size：颜色量化宽度，建议 16 或 32。
    """
    h, w = original_image.shape[:2]
    out = original_image.copy()
    triangles = mesh['triangles']
    vertices = mesh['vertices']

    for tri in triangles:
        # 顶点坐标转整型并 reshape
        pts = vertices[tri].astype(np.int32).reshape(-1, 1, 2)
        # 掩码 & 像素提取
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        ys, xs = np.where(mask == 255)

        if len(xs) == 0:
            # 区域空时采样质心
            cen = np.mean(vertices[tri], axis=0).astype(np.int32)
            cx, cy = np.clip(cen[0], 0, w - 1), np.clip(cen[1], 0, h - 1)
            fill_color = original_image[cy, cx].tolist()
        else:
            # 提取原始 BGR 像素
            pix = original_image[ys, xs]  # shape: (n,3)
            # 量化并取桶键
            keys, centers = quantize_colors(pix, bucket_size)
            # 统计众数桶
            unique_keys, counts = np.unique(keys, return_counts=True)
            mode_idx = unique_keys[np.argmax(counts)]
            # 找到对应中心色（任选 centers 中任一对应像素的 center）
            # 也可选用该桶所有原始像素的均值或中心色
            # 这里直接取第一个匹配的中心色
            fill_color = centers[keys == mode_idx][0].tolist()

        # 填充三角形
        cv2.fillPoly(out, [pts], fill_color)

    return out

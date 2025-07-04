import numpy as np
import cv2
import random
from non_uniform_sample.poisson_disk_sampling import poisson


def get_free_points(saliency_map, image_shape, Nc = 0, deltaN = 0, threshold=32, eta=0.02, lambda_=0.7):
    """
    根据显著性图进行显著性采样，返回自由采样点集。

    参数:
        saliency_map (numpy.ndarray): 输入的显著性图，取值范围 [0, 255]。
        image_shape (tuple): 原始图像的尺寸，格式为 (height, width)。
        threshold (int): 区分显著区域和背景区域的阈值，默认值为 64。
        eta (float): 控制采样密度的参数，默认值为 0.02。
        lambda_ (float): 控制显著区域与背景区域采样点数量比例的参数，取值范围在 [0,1]，默认值为 0.7。

    返回:
        free_points (list of tuple): 自由采样点的列表，每个元素为 (x, y) 坐标。
    """
    height, width = image_shape[:2]
    # 计算采样间隔 Li
    Li = eta * (width + height)

    # 计算总的采样点数 N
    N = int(np.floor(width / Li) * np.floor(height / Li))
    N += Nc
    N += deltaN

    # 根据阈值将显著性图划分为显著区域和背景区域
    salient_region = saliency_map >= threshold
    background_region = saliency_map < threshold

    # 计算显著区域和背景区域的采样点数量
    Ns = max(int(lambda_ * (N - Nc)) , 1)
    Nb = max(int((1 - lambda_) * (N - Nc)), 1)

    # 在显著区域进行泊松盘采样
    points_s = poisson(salient_region, Ns, Li)

    # 在背景区域进行泊松盘采样
    points_b = poisson(background_region, Nb, Li)

    # 合并自由采样点
    free_points = points_s + points_b

    return free_points

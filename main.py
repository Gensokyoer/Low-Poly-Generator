import cv2
import numpy as np
import time

from constraints_generate.edge_simplify import genetrate_simpl_edges
from constrained_triangulation.lloyd import lloyd_relaxation
from edge_detect.cvED import edge_detection
from feature_map.distance_map import calculate_distance_map
from feature_map.feature_flow_map import generate_feature_flow_map
from non_uniform_sample.saliency_map import get_saliency_map
from non_uniform_sample.saliency_sample import get_free_points
from constrained_triangulation.cdt import generate_constrained_delaunay
from colorize.mid_colorize import colorize_delaunay
from colorize.mode_colorize import colorize_delaunay_mode
from colorize.postprocess import postprocess_artistic
from stereoscopic.depth_detect import get_depth
from constrained_triangulation.valid_check import validate_triangle_input, dedupe_vertices_and_segments


def get_all_imgs(img, blur_size = 5, epsilon = 4.0, relaxation_iterations=10, saliency_thres = 32, deltaN = 0, eta=0.02):
    start_time = time.time()
    start = start_time

    # 图像读取与预处理
    print("开始图像读取...")
    if img is None:
        raise FileNotFoundError(f"无法加载图像")
    height, width = img.shape[:2]
    end = time.time()
    print(f"图像读取完成，耗时: {end - start:.4f} 秒")
    start = end

    # 边缘检测
    print("开始边缘检测...")
    contours, edge_img = edge_detection(img, blur_size= blur_size)
    if not contours:
        raise RuntimeError("未检测到有效边缘")
    end = time.time()
    print(f"边缘检测完成，耗时: {end - start:.4f} 秒")
    start = end

    # 边缘简化
    print("开始边缘简化...")
    constrained_points, constrained_edges, simplified_edges_img = genetrate_simpl_edges(contours, img, eta, epsilon)
    end = time.time()
    print(f"边缘简化完成，耗时: {end - start:.4f} 秒")
    start = end

    # 得到特征流图
    print("开始计算特征流场...")
    dist = calculate_distance_map(edge_img)
    feat = generate_feature_flow_map(dist, m = eta * (height + width) / 2)
    end = time.time()
    print(f"特征流场计算完成，耗时: {end - start:.4f} 秒")
    start = end

    # 显著性采样
    print("开始显著性采样...")
    saliency_map = get_saliency_map(img)
    free_points = get_free_points(saliency_map, img.shape[:2], Nc = len(constrained_points),deltaN = deltaN, threshold = saliency_thres, eta = eta, lambda_=0.7)
    end = time.time()
    print(f"显著性采样完成，耗时: {end - start:.4f} 秒")
    start = end

    # Lloyd松弛优化
    print("开始Lloyd松弛优化...")
    optimized_free_points = lloyd_relaxation(
        free_points,
        feat,
        image_size = (height, width),
        constraints = constrained_points,
        iterations = relaxation_iterations
    )
    end = time.time()
    print(f"Lloyd松弛优化完成，耗时: {end - start:.4f} 秒")
    start = end

    # 约束三角剖分
    print("开始约束三角剖分...")
    all_points = np.vstack((constrained_points, optimized_free_points))
    print("总点数: ", len(all_points))
    validate_triangle_input(all_points, constrained_edges)
    unique_coords, new_segments = dedupe_vertices_and_segments(all_points, constrained_edges, decimal_round=6)
    tri_img, mesh = generate_constrained_delaunay(unique_coords, new_segments, (height,width))
    end = time.time()
    print(f"约束三角剖分完成，耗时: {end - start:.4f} 秒")
    start = end

    # 上色
    print("开始三角剖分上色...")
    low_poly_img = colorize_delaunay(mesh, img)
    # low_poly_img = colorize_delaunay_mode(mesh, img)
    end = time.time()
    print(f"三角剖分上色完成，耗时: {end - start:.4f} 秒")
    start = end

    # 模型输出
    print("开始立体深度计算...")
    depth_img, depth = get_depth(img)
    end = time.time()
    print(f"立体深度计算完成，耗时: {end - start:.4f} 秒")
    start = end

    # 着色后处理
    print("开始艺术化后处理...")
    artistic_results = postprocess_artistic(low_poly_img)
    end = time.time()
    print(f"艺术化后处理完成，耗时: {end - start:.4f} 秒")
    start = end

    result = {
        'original' : img, # 对应原图
        'edge' : edge_img, # 对应边缘特征
        'simpl_edge' : simplified_edges_img, # 简化后的边缘特征
        'tri_img' : tri_img, # 网格图像
        'dist' : dist, # 对应距离图
        'feature_flow_field' : feat, # 对应特征流场
        'saliency' : saliency_map, # 对应显著性区域
        'mesh' : mesh, # 网格
        'depth' : depth, # 深度图
        'low_poly' : low_poly_img, # 对应低多边形原色
        'depth_img' : depth_img, # 深度图
        'rgb_equa' : artistic_results['rgb_equalized'], # RGB均衡化结果
        'ycrcb_equa' : artistic_results['ycrcb_equalized'], # Ycrcb均衡化结果
        'lab_clahe' : artistic_results['lab_clahe'] # LAB CLAHE结果
    }

    end = time.time()  # 总结束时间
    print(f"全部处理完成，总耗时: {end - start_time:.4f} 秒")

    return result

def get_selected_region_imgs(img, mask=None, blur_size=5, epsilon=4.0, relaxation_iterations=10,
                           saliency_thres=32, deltaN=0, eta=0.02):
    # 1. 如果没有提供掩码，默认处理整个图像
    if mask is None:
        mask = np.ones_like(img[:, :, 0]) * 255  # 全白掩码

    # 2. 确保掩码是二值的
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 3. 对选中区域进行 Low Poly 处理
    selected_region = cv2.bitwise_and(img, img, mask=mask)
    original_mask = mask.copy()

    low_poly_result = get_all_imgs(
        selected_region,
        blur_size=blur_size,
        epsilon=epsilon,
        relaxation_iterations=relaxation_iterations,
        saliency_thres=saliency_thres,
        deltaN=deltaN,
        eta=eta
    )

    for key in ['low_poly', 'rgb_equa', 'ycrcb_equa', 'lab_clahe']:
        # 转换为三通道格式（兼容灰度mask）
        if len(original_mask.shape) == 2:
            mask_3ch = cv2.cvtColor(original_mask, cv2.COLOR_GRAY2BGR)
        else:
            mask_3ch = original_mask

        # 应用mask确保非处理区域为黑色
        low_poly_result[key] = cv2.bitwise_and(low_poly_result[key], mask_3ch)

    # 4. 合并 Low Poly 结果到原图
    def merge_images(low_poly_img):
        # 生成 Low Poly 区域的掩码（非黑色像素）
        low_poly_mask = cv2.cvtColor(low_poly_img, cv2.COLOR_BGR2GRAY)
        _, low_poly_mask = cv2.threshold(low_poly_mask, 0, 255, cv2.THRESH_BINARY)

        # 提取 Low Poly 有效区域
        processed_part = cv2.bitwise_and(low_poly_img, low_poly_img, mask=low_poly_mask)

        # 提取原图中未被 Low Poly 覆盖的区域
        original_part = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(low_poly_mask))

        # 合并两部分
        return cv2.add(original_part, processed_part)

    # 5. 生成最终结果
    merged_low_poly = merge_images(low_poly_result['low_poly'])
    merged_rgb_equa = merge_images(low_poly_result['rgb_equa'])
    merged_ycrcb_equa = merge_images(low_poly_result['ycrcb_equa'])
    merged_lab_clahe = merge_images(low_poly_result['lab_clahe'])

    # 6. 返回结果（与原函数格式一致）
    return {
        'original': img,
        'low_poly': merged_low_poly,
        'rgb_equa': merged_rgb_equa,
        'ycrcb_equa': merged_ycrcb_equa,
        'lab_clahe': merged_lab_clahe,
        # 其他字段（如边缘、距离图等）保持不变
        **{k: v for k, v in low_poly_result.items()
            if k not in ['low_poly', 'rgb_equa', 'ycrcb_equa', 'lab_clahe']}
    }







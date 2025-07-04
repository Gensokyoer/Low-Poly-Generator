import numpy as np
import cv2

def calculate_distance_map(edges):
    inverted_edges = cv2.bitwise_not(edges)
    dist = cv2.distanceTransform(inverted_edges, cv2.DIST_L2, 5)
    # 归一化并转换为8位图像
    dist_normalized = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
    dist_uint8 = np.uint8(dist_normalized)
    return dist_uint8

# from cvED import edge_detection
# image_path = 'D:\\photos\\lena.png'
# contours, edges = edge_detection(image_path)
#
# dist = calculate_distance_map(edges)
#
# # 使用颜色映射增强显示效果
# dist_color = cv2.applyColorMap(dist, cv2.COLORMAP_JET)
# cv2.imshow('Distance Map', dist_color)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

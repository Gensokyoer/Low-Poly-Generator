import cv2
import numpy as np

def generate_feature_flow_map(dist, m = 0.1):
    lane_idx = np.floor(dist / m).astype(np.int32)  # 每个像素所属车道的编号
    remainder = np.mod(dist, m)  # 在车道内的余量
    # 初始化特征流图（float类型）
    flow = np.zeros_like(dist)
    # 对应偶数车道：F(x) = (255/m) * remainder
    mask_even = (lane_idx % 2 == 0)
    # 对应奇数车道：F(x) = (255/m) * (m - remainder)
    mask_odd = np.logical_not(mask_even)

    flow[mask_even] = (255 / m) * remainder[mask_even]
    flow[mask_odd] = (255 / m) * (m - remainder[mask_odd])

    # 限制到[0,255]并转换为uint8方便显示
    feature_flow = np.clip(flow, 0, 255).astype(np.uint8)
    return feature_flow

# from edge_drawing import edge_detection
# from distance_map import calculate_distance_map
# image_path = 'D:\\photos\\mickey-mouse-1.jpg'
# img = cv2.imread(image_path)
# rows, cols = img.shape[:2]
# print(rows, cols)
# contours, edges = edge_detection(image_path)
# dist = calculate_distance_map(edges)
# generate_feature_flow_map(dist,m = 0.01*(rows + cols))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

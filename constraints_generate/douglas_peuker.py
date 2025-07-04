from utils.distance_calculate import *

def rdp(points, epsilon, maxLen):
    if len(points) <= 2:
        return points

    # 找到最大距离的点
    start_point = points[0]
    end_point = points[-1]
    max_dist = 0
    index = 0
    for i in range(1, len(points) - 1):
        point = points[i]
        dist = perpendicular_distance(start_point, end_point, point)
        # 调试输出
        # print(f"点 {i}, 距离: {dist}")
        if dist > max_dist:
            max_dist = dist
            index = i

    # 计算线段长度
    line_length = euclidean_distance(start_point, end_point)

    # 确定是否需要递归分割
    if max_dist > epsilon:
        # 根据最大距离点进行递归
        left = rdp(points[:index + 1], epsilon, maxLen)
        right = rdp(points[index:], epsilon, maxLen)
        result = left[:-1] + right
    elif line_length > maxLen:
        mid_index = len(points) // 2
        left = rdp(points[:mid_index + 1], epsilon, maxLen)
        right = rdp(points[mid_index:], epsilon, maxLen)
        result = left[:-1] + right
    else:
        result = [start_point, end_point]

    return result

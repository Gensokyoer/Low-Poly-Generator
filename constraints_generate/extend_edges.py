import numpy as np


def extend_constrained_edges(constraint_edges, new_points_arr, points_arr_length):
    """
    扩充约束边

    参数:
        constraint_edges: 原有的约束边 (Nx2 NumPy数组)
        new_points_arr: 新增的点集 (列表，按顺序定义新点)
        points_arr_length: 当前点集的长度 (int)，即原点集的长度

    返回:
        extended_edges: 扩展后的约束边 (NumPy数组)
    """
    # 计算新点的起始索引
    start_index = points_arr_length

    # 为新点生成对应的约束边
    new_edges = []
    for i in range(len(new_points_arr) - 1):
        new_edges.append([start_index + i, start_index + i + 1])

    # 合并原有的约束边和新增的约束边
    extended_edges = np.vstack([constraint_edges, np.array(new_edges)])

    return extended_edges


# # 示例数据
# constraint_edges = np.array([
#     [0, 1],
#     [1, 2]
# ])
#
# new_points_arr = [
#     (np.int32(787), np.int32(360)),
#     (np.int32(786), np.int32(340)),
#     (np.int32(783), np.int32(321))
# ]
#
# points_arr_length = 3 # 假设原点集有3个点
#
# # 调用函数
# result = extend_constraint_edges(constraint_edges, new_points_arr, points_arr_length)
# print(constraint_edges)
# print(result)

import numpy as np

def validate_triangle_input(coords: np.ndarray, segments: list[list[int]], eps: float = 1e-8):
    # 1. 顶点数组形状与数值检查
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"vertices 应为 (N,2) 形状，当前为 {coords.shape}")
    if np.isnan(coords).any() or np.isinf(coords).any():
        raise ValueError("vertices 含 NaN 或 Inf")
    # 2. 去重与最短边检测
    # 2.1 重复顶点
    unique_coords, inverse_idx = np.unique(coords.round(decimals=12), axis=0, return_inverse=True)
    if len(unique_coords) < len(coords):
        print("警告：存在重复或近似重复顶点，建议先去重")  # 可调用 numpy.unique 或自定义去重逻辑
    # 2.2 最短边
    # 计算所有约束边长度
    lengths = []
    for i, j in segments:
        p, q = coords[i], coords[j]
        dist = np.linalg.norm(p - q)
        lengths.append(dist)
        if dist < eps:
            raise ValueError(f"检测到过短边 (i={i}, j={j})，长度 {dist:.3e} < {eps}")
    # 3. segments 索引范围与自环检查
    for seg in segments:
        i, j = seg
        if not (0 <= i < len(coords) and 0 <= j < len(coords)):
            raise IndexError(f"segment 索引越界：{seg} 超出 [0, {len(coords)-1}] 范围")
        if i == j:
            raise ValueError(f"segment 自环 (i == j == {i}) 不被允许")
    print("Triangle 输入检查通过。")


def dedupe_vertices_and_segments(coords: np.ndarray,
                                  segments: list[list[int]],
                                  decimal_round: int = 8):
    """
    对 coords 去近似重复顶点，并重映射 segments。

    参数
    ----
    coords : (N,2) 浮点数组
        原始顶点坐标。
    segments : List[List[int]]
        原始边界段索引列表，每个元素形如 [i, j]。
    decimal_round : int
        坐标量化的小数位数。距离小于 10^-decimal_round 的顶点会被合并。

    返回
    ----
    unique_coords : (M,2) 浮点数组
        去重后的顶点坐标。
    new_segments : List[List[int]]
        经重映射并去除自环/重复后的段索引列表。
    """
    # 1. 坐标量化与去重
    #    round 后的坐标用于判相等
    quantized = np.round(coords, decimals=decimal_round)
    #    unique: 得到去重后的 coord 以及 inverse_indices
    unique_coords, inverse_indices = np.unique(
        quantized, axis=0, return_inverse=True)

    # 2. 重映射 segments
    remapped = []
    for i, j in segments:
        ni = inverse_indices[i]
        nj = inverse_indices[j]
        # 3. 过滤自环
        if ni != nj:
            # 保证有序以便后面去重（可选）
            if ni < nj:
                remapped.append((ni, nj))
            else:
                remapped.append((nj, ni))

    # 4. 去重 segments
    new_segments = list({seg for seg in remapped})

    # 返回去重后坐标和新 segments
    return unique_coords, new_segments


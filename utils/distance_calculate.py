def perpendicular_distance(start, end, point):
    # 计算点到线段的垂直距离
    if start == end:
        return euclidean_distance(point, start)
    else:
        x0, y0 = point
        x1, y1 = start
        x2, y2 = end
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
        return numerator / denominator


def euclidean_distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
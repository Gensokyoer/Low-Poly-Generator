import numpy as np
import cv2
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, Point


def unweighted_lloyd_relaxation(points, image_size=(512, 512), iterations=10):
    img_w, img_h = image_size
    # print(image_size)
    print(points)
    points = np.array(points)
    image_bounds = Polygon([(0, 0), (img_w, 0), (img_w, img_h), (0, img_h)])


    for _ in range(iterations):
        # 生成Voronoi图
        vor = Voronoi(points)
        #print("Hello")
        new_points = []
        for idx in range(len(vor.points)):
            # 获取Voronoi单元
            region = vor.regions[vor.point_region[idx]]

            # 处理无界区域
            if -1 in region or len(region) == 0:
                # 保持原位置
                new_points.append(vor.points[idx])
                continue

            # 获取多边形顶点
            vertices = vor.vertices[region]

            # 创建多边形并裁剪到图像边界
            poly = Polygon(vertices).intersection(image_bounds)

            # 计算质心
            if poly.is_empty:
                centroid = vor.points[idx]
            else:
                centroid = np.array([poly.centroid.x, poly.centroid.y])

                # 边界约束
                centroid[0] = np.clip(centroid[0], 0, img_w)
                centroid[1] = np.clip(centroid[1], 0, img_h)

            new_points.append(centroid)

        # 更新点集
        points = np.array(new_points)

    return points.tolist()


# 辅助函数：计算多边形面积
def polygon_area(vertices):
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


# 辅助函数：约束点移动范围
def constrain_point(p, img_w, img_h):
    return (max(0, min(p[0], img_w)), max(0, min(p[1], img_h)))




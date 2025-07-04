import cv2
import numpy as np
from cv2.ximgproc import createEdgeDrawing


def edge_detection(img, anchor_threshold=8, scan_interval=4,
                   gradient_threshold=36, blur_size=3, blur_sigma=1.0):
    # 读取图像并转换为灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 对图像进行高斯模糊
    # 确保 blur_size 为奇数
    blur_size = blur_size | 1
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), blur_sigma)

    # 创建 EdgeDrawing 检测器
    ed = createEdgeDrawing()

    # 应用参数设置
    EDParams = cv2.ximgproc_EdgeDrawing_Params()
    EDParams.MinPathLength = 20     # try changing this value between 5 to 1000
    EDParams.PFmode = True         # defaut value try to swich it to True
    EDParams.MinLineLength = 10     # try changing this value between 5 to 100
    EDParams.NFAValidation = True   # defaut value try to swich it to False

    ed.setParams(EDParams)

    # 执行边缘检测，并传入模糊后的图像
    ed.detectEdges(blurred)

    # 获取边缘图像，并添加边界填充（使其与原函数结果对齐）
    edges = ed.getEdgeImage()
    edges = cv2.copyMakeBorder(edges, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    # 获取检测到的边缘线段
    segments = ed.getSegments()

    # 将线段转换为轮廓（序列化连续的坐标点），并过滤较短的边缘
    contours = []
    for seg in segments:
        current_contour = []
        for point in seg:
            # 注意：交换点的顺序 (y, x) 如有需要，可根据实际情况调整
            current_contour.append((point[1], point[0]))

        contours.append(current_contour)

    return contours, edges


# 调用示例
if __name__ == "__main__":
    path = 'D:\\photos\\lena.png'
    contours, edges = edge_detection(path,
                                     anchor_threshold=10,
                                     scan_interval=5,
                                     gradient_threshold=40,
                                     blur_size=7,
                                     blur_sigma=1.5)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

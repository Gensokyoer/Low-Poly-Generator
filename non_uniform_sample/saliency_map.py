import cv2
import numpy as np

def get_saliency_map(image):
    """
    使用静态细粒度显著性检测获取输入图像的显著性图。

    参数:
        image (opencv): 输入图像。

    返回:
        saliency_map (numpy.ndarray): 处理后的显著性图，取值范围为 [0, 255]。
    """
    # 初始化显著性检测器
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

    # 计算显著性图
    success, saliency_map = saliency.computeSaliency(image)

    if not success:
        raise RuntimeError("显著性图计算失败。")

    # 将显著性图缩放到 [0, 255] 范围，并转换为 8 位图像
    saliency_map = (saliency_map * 255).astype(np.uint8)

    return saliency_map

# image_path = 'D:\\photos\\mickey-mouse-1.jpg'
# saliency_map = get_saliency_map(image_path)
# # 应用阈值
# # thresh = 64
# # _, binary_map = cv2.threshold(saliency_map, thresh, 255, cv2.THRESH_BINARY)
# cv2.imshow('Saliency Map', saliency_map)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
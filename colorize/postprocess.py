import cv2
import numpy as np


def postprocess_artistic(image):
    """
    艺术化后处理流程（对应图13效果）
    :param image: 低多边形着色后的BGR图像
    :return: 包含两种处理结果的字典
    """
    results = {}

    # 方法1：RGB空间直方图均衡化（对应图13b）
    # 将元组转换为列表以支持修改
    channels = list(cv2.split(image))  # 关键修正：转换为可变的list
    for i in range(3):  # 对B/G/R三个通道分别处理
        channels[i] = cv2.equalizeHist(channels[i])
    results['rgb_equalized'] = cv2.merge(channels)

    # 方法2：YCrCb空间直方图均衡化（对应图13c）
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = list(cv2.split(ycrcb))  # 转换为可变的list
    channels[0] = cv2.equalizeHist(channels[0])  # 仅处理Y通道
    results['ycrcb_equalized'] = cv2.cvtColor(cv2.merge(channels), cv2.COLOR_YCrCb2BGR)

    # 方法3：LAB空间CLAHE增强
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    # ensure a and b are also 8-bit:
    a_u8 = cv2.convertScaleAbs(a)
    b_u8 = cv2.convertScaleAbs(b)
    # merge back into an 8-bit LAB image:
    lab_u8 = cv2.merge((l_eq, a_u8, b_u8))
    # now correctly convert to BGR:
    results['lab_clahe'] = cv2.cvtColor(lab_u8, cv2.COLOR_LAB2BGR)
    # results['lab_clahe'] = cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)

    return results
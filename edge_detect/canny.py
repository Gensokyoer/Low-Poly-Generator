import cv2
import numpy as np

def edge_detection(image_path):
    """Edge detection using Canny edge detector"""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

    # 使用Canny边缘检测
    edges = cv2.Canny(blurred, 100, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge_image = np.zeros_like(image)
    cv2.drawContours(edge_image, contours, -1, (0, 255, 0), 1)
    cv2.imshow('Edge Drawing', edge_image)

    return contours


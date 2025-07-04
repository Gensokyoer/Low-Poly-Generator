import numpy as np
import cv2

def draw_points(points, img):
    height, width = img.shape[:2]
    after_img = np.zeros((height, width), dtype=np.uint8)
    for (x, y) in points:
        if 0 <= x < width and 0 <= y < height:
            after_img[int(y), int(x)] = 255
    return after_img
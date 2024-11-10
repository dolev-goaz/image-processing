import numpy as np
import cv2

def image_to_grayscale(img: cv2.typing.MatLike):
    h, w = img.shape[:2]
    grayscaled_img = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            b, g, r = img[i, j];
            grayscaled_img[i, j] = int(0.2989 * r + 0.5870 * g + 0.1140 * b)
    return grayscaled_img

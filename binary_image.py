import cv2
import numpy as np
import numpy.matlib

def binary_image(gray_scale: cv2.typing.MatLike):
    black_and_white = 255 * (gray_scale > 128)
    return black_and_white.astype(np.uint8)
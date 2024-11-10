import cv2
import numpy as np
import numpy.matlib

base_matrix = np.array([[51 * 1, 51 * 3],
                        [51 * 4, 51 * 2]])
def dither_image(gray_scale: cv2.typing.MatLike):
    h, w = gray_scale.shape[:2]
    gen_matrix = np.matlib.repmat(base_matrix, int(h / 2), int(w / 2))
    delta = gray_scale - gen_matrix
    dither_image = 255 * (delta > 0)
    return dither_image.astype(np.uint8)